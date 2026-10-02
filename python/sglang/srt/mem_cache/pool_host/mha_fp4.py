"""Host (L2) pool for a block-scaled FP4 MHA KV cache (NVFP4, FP4-MX block 16).

The device pool packs each K/V row at half the model's head_dim and keeps one
scale byte per 16 values in separate token-linear buffers. Both must reach host
memory: a packed row restored under another token's scales dequantizes to
garbage. This subclass sizes the payload from the device buffers' real row
shape and adds host buffers for the scales, which move with the same kernels
as the payload at their own row width.

The dequant workspace is not backed up: prefill refills it from the packed
rows and scales on every forward.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_jit_kernel,
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_staged_lf_pf as jit_transfer_hicache_all_layer_staged_lf_pf,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import _WRITE_BACK_STAGING_PAGE_CHUNK
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)

if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_per_layer_pf_lf


def has_block_scaled_fp4_rows(device_pool) -> bool:
    """Whether an MHA device pool stores quantized rows with separate scales."""
    return bool(getattr(device_pool, "is_quantized_kv_cache", False)) and (
        getattr(device_pool, "k_scale_buffer", None) is not None
        or getattr(device_pool, "native_k_scale_buffer", None) is not None
    )


class MHATokenToKVPoolFP4Host(MHATokenToKVPoolHost):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence = (),
        pool_label: str = "kv",
    ):
        if layout != "page_first":
            raise NotImplementedError(
                f"FP4 KV host pool supports only the page_first layout, got {layout!r}."
            )
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "FP4 KV host pool does not pack MTP draft KV layers: their rows "
                "are not the width of the packed target rows. The draft gets a "
                "sidecar host pool instead."
            )
        if (
            device_pool.k_scale_buffer is None
            or device_pool.native_k_scale_buffer is not None
        ):
            raise NotImplementedError(
                "FP4 KV host pool supports only token-linear block scales (the "
                "XQA layout on SM90/SM120); the TRT-LLM GenMHA scale layout is "
                "not backed up. Disable the hierarchical cache."
            )
        if device_pool.use_hnd or device_pool.layer_shard_enabled:
            raise NotImplementedError(
                "FP4 KV host pool supports neither the HND KV layout nor "
                "layer-sharded device pools."
            )
        self.payload_row_shape = tuple(device_pool.k_buffer[0].shape[1:])
        self.scale_row_shape = tuple(device_pool.k_scale_buffer[0].shape[1:])
        if (
            len(self.payload_row_shape) != 2
            or tuple(device_pool.v_buffer[0].shape[1:]) != self.payload_row_shape
            or tuple(device_pool.v_scale_buffer[0].shape[1:]) != self.scale_row_shape
        ):
            raise NotImplementedError(
                "FP4 KV host pool needs [token, head, dim] rows of one shape for "
                "K and V."
            )
        self.scale_kv_buffer: torch.Tensor | None = None
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            pool_label=pool_label,
        )
        if not self.can_use_write_back_jit:
            raise NotImplementedError(
                "FP4 KV host pool needs the staged JIT write-back kernel "
                "(io_backend='kernel', page_first layout, CUDA or HIP)."
            )
        self._init_scale_buffers()

    def get_size_per_token(self):
        # The packed row, not the model's head_dim, is what the kernels copy.
        self.head_num, self.head_dim = self.payload_row_shape
        self.layer_num = self.target_layer_num
        self.scale_row_dim = int(np.prod(self.scale_row_shape))
        self.scale_row_bytes = self.scale_row_dim * self.dtype.itemsize
        payload_row_bytes = self.head_num * self.head_dim * self.dtype.itemsize
        return (payload_row_bytes + self.scale_row_bytes) * self.layer_num * 2

    def _init_write_back_staging_buffers(self):
        # prepare_mha_write_back_staging shapes its buffers from the model's
        # head_dim, so the packed payload and the scales get their own here.
        self.staging_page_capacity = 0
        self.staging_token_capacity = 0
        self.staging_k_buffer = None
        self.staging_v_buffer = None
        self.scale_staging_k_buffer = None
        self.scale_staging_v_buffer = None
        self.can_use_write_back_jit = False
        if not (_is_cuda or _is_hip) or not all(
            can_use_write_back_jit_kernel(element_size=size)
            for size in (self.token_stride_size, self.scale_row_bytes)
        ):
            return
        page_capacity = min(self.page_num, _WRITE_BACK_STAGING_PAGE_CHUNK)
        token_capacity = page_capacity * self.page_size
        (
            self.staging_k_buffer,
            self.staging_v_buffer,
            self.scale_staging_k_buffer,
            self.scale_staging_v_buffer,
        ) = (
            torch.empty(
                (token_capacity, self.layer_num, *row_shape),
                dtype=self.dtype,
                device=self.device_pool.device,
            )
            for row_shape in (self.payload_row_shape, self.scale_row_shape)
            for _ in range(2)
        )
        self.can_use_write_back_jit = True
        self.staging_page_capacity = page_capacity
        self.staging_token_capacity = token_capacity

    def _init_scale_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        # Page-first like the payload: [K/V, token, layer, head, blocks].
        self.scale_layout_dim = self.scale_row_bytes * self.layer_num
        self.scale_kv_buffer = alloc_func(
            (2, self.size, self.layer_num, *self.scale_row_shape),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=self.page_size * self.scale_layout_dim,
        )
        self.k_scale_host = self.scale_kv_buffer[0]
        self.v_scale_host = self.scale_kv_buffer[1]
        # [token, layer, ...] -> per-layer strided [token, ...] views for H2D.
        self.k_scale_host_layers = list(self.k_scale_host.transpose(0, 1))
        self.v_scale_host_layers = list(self.v_scale_host.transpose(0, 1))
        self.k_scale_device_ptrs = torch.tensor(
            [buf.data_ptr() for buf in self.device_pool.k_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_scale_device_ptrs = torch.tensor(
            [buf.data_ptr() for buf in self.device_pool.v_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        # A scale row is narrower than the register kernel's 128-byte copy
        # round; without the TMA kernel the AOT transfer moves it instead.
        self.can_use_scale_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
            page_size=self.page_size,
            element_size=self.scale_row_bytes,
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if io_backend != "kernel":
            raise NotImplementedError(
                f"FP4 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        jit_transfer_hicache_all_layer_staged_lf_pf(
            k_ptr_src=self.k_scale_device_ptrs,
            v_ptr_src=self.v_scale_device_ptrs,
            src_indices=device_indices,
            dst_indices=host_indices,
            staging_k=self.scale_staging_k_buffer,
            staging_v=self.scale_staging_v_buffer,
            dst_k=self.k_scale_host,
            dst_v=self.v_scale_host,
            page_size=self.page_size,
        )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise NotImplementedError("FP4 KV host pool has no draft layers.")
        if io_backend != "kernel":
            raise NotImplementedError(
                f"FP4 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        if self.can_use_scale_jit:
            jit_transfer_hicache_one_layer(
                page_size=self.page_size,
                k_cache_dst=device_pool.k_scale_buffer[layer_id],
                v_cache_dst=device_pool.v_scale_buffer[layer_id],
                k_cache_src=self.k_scale_host_layers[layer_id],
                v_cache_src=self.v_scale_host_layers[layer_id],
                indices_dst=device_indices,
                indices_src=host_indices,
                element_dim=self.scale_row_dim,
            )
        else:
            transfer_kv_per_layer_pf_lf(
                src_k=self.k_scale_host,
                dst_k=device_pool.k_scale_buffer[layer_id],
                src_v=self.v_scale_host,
                dst_v=device_pool.v_scale_buffer[layer_id],
                src_indices=host_indices,
                dst_indices=device_indices,
                layer_id=layer_id,
                item_size=self.scale_row_bytes,
                src_layout_dim=self.scale_layout_dim,
            )

    def destroy(self):
        if (
            self.scale_kv_buffer is not None
            and self.pin_memory
            and (_is_cuda or _is_hip)
        ):
            _cuda_host_unregister(self.scale_kv_buffer)
        self.scale_kv_buffer = None
        self.k_scale_host = None
        self.v_scale_host = None
        super().destroy()

    def _storage_pages_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "FP4 KV host pool does not expose flat storage (L3) pages yet: "
            "a page's block scales live outside kv_buffer."
        )

    def get_hybrid_pool_buffer(self):
        raise self._storage_pages_unsupported()

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._storage_pages_unsupported()

    def get_page_buffer_meta(self, indices):
        raise self._storage_pages_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise self._storage_pages_unsupported()
