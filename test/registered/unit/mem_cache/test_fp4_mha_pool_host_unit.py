"""Unit tests for the FP4 MHA host pool: packed rows must move at their real
width and the block scales must round-trip through L2 with them."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.mha_fp4 import (
    MHATokenToKVPoolFP4Host,
    has_block_scaled_fp4_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FP4_MODULE = "sglang.srt.mem_cache.pool_host.mha_fp4"
MHA_MODULE = "sglang.srt.mem_cache.pool_host.mha"

PAGE_SIZE = 4
LAYER_NUM = 3
HEAD_NUM = 2
HEAD_DIM = 32  # the model's; rows are packed at HEAD_DIM // 2
PAYLOAD_ROW = (HEAD_NUM, HEAD_DIM // 2)
SCALE_ROW = (HEAD_NUM, HEAD_DIM // 16)
PAYLOAD_BYTES = HEAD_NUM * HEAD_DIM // 2
SCALE_BYTES = HEAD_NUM * HEAD_DIM // 16
DEVICE_PAGES = 6
DEVICE_ROWS = DEVICE_PAGES * PAGE_SIZE


def _ptr_key(ptrs: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(ptr) for ptr in ptrs.cpu().tolist())


def _random_layers(row_shape):
    return [
        torch.randint(0, 256, (DEVICE_ROWS, *row_shape), dtype=torch.uint8)
        for _ in range(LAYER_NUM)
    ]


def _make_device_pool(**overrides):
    """A CPU stand-in with the buffers NVFP4KVCacheMethod.create_buffers makes
    for XQA: packed rows plus token-linear block scales."""
    pool = SimpleNamespace(
        is_quantized_kv_cache=True,
        k_buffer=_random_layers(PAYLOAD_ROW),
        v_buffer=_random_layers(PAYLOAD_ROW),
        k_scale_buffer=_random_layers(SCALE_ROW),
        v_scale_buffer=_random_layers(SCALE_ROW),
        native_k_scale_buffer=None,
        native_v_scale_buffer=None,
        store_dtype=torch.uint8,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        row_dim=HEAD_NUM * HEAD_DIM,
        layer_num=LAYER_NUM,
        start_layer=0,
        end_layer=LAYER_NUM - 1,
        size=DEVICE_ROWS - PAGE_SIZE,
        page_size=PAGE_SIZE,
        device="cpu",
        use_hnd=False,
        layer_shard_enabled=False,
        hicache_write_back_staging=None,
    )
    pool.k_data_ptrs = torch.tensor(
        [buf.data_ptr() for buf in pool.k_buffer], dtype=torch.uint64
    )
    pool.v_data_ptrs = torch.tensor(
        [buf.data_ptr() for buf in pool.v_buffer], dtype=torch.uint64
    )
    for name, value in overrides.items():
        setattr(pool, name, value)
    return pool


def _make_host(device_pool, **kwargs):
    """Run the real constructor on CPU: unpinned host memory, kernels reported
    as available so the staging buffers and scale buffers are built."""
    with (
        mock.patch(f"{FP4_MODULE}._is_cuda", True),
        mock.patch(f"{MHA_MODULE}._is_cuda", True),
        mock.patch(f"{FP4_MODULE}.can_use_write_back_jit_kernel", return_value=True),
        mock.patch(f"{FP4_MODULE}.can_use_hicache_jit_kernel", return_value=True),
        mock.patch(f"{MHA_MODULE}.can_use_hicache_jit_kernel", return_value=True),
    ):
        return MHATokenToKVPoolFP4Host(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PAGE_SIZE,
            layout="page_first",
            pin_memory=False,
            **kwargs,
        )


def _cpu_staged_lf_pf_copy(
    registry,
    *,
    k_ptr_src,
    v_ptr_src,
    src_indices,
    dst_indices,
    staging_k,
    staging_v,
    dst_k,
    dst_v,
    page_size,
    **_,
):
    """CPU stand-in for the staged D2H kernel. Like the kernel, it takes the
    row width from the staging buffer, not from the source tensors."""
    for ptrs, staging, dst in ((k_ptr_src, staging_k, dst_k), (v_ptr_src, staging_v, dst_v)):
        row_bytes = staging[0, 0].numel()
        for layer_id, src in enumerate(registry[_ptr_key(ptrs)]):
            rows = src.reshape(-1).view(-1, row_bytes)[src_indices]
            dst[dst_indices, layer_id] = rows.view(-1, *dst.shape[2:])


def _cpu_one_layer_copy(
    *,
    k_cache_dst,
    v_cache_dst,
    indices_dst,
    k_cache_src,
    v_cache_src,
    indices_src,
    element_dim,
    **_,
):
    """CPU stand-in for the H2D kernel: both sides are viewed at element_dim."""
    for dst, src in ((k_cache_dst, k_cache_src), (v_cache_dst, v_cache_src)):
        dst.view(-1, element_dim)[indices_dst] = src.reshape(-1, element_dim)[
            indices_src
        ]


class TestFP4MHATokenToKVPoolHost(CustomTestCase):
    def test_factory_selects_fp4_host_pool(self):
        plain_pool = SimpleNamespace(head_dim=4, v_head_dim=4)
        unscaled_quantized_pool = SimpleNamespace(
            is_quantized_kv_cache=True,
            k_scale_buffer=None,
            native_k_scale_buffer=None,
            head_dim=4,
            v_head_dim=4,
        )

        self.assertIs(get_mha_host_pool_cls(_make_device_pool()), MHATokenToKVPoolFP4Host)
        self.assertIs(get_mha_host_pool_cls(plain_pool), MHATokenToKVPoolHost)
        self.assertIs(
            get_mha_host_pool_cls(unscaled_quantized_pool), MHATokenToKVPoolHost
        )
        self.assertFalse(has_block_scaled_fp4_rows(plain_pool))

    def test_geometry_follows_packed_rows(self):
        host = _make_host(_make_device_pool())

        self.assertEqual((host.head_num, host.head_dim), PAYLOAD_ROW)
        self.assertEqual(host.token_stride_size, PAYLOAD_BYTES)
        self.assertEqual(host.element_dim, PAYLOAD_BYTES)
        self.assertEqual(
            host.get_size_per_token(), 2 * LAYER_NUM * (PAYLOAD_BYTES + SCALE_BYTES)
        )
        self.assertEqual(
            tuple(host.kv_buffer.shape), (2, host.size, LAYER_NUM, *PAYLOAD_ROW)
        )
        self.assertEqual(
            tuple(host.scale_kv_buffer.shape), (2, host.size, LAYER_NUM, *SCALE_ROW)
        )
        self.assertEqual(tuple(host.staging_k_buffer.shape[1:]), (LAYER_NUM, *PAYLOAD_ROW))
        self.assertEqual(
            tuple(host.scale_staging_k_buffer.shape[1:]), (LAYER_NUM, *SCALE_ROW)
        )
        self.assertTrue(host.can_use_write_back_jit)

    def test_unsupported_configurations_are_rejected(self):
        draft_pool = SimpleNamespace()
        cases = {
            "layout": dict(pool=_make_device_pool(), layout="layer_first"),
            "packed draft": dict(
                pool=_make_device_pool(), mtp_draft_device_pools=(draft_pool,)
            ),
            "native scales": dict(
                pool=_make_device_pool(native_k_scale_buffer=[torch.zeros(1)])
            ),
            "no linear scales": dict(
                pool=_make_device_pool(
                    k_scale_buffer=None, native_k_scale_buffer=[torch.zeros(1)]
                )
            ),
            "hnd": dict(pool=_make_device_pool(use_hnd=True)),
        }
        for name, case in cases.items():
            with self.subTest(name), self.assertRaises(NotImplementedError):
                pool = case.pop("pool")
                if "layout" in case:
                    MHATokenToKVPoolFP4Host(
                        pool, 2.0, 0, PAGE_SIZE, case["layout"], pin_memory=False
                    )
                else:
                    _make_host(pool, **case)

    def test_rows_and_scales_round_trip_device_host_device(self):
        device_pool = _make_device_pool()
        host = _make_host(device_pool)
        buffers = {
            "k": device_pool.k_buffer,
            "v": device_pool.v_buffer,
            "k_scale": device_pool.k_scale_buffer,
            "v_scale": device_pool.v_scale_buffer,
        }
        original = {name: [b.clone() for b in bufs] for name, bufs in buffers.items()}
        registry = {
            _ptr_key(device_pool.k_data_ptrs): device_pool.k_buffer,
            _ptr_key(device_pool.v_data_ptrs): device_pool.v_buffer,
            _ptr_key(host.k_scale_device_ptrs): device_pool.k_scale_buffer,
            _ptr_key(host.v_scale_device_ptrs): device_pool.v_scale_buffer,
        }
        # Device pages 4 and 5 are in the upper half of the pool: at twice the
        # row width they would be read past the end of each layer's buffer.
        saved = torch.arange(4 * PAGE_SIZE, 6 * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.cat(
            [torch.arange(PAGE_SIZE, 2 * PAGE_SIZE), torch.arange(0, PAGE_SIZE)]
        ).to(torch.int64)
        # Restore into other slots, as after an eviction.
        restored = torch.arange(1 * PAGE_SIZE, 3 * PAGE_SIZE, dtype=torch.int64)

        with (
            mock.patch(
                f"{MHA_MODULE}.jit_transfer_hicache_all_layer_staged_lf_pf",
                side_effect=lambda **kw: _cpu_staged_lf_pf_copy(registry, **kw),
            ) as payload_backup,
            mock.patch(
                f"{FP4_MODULE}.jit_transfer_hicache_all_layer_staged_lf_pf",
                side_effect=lambda **kw: _cpu_staged_lf_pf_copy(registry, **kw),
            ) as scale_backup,
            mock.patch(
                f"{MHA_MODULE}.jit_transfer_hicache_one_layer",
                side_effect=_cpu_one_layer_copy,
            ) as payload_load,
            mock.patch(
                f"{FP4_MODULE}.jit_transfer_hicache_one_layer",
                side_effect=_cpu_one_layer_copy,
            ) as scale_load,
        ):
            host.backup_from_device_all_layer(
                device_pool, host_indices, saved, io_backend="kernel"
            )
            # Device page 4 landed in host page 1, device page 5 in host page 0.
            for layer in range(LAYER_NUM):
                self.assertTrue(
                    torch.equal(
                        host.k_buffer[:PAGE_SIZE, layer],
                        original["k"][layer][5 * PAGE_SIZE : 6 * PAGE_SIZE],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        host.v_scale_host[PAGE_SIZE : 2 * PAGE_SIZE, layer],
                        original["v_scale"][layer][4 * PAGE_SIZE : 5 * PAGE_SIZE],
                    )
                )

            for bufs in buffers.values():
                for buf in bufs:
                    buf.zero_()
            for layer_id in range(LAYER_NUM):
                host.load_to_device_per_layer(
                    device_pool, host_indices, restored, layer_id, io_backend="kernel"
                )

        payload_backup.assert_called_once()
        scale_backup.assert_called_once()
        self.assertEqual(payload_load.call_count, LAYER_NUM)
        self.assertEqual(scale_load.call_count, LAYER_NUM)
        untouched = torch.ones(DEVICE_ROWS, dtype=torch.bool)
        untouched[restored] = False
        for name, bufs in buffers.items():
            for layer, buf in enumerate(bufs):
                self.assertTrue(
                    torch.equal(buf[restored], original[name][layer][saved]),
                    f"{name} layer {layer}",
                )
                self.assertTrue(torch.all(buf[untouched] == 0), f"{name} layer {layer}")

    def test_scales_fall_back_to_the_aot_transfer(self):
        device_pool = _make_device_pool()
        host = _make_host(device_pool)
        host.can_use_scale_jit = False
        indices = torch.arange(PAGE_SIZE, dtype=torch.int64)

        with (
            mock.patch.object(MHATokenToKVPoolHost, "load_to_device_per_layer"),
            mock.patch(
                f"{FP4_MODULE}.transfer_kv_per_layer_pf_lf", create=True
            ) as aot_transfer,
            mock.patch(f"{FP4_MODULE}.jit_transfer_hicache_one_layer") as jit_transfer,
        ):
            host.load_to_device_per_layer(
                device_pool, indices, indices, 1, io_backend="kernel"
            )

        jit_transfer.assert_not_called()
        kwargs = aot_transfer.call_args.kwargs
        self.assertEqual(kwargs["item_size"], SCALE_BYTES)
        self.assertEqual(kwargs["src_layout_dim"], SCALE_BYTES * LAYER_NUM)
        self.assertEqual(kwargs["layer_id"], 1)
        self.assertIs(kwargs["dst_k"], device_pool.k_scale_buffer[1])

    def test_storage_pages_are_rejected(self):
        host = MHATokenToKVPoolFP4Host.__new__(MHATokenToKVPoolFP4Host)
        with self.assertRaises(NotImplementedError):
            host.get_dummy_flat_data_page()
        with self.assertRaises(NotImplementedError):
            host.get_data_page(0)


if __name__ == "__main__":
    unittest.main()
