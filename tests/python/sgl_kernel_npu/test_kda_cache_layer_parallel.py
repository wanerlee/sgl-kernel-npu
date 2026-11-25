"""Correctness coverage for request/layer-parallel KDA state updates.

Run on NPU with unittest or pytest. Set TRITON_INTERPRET=1 to exercise the
same kernels with CPU tensors in Triton's interpreter (not a performance test).
"""

import importlib.util
import itertools
import os
import unittest
from pathlib import Path

import torch


class TestKDACacheLayerParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("TRITON_INTERPRET") == "1":
            cls.device = "cpu"
            # Avoid loading NPU custom extensions when using the CPU interpreter.
            repo = Path(__file__).resolve().parents[3]
            source = (
                repo
                / "python/sgl_kernel_npu/sgl_kernel_npu/mamba"
                / "mamba_state_update_triton.py"
            )
            spec = importlib.util.spec_from_file_location("kda_state_update", source)
            cls.ops = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.ops)
        else:
            try:
                import torch_npu  # noqa: F401
            except ImportError as exc:
                raise unittest.SkipTest("torch_npu is required") from exc
            if not torch.npu.is_available():
                raise unittest.SkipTest("NPU is not available")
            from sgl_kernel_npu.mamba import mamba_state_update_triton

            cls.device = "npu"
            cls.ops = mamba_state_update_triton

    @staticmethod
    def cases():
        return itertools.product(
            [(1, 1), (4, 0), (4, 1), (32, 1), (4, 8), (4, 128)],
            [False, True],
            [torch.float32, torch.bfloat16],
        )

    @staticmethod
    def accepted_steps(batch_size, draft_tokens):
        pattern = [0, -1, 1, draft_tokens - 1]
        return [pattern[i % len(pattern)] for i in range(batch_size)]

    def test_move_intermediate_cache_kda(self):
        for (layers, batch), transposed, dtype in self.cases():
            with self.subTest(
                layers=layers, batch=batch, transposed=transposed, dtype=dtype
            ):
                torch.manual_seed(42)
                pool, drafts, heads, v_dim, k_dim = 2 * batch + 3, 4, 2, 65, 19
                source = torch.randn(layers, pool, drafts, heads, v_dim, k_dim).to(
                    dtype
                )
                shape = (
                    (layers, pool, heads, k_dim, v_dim)
                    if transposed
                    else (layers, pool, heads, v_dim, k_dim)
                )
                storage = torch.randn(shape).to(dtype)
                expected_storage = storage.clone()
                expected = (
                    expected_storage.transpose(-1, -2)
                    if transposed
                    else expected_storage
                )
                dst_indices = list(range(1, 2 * batch + 1, 2))
                src_indices = list(reversed(range(batch)))
                steps = self.accepted_steps(batch, drafts)
                for dst, src, step in zip(dst_indices, src_indices, steps):
                    if step >= 0:
                        expected[:, dst] = source[:, src, step]

                actual_storage = storage.to(self.device).clone()
                actual = (
                    actual_storage.transpose(-1, -2) if transposed else actual_storage
                )
                result = self.ops.move_intermediate_cache_kda(
                    actual,
                    source.to(self.device),
                    torch.tensor(dst_indices, dtype=torch.int32, device=self.device),
                    torch.tensor(src_indices, dtype=torch.int32, device=self.device),
                    torch.tensor(steps, dtype=torch.int32, device=self.device),
                )
                self.assertEqual(result.data_ptr(), actual.data_ptr())
                torch.testing.assert_close(
                    actual_storage.cpu(), expected_storage, rtol=0, atol=0
                )

    def test_conv_state_rollback(self):
        for (layers, batch), transposed, dtype in self.cases():
            with self.subTest(
                layers=layers, batch=batch, transposed=transposed, dtype=dtype
            ):
                torch.manual_seed(43)
                pool, drafts, window, channels = 2 * batch + 3, 4, 6, 32
                shape = (
                    (layers, pool, channels, window)
                    if transposed
                    else (layers, pool, window, channels)
                )
                storage = torch.randn(shape).to(dtype)
                expected_storage = storage.clone()
                expected = (
                    expected_storage.transpose(-1, -2)
                    if transposed
                    else expected_storage
                )
                indices = list(range(1, 2 * batch + 1, 2))
                steps = self.accepted_steps(batch, drafts)
                for idx, step in zip(indices, steps):
                    shift = drafts - 1 - step
                    if step >= 0 and shift > 0:
                        expected[:, idx, shift:] = expected[:, idx, :-shift].clone()

                actual_storage = storage.to(self.device).clone()
                actual = (
                    actual_storage.transpose(-1, -2) if transposed else actual_storage
                )
                result = self.ops.conv_state_rollback(
                    actual,
                    torch.tensor(indices, dtype=torch.int64, device=self.device),
                    torch.tensor(steps, dtype=torch.int64, device=self.device),
                    drafts,
                )
                if batch:
                    self.assertEqual(result.data_ptr(), actual.data_ptr())
                    self.assertEqual(result.stride(), actual.stride())
                torch.testing.assert_close(
                    actual_storage.cpu(), expected_storage, rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
