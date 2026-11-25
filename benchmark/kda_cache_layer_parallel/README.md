# Kimi3.1 KDA cache operator tests

Run from the `sgl-kernel-npu` repository root on one idle Ascend NPU with
PyTorch, torch-npu and Triton-Ascend installed:

```sh
python benchmark/kda_cache_layer_parallel/bench.py
```

No SGLang server, model weights, distributed launch, or config file is needed.
The script creates random tensors and calls the two operator wrappers directly.
It loads optimized code from this checkout and the frozen serial-layer baseline
from commit `62f78d1c6478037b708a5108779645ce2fa22520`.

## Fixed test configuration

Model: `KimiK31ForConditionalGeneration` (`model_type=kimi_k31`).
The supplied K3.1 config has the same KDA cache dimensions as the
previous K3 config; its DSA settings do not change these two operators'
test tensors.

The supplied Kimi3.1 config has 93 layers, of which 69 are KDA and 24 are full
attention, 96 KDA heads, head dimension 128, and convolution kernel size 4.
We emulate **one TP=16 shard** (6 heads) on **one NPU**, with 4 draft tokens.
TP and draft count are test choices, not settings inferred from the model config.
TP=16 limits tensor memory for the batch-256 case; it does not require 16 devices.
Only `TEST_TP` and `DRAFT_TOKENS` at the top of `bench.py` need editing if another
test shard or draft depth is desired; all dependent dimensions are derived.

| Tensor | Logical shape | dtype / layout |
| --- | --- | --- |
| Intermediate SSM | `[69, B, 4, 6, 128, 128]` | FP32 / contiguous |
| Destination SSM | `[69, 257, 6, 128, 128]` | FP32 / last two axes transposed |
| Convolution state | `[69, 257, 6, 2304]` | BF16 / contiguous |

SSM FP32 and conv BF16 follow SGLang's default cache types, without environment
overrides. The convolution window is `(4-1)+(4-1)=6`; channels are `3*6*128=2304`.
Shapes are not rounded up or padded to make a test pass. In particular, rollback
uses `tl.arange(0, num_dims)`; support for the non-power-of-two channel count must
be checked with the target Triton-Ascend compiler. Compilation failures are failures,
not skipped or reported as speedups.

Batch sizes are `1, 2, 4, 8, 16, 32, 64, 128, 256`, matching the sweep published in
[PR #831](https://github.com/sgl-project/sgl-kernel-npu/pull/831). That PR does not
publish its complete tensor configuration, so these are not its reproduced results.
The largest case needs about **52 GiB of tensor memory**, plus allocator/compiler
overhead. Allocation errors fail the run rather than silently reducing the workload.

## Checks and timing

- 48 small correctness cases cover both dtypes/layouts, empty batches, skipped
  requests, full acceptance, overlapping rollback and partial V/K tiles.
- Two Kimi3.1-shaped mixed-step cases and every benchmark case compare **both old
  and new code** against an independent PyTorch reference with zero tolerance.
  The entire destination pool, including untouched slots, is checked.
- Performance uses accepted step 0 for every request (maximum rollback), three
  warmups, and 20 alternating-order NPU event samples, following the historical
  timing protocol in [PR #838](https://github.com/sgl-project/sgl-kernel-npu/pull/838).
  Each in-place destination is reset outside the measured interval. JIT, reference
  work and reset copies are excluded. Median time is in microseconds; speedup is
  `old/new`. Event elapsed time can include host enqueue gaps and is not the
  shared profiler's summed NPU task duration.
- `results/kda_cache_layer_parallel/` contains summary CSV, raw sample CSV and
  environment/source metadata. Rerunning replaces these files. Assess large-batch
  regressions as well as small-batch gains; the script assumes no speedup threshold.

Accuracy only on NPU:

```sh
python -m pytest tests/python/sgl_kernel_npu/test_kda_cache_layer_parallel.py -q
```

`TRITON_INTERPRET=1 python benchmark/kda_cache_layer_parallel/bench.py` runs only
the 48 small cases on CPU. It does not validate NPU compilation or performance,
and does not run the full Kimi3.1 shapes. No CI selection files are modified.
