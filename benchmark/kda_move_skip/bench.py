"""Measure the SECOND KDA Move with mixed steps and all steps equal to -1.

Run from the checkout: python -u benchmark/kda_move_skip/bench.py
Defaults: logical NPU 0, batch 1..256, three warmups, 20 alternating samples.
Optional: --device 1 or --batches 1 8 32. No package install/model server needed;
the image must already provide torch, torch_npu and Triton-Ascend.

Reset and the first Move run outside timing. Report NPU-event elapsed time per
second wrapper call (including possible host enqueue gaps), not profiler task
duration. Results and raw samples go to a new results/kda_move_skip/<run>/ folder.
TRITON_INTERPRET=1 runs reduced-shape CPU correctness checks only, never timings.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

BASELINE_COMMIT = "458e72caf6cdfb5b9c06711e6f84d134d15e3f97"
REPO = Path(__file__).resolve().parents[2]
KERNEL = "python/sgl_kernel_npu/sgl_kernel_npu/implementation/triton/mamba/mamba_state_update.py"
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WARMUP, SAMPLES = 3, 20
# Kimi K3.1 test shard: TP=16, 96 total heads, 128x128 state, four drafts.
LAYERS, HEADS, VALUE_DIM, KEY_DIM, DRAFTS = 69, 6, 128, 128, 4


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_case(
    batch, device, *, layers=LAYERS, heads=HEADS, vdim=VALUE_DIM, kdim=KEY_DIM
):
    torch.manual_seed(42)
    pool = 2 * batch + 3  # Disjoint current/track slots plus untouched slots.
    initial = torch.randn(
        layers, pool, heads, kdim, vdim, dtype=torch.float32, device=device
    ).transpose(-1, -2)
    state = torch.empty_like(initial)
    source = torch.randn(
        layers, batch, DRAFTS, heads, vdim, kdim, dtype=torch.float32, device=device
    )
    src = torch.arange(batch, dtype=torch.int32, device=device)
    current_slots, track_slots = src, src + batch + 1
    current_steps = torch.full_like(src, DRAFTS - 1)
    track_steps = torch.empty_like(src)
    return dict(
        batch=batch,
        pool=pool,
        initial=initial,
        state=state,
        source=source,
        first=(state, source, current_slots, src, current_steps),
        second=(state, source, track_slots, src, track_steps),
    )


def set_pattern(data, pattern):
    batch = data["batch"]
    if pattern == "mixed":
        if batch < 2:
            raise ValueError("A mixed batch needs at least two requests")
        values = [(i // 2) % DRAFTS if i % 2 == 0 else -1 for i in range(batch)]
    elif pattern == "all_skip":
        values = [-1] * batch
    else:
        raise ValueError(pattern)
    steps = data["second"][-1]
    steps.copy_(torch.tensor(values, dtype=steps.dtype, device=steps.device))
    data["valid_rows"] = [i for i, step in enumerate(values) if step >= 0]
    data["valid_steps"] = [step for step in values if step >= 0]
    data["pattern"] = pattern


def prepare_first(data, module):
    data["state"].copy_(data["initial"])
    module.move_intermediate_cache_kda(*data["first"])


def check_accuracy(data, module):
    prepare_first(data, module)
    result = module.move_intermediate_cache_kda(*data["second"])
    assert result.data_ptr() == data["state"].data_ptr()
    assert result.stride() == data["initial"].stride()
    device = data["state"].device
    rows = torch.tensor(data["valid_rows"], dtype=torch.int64, device=device)
    steps = torch.tensor(data["valid_steps"], dtype=torch.int64, device=device)
    # One-layer reference bounds memory; compare ALL layers and pool slots.
    for layer in range(data["state"].shape[0]):
        expected = data["initial"][layer].clone()
        expected[: data["batch"]] = data["source"][layer, :, DRAFTS - 1]
        if data["valid_rows"]:
            expected[rows + data["batch"] + 1] = data["source"][layer, rows, steps]
        for start in range(0, data["pool"], 8):
            torch.testing.assert_close(
                data["state"][layer, start : start + 8],
                expected[start : start + 8],
                atol=0,
                rtol=0,
            )


def measure(data, modules):
    for _ in range(WARMUP):
        for module in modules.values():
            prepare_first(data, module)
            module.move_intermediate_cache_kda(*data["second"])
    torch.npu.synchronize()
    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
        enable_timing=True
    )
    records = []
    for sample in range(SAMPLES):
        order = ("old", "new") if sample % 2 == 0 else ("new", "old")
        record = {"sample": sample, "first": order[0]}
        for name in order:
            module = modules[name]
            prepare_first(data, module)
            torch.npu.synchronize()
            begin.record()
            module.move_intermediate_cache_kda(*data["second"])
            end.record()
            end.synchronize()
            elapsed = begin.elapsed_time(end) * 1000.0
            if elapsed <= 0:
                raise RuntimeError("Nonpositive NPU event duration")
            record[name + "_us"] = elapsed
        records.append(record)
    return records


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not reported"


def write_environment(output, args, modules):
    import torch_npu

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "not reported"
    metadata = dict(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        chip=torch.npu.get_device_name(args.device),
        logical_device_index=args.device,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        triton_ascend=package_version("triton-ascend"),
        cann_version=os.environ.get("CANN_VERSION", "not reported"),
        validation_commit=revision,
        baseline_commit=BASELINE_COMMIT,
        source_paths={
            name: str(Path(module.__file__).resolve())
            for name, module in modules.items()
        },
        source_sha256={
            name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in modules.items()
        },
        layers=LAYERS,
        local_heads=HEADS,
        test_tp=16,
        key_dim=KEY_DIM,
        value_dim=VALUE_DIM,
        draft_tokens=DRAFTS,
        dtype="torch.float32",
        destination_layout="transposed",
        pool="2 * batch + 3",
        batches=args.batches,
        warmup=WARMUP,
        samples=SAMPLES,
        metric="npu_event_elapsed_us_per_second_move_wrapper_call",
        statistic="median",
        preparation="reset destination; run first Move with the same implementation; synchronize; time only second Move",
    )
    (output / "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", type=int, default=0, help="visible logical NPU index (default: 0)"
    )
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    args = parser.parse_args()
    if args.device < 0 or any(batch < 1 for batch in args.batches):
        parser.error("device must be nonnegative and batches must be positive")
    args.batches = list(dict.fromkeys(args.batches))
    cpu = os.environ.get("TRITON_INTERPRET") == "1"
    if cpu:
        print(
            "CPU interpreter: reduced-shape correctness only; no timings.", flush=True
        )
        device = "cpu"
    else:
        import torch_npu  # noqa: F401

        print(f"Initializing logical npu:{args.device} ...", flush=True)
        torch.npu.set_device(args.device)
        device = f"npu:{args.device}"
        print(f"Device: {torch.npu.get_device_name(args.device)}", flush=True)
    modules = {
        "old": load_source(
            "kda_move_skip_baseline", Path(__file__).with_name("baseline.py")
        ),
        "new": load_source("kda_move_skip_optimized", REPO / KERNEL),
    }
    if cpu:
        for batch in (1, 2, 8, 32):
            data = make_case(batch, device, layers=3, heads=2, vdim=65, kdim=19)
            for pattern in ("mixed", "all_skip"):
                if pattern == "mixed" and batch == 1:
                    continue
                set_pattern(data, pattern)
                for module in modules.values():
                    check_accuracy(data, module)
                print(f"PASS: batch={batch}, {pattern}, old and new", flush=True)
        return

    run = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = REPO / "results/kda_move_skip" / run
    output.mkdir(parents=True, exist_ok=False)
    write_environment(output, args, modules)
    print(f"Results: {output}", flush=True)
    fields = [
        "pattern",
        "batch",
        "valid",
        "skipped",
        "pool",
        "accuracy",
        "old_us",
        "new_us",
        "delta_us",
        "speedup",
        "old_min_us",
        "old_max_us",
        "new_min_us",
        "new_max_us",
    ]
    with (output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as summary_file, (output / "samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as samples_file:
        summary = csv.DictWriter(summary_file, fieldnames=fields)
        raw = csv.DictWriter(
            samples_file,
            fieldnames=["pattern", "batch", "sample", "first", "old_us", "new_us"],
        )
        summary.writeheader()
        raw.writeheader()
        for batch in args.batches:
            size_gib = (
                LAYERS
                * HEADS
                * VALUE_DIM
                * KEY_DIM
                * 4
                * (2 * (2 * batch + 3) + DRAFTS * batch)
                / 2**30
            )
            print(
                f"Allocating batch={batch}: about {size_gib:.2f} GiB for source and destination buffers (plus check temporaries).",
                flush=True,
            )
            data = make_case(batch, device)
            for pattern in ("mixed", "all_skip"):
                if pattern == "mixed" and batch == 1:
                    continue
                set_pattern(data, pattern)
                valid = len(data["valid_rows"])
                print(
                    f"Checking batch={batch}, {pattern}: valid={valid}, skipped={batch - valid} ...",
                    flush=True,
                )
                for module in modules.values():
                    check_accuracy(data, module)
                print(
                    f"Accuracy passed; warming up and collecting {SAMPLES} alternating samples ...",
                    flush=True,
                )
                records = measure(data, modules)
                old, new = (
                    statistics.median(row[name + "_us"] for row in records)
                    for name in ("old", "new")
                )
                summary.writerow(
                    dict(
                        pattern=pattern,
                        batch=batch,
                        valid=valid,
                        skipped=batch - valid,
                        pool=data["pool"],
                        accuracy="passed",
                        old_us=old,
                        new_us=new,
                        delta_us=new - old,
                        speedup=old / new,
                        **{
                            name
                            + "_"
                            + label
                            + "_us": fn(row[name + "_us"] for row in records)
                            for name in ("old", "new")
                            for label, fn in (("min", min), ("max", max))
                        },
                    )
                )
                raw.writerows(
                    dict(pattern=pattern, batch=batch, **row) for row in records
                )
                summary_file.flush()
                samples_file.flush()
                print(
                    f"{pattern} batch={batch}: old={old:.3f} us, new={new:.3f} us, delta={new - old:+.3f} us, speedup={old / new:.3f}x",
                    flush=True,
                )
            del data
        print(f"Done. Summary: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
