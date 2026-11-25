"""Parameterize the independent-reference tests used by the standalone benchmark."""

import importlib.util
import sys
from pathlib import Path

import pytest

_source = (
    Path(__file__).resolve().parents[3] / "benchmark/kda_cache_layer_parallel/bench.py"
)
_spec = importlib.util.spec_from_file_location("kda_cache_benchmark_tests", _source)
bench = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = bench
_spec.loader.exec_module(bench)


@pytest.fixture(scope="module")
def operators():
    return bench.load_operators()


@pytest.mark.parametrize(
    "op,layer_batch,transposed,dtype", list(bench.correctness_cases())
)
def test_kda_cache_layer_parallel(op, layer_batch, transposed, dtype, operators):
    device, baseline, optimized = operators
    layers, batch = layer_batch
    shape = dict(
        layers=layers,
        drafts=4,
        heads=2,
        value_dim=65,
        key_dim=19,
        window=6,
        channels=32,
    )
    data = bench.generate_data(op, batch, shape, device, dtype, transposed, mixed=True)
    bench.check_accuracy(data, baseline, optimized)


@pytest.mark.parametrize("op,shape,dtype,transposed", bench.MODEL_CASES)
def test_kimi31_cache_shape(op, shape, dtype, transposed, operators):
    device, baseline, optimized = operators
    if device == "cpu":
        pytest.skip(
            "Kimi3.1 integration shapes require the target Triton-Ascend compiler"
        )
    data = bench.generate_data(op, 8, shape, device, dtype, transposed, mixed=True)
    bench.check_accuracy(data, baseline, optimized)
