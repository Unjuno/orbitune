from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "perf_sim.py"
spec = importlib.util.spec_from_file_location("orbitune_perf_sim", MODULE_PATH)
assert spec and spec.loader
perf_sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf_sim)


def _predict(fmt: str, **kwargs):
    model = perf_sim.ModelSpec(context=1024, vocab=1200)
    return perf_sim.predict(
        model,
        fmt=fmt,
        bandwidth_gbs=35.0,
        effective_gops=60.0,
        runtime_overhead_ms=0.35,
        **kwargs,
    )


def test_kv_cache_is_critical_for_autoregressive_inference():
    with_kv = _predict("fp16", kv_cache=True)
    no_kv = _predict("fp16", kv_cache=False)
    assert no_kv["predicted_ms_per_token"] > with_kv["predicted_ms_per_token"] * 100


def test_native_ternary_beats_dequantized_ternary_in_model():
    native = _predict("ternary", kv_cache=True, ternary_mode="native")
    dequant = _predict("ternary", kv_cache=True, ternary_mode="dequant-fp16")
    assert native["predicted_ms_per_token"] < dequant["predicted_ms_per_token"]
    assert native["packed_weight_mb"] == dequant["packed_weight_mb"]


def test_ternary_kernel_efficiency_controls_int8_break_even():
    int8 = _predict("int8", kv_cache=True)
    good = _predict(
        "ternary", kv_cache=True, ternary_mode="native", ternary_kernel_efficiency=0.60
    )
    poor = _predict(
        "ternary", kv_cache=True, ternary_mode="native", ternary_kernel_efficiency=0.30
    )
    assert good["predicted_ms_per_token"] < int8["predicted_ms_per_token"]
    assert poor["predicted_ms_per_token"] > int8["predicted_ms_per_token"]
