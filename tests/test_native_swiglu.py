"""Offline adapter/contracts. Fake objects are not kernel correctness evidence."""
from types import SimpleNamespace

import pytest

import ollm.runtime.native_swiglu as native


class Tensor:
    def __init__(self, shape, dtype="mlx.core.bfloat16"):
        self.shape, self.dtype = shape, dtype


class Projection(dict):
    def __init__(self):
        super().__init__()
        self.mode, self.bits, self.group_size = "affine", 4, 64
        self.weight = Tensor((17, 16), "mlx.core.uint32")
        self.scales = Tensor((17, 2))
        self.biases = Tensor((17, 2))


class MLP:
    def __init__(self):
        self.gate_proj, self.up_proj = Projection(), Projection()
        self.down_proj = lambda activation: ("down", activation)

    def __call__(self, x):
        return "stock"


class Model:
    def __init__(self):
        self.model_type = "qwen3"
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=MLP()) for _ in range(2)])


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(native, "_require_reserve", lambda: None)
    monkeypatch.setattr(native, "_require_versions", lambda: None)
    monkeypatch.setattr(native, "_types", lambda: (
        SimpleNamespace(Module=object, QuantizedLinear=Projection),
        SimpleNamespace(Model=Model, MLP=MLP)))
    monkeypatch.setattr(native, "_activation", lambda *args: "synthetic-activation")
    return Model()


@pytest.mark.parametrize("rows,width,outputs,variant,expected", [
    (1, 128, 17, "simd_decode", "simd"),
    (7, 128, 17, "tiled_prefill", "tiled"),
    (512, 1024, 3072, "fused_both", "tiled"),
    (1, 128, 17, "tiled_prefill", None),
    (7, 128, 17, "simd_decode", None),
    (513, 128, 17, "fused_both", None),
    (1, 127, 17, "fused_both", None),
    (512, 8192, 8192, "fused_both", None),
    (0, 128, 17, "fused_both", None),
])
def test_phase_and_byte_bounds(rows, width, outputs, variant, expected):
    assert native.select_kernel(rows, width, outputs, variant) == expected


def test_unknown_variant_and_boolean_dimensions_rejected():
    with pytest.raises(ValueError):
        native.select_kernel(1, 128, 17, "auto")
    with pytest.raises(ValueError):
        native.select_kernel(True, 128, 17, "fused_both")


def test_disabled_default_does_not_import_dispatch_or_mutate(monkeypatch):
    def forbidden():
        pytest.fail("disabled experiment must not touch hardware or dependencies")
    monkeypatch.setattr(native, "_types", forbidden)
    monkeypatch.setattr(native, "_require_reserve", forbidden)
    model = Model()
    originals = [layer.mlp for layer in model.model.layers]
    with native.experimental_qwen3_swiglu(model) as stats:
        assert stats.installed_layers == 0
        assert [layer.mlp for layer in model.model.layers] == originals


def test_candidate_retains_original_packed_parameters_and_restores(fake_runtime):
    model = fake_runtime
    originals = [layer.mlp for layer in model.model.layers]
    with native.experimental_qwen3_swiglu(model, enabled=True) as stats:
        assert stats.installed_layers == 2
        for layer, original in zip(model.model.layers, originals):
            assert layer.mlp.original is original
            assert layer.mlp.original.gate_proj.weight is original.gate_proj.weight
            assert layer.mlp(Tensor((1, 7, 128))) == ("down", "synthetic-activation")
            assert layer.mlp(Tensor((1, 1, 128))) == ("down", "synthetic-activation")
        assert stats.tiled_graph_calls == stats.simd_graph_calls == 2
        assert stats.theoretical_intermediate_bytes_eliminated == 4 * 8 * 17 * 2
    assert [layer.mlp for layer in model.model.layers] == originals


@pytest.mark.parametrize("shape,dtype", [
    ((2, 1, 128), "mlx.core.bfloat16"),
    ((1, 513, 128), "mlx.core.bfloat16"),
    ((1, 7, 128), "mlx.core.float32"),
    ((7, 128), "mlx.core.bfloat16"),
])
def test_unsupported_calls_fall_back_before_native_work(fake_runtime, monkeypatch, shape, dtype):
    def forbidden(*args):
        pytest.fail("unsupported input must not reach native submission")
    monkeypatch.setattr(native, "_activation", forbidden)
    with native.experimental_qwen3_swiglu(fake_runtime, enabled=True) as stats:
        assert fake_runtime.model.layers[0].mlp(Tensor(shape, dtype)) == "stock"
        assert stats.fallback_calls == 1
        assert stats.simd_graph_calls == stats.tiled_graph_calls == 0


@pytest.mark.parametrize("mutation", [
    lambda mlp: setattr(mlp.up_proj, "bits", 8),
    lambda mlp: setattr(mlp.up_proj, "mode", "mxfp4"),
    lambda mlp: mlp.gate_proj.update(bias="unsupported"),
    lambda mlp: setattr(mlp.up_proj, "scales", Tensor((17, 1))),
    lambda mlp: setattr(mlp.gate_proj, "biases", None),
    lambda mlp: setattr(mlp.up_proj, "weight", Tensor((18, 16), "mlx.core.uint32")),
    lambda mlp: setattr(mlp.gate_proj, "scales", Tensor((17, 2), "mlx.core.float32")),
])
def test_all_layers_validated_before_any_substitution(fake_runtime, mutation):
    model = fake_runtime
    originals = [layer.mlp for layer in model.model.layers]
    mutation(originals[-1])
    with pytest.raises(ValueError):
        with native.experimental_qwen3_swiglu(model, enabled=True):
            pytest.fail("invalid layer must prevent installation")
    assert [layer.mlp for layer in model.model.layers] == originals


def test_native_failure_propagates_without_state_splice_and_restores(fake_runtime, monkeypatch):
    originals = [layer.mlp for layer in fake_runtime.model.layers]
    def fail(*args):
        raise RuntimeError("simulated native failure")
    monkeypatch.setattr(native, "_activation", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        with native.experimental_qwen3_swiglu(fake_runtime, enabled=True) as stats:
            fake_runtime.model.layers[0].mlp(Tensor((1, 1, 128)))
    assert stats.fallback_calls == 0
    assert [layer.mlp for layer in fake_runtime.model.layers] == originals


def test_reserve_failure_precedes_model_mutation(fake_runtime, monkeypatch):
    original = fake_runtime.model.layers[0].mlp
    def fail():
        raise ValueError("40 GiB")
    monkeypatch.setattr(native, "_require_reserve", fail)
    with pytest.raises(ValueError, match="40 GiB"):
        with native.experimental_qwen3_swiglu(fake_runtime, enabled=True):
            pass
    assert fake_runtime.model.layers[0].mlp is original


def test_non_qwen_model_and_nested_installations_rejected(fake_runtime):
    fake_runtime.model_type = "qwen3_next"
    with pytest.raises(ValueError, match="exact dense Qwen3"):
        with native.experimental_qwen3_swiglu(fake_runtime, enabled=True):
            pass
    fake_runtime.model_type = "qwen3"
    with native.experimental_qwen3_swiglu(fake_runtime, enabled=True):
        outer = fake_runtime.model.layers[0].mlp
        with pytest.raises(ValueError):
            with native.experimental_qwen3_swiglu(fake_runtime, enabled=True):
                pass
        assert fake_runtime.model.layers[0].mlp is outer


def test_native_source_is_packaged_not_python_tensor_math():
    source = native.metal_source()
    assert "simd_sum" in source and "threadgroup_barrier" in source
    assert "gate_weights" in source and "up_weights" in source
