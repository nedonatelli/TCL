"""Backend auto-selection must not hide a CUDA failure behind MLX float32."""

import types
import warnings

import pytest

from pytcl.core.exceptions import DependencyError
from pytcl.gpu import _backend
from pytcl.gpu._backend import compute_backend_available, get_compute_backend


def _raising(exc):
    def ctor():
        raise exc

    return ctor


class _Float32Fallback:
    name = "mlx"
    supports_float64 = False


class _Float64Fallback:
    name = "double"
    supports_float64 = True


@pytest.fixture(autouse=True)
def _stand_in_mlx(monkeypatch):
    # Selection logic is under test, not MLX itself: a stand-in lets every CI
    # runner exercise it (none has MLX or CuPy, so a real-backend skipif
    # left these tests, and the selection code, unrun in CI).
    monkeypatch.setattr(_backend, "MLXBackend", _Float32Fallback)


def test_cupy_construction_failure_warns_before_falling_back(monkeypatch):
    monkeypatch.setattr(
        _backend, "CuPyBackend", _raising(RuntimeError("CUDA driver version"))
    )
    with pytest.warns(UserWarning, match="CUDA driver version"):
        backend = get_compute_backend()
    assert backend.name != "cupy"


def test_precision_loss_on_fallback_is_announced(monkeypatch):
    monkeypatch.setattr(_backend, "CuPyBackend", _raising(RuntimeError("boom")))
    with pytest.warns(UserWarning, match="float32") as record:
        backend = get_compute_backend()
    assert backend.supports_float64 is False
    assert any("boom" in str(w.message) for w in record)


def test_prefer_cupy_failure_still_raises_with_original_message(monkeypatch):
    monkeypatch.setattr(_backend, "CuPyBackend", _raising(RuntimeError("boom")))
    with pytest.raises(DependencyError, match="boom") as info:
        get_compute_backend(prefer="cupy")
    assert isinstance(info.value.__cause__, RuntimeError)


def test_cupy_not_installed_is_quiet(monkeypatch):
    absent = ModuleNotFoundError("No module named 'cupy'", name="cupy")
    monkeypatch.setattr(_backend, "CuPyBackend", _raising(absent))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend = get_compute_backend()
    assert backend.name != "cupy"


@pytest.mark.parametrize(
    "exc",
    [
        ImportError("libcudart.so.12: cannot open shared object file"),
        ModuleNotFoundError("No module named 'fastrlock'", name="fastrlock"),
        ModuleNotFoundError(
            "No module named 'cupy_backends.cuda'", name="cupy_backends.cuda"
        ),
    ],
)
def test_broken_cupy_install_warns_with_original_text(monkeypatch, exc):
    monkeypatch.setattr(_backend, "CuPyBackend", _raising(exc))
    with pytest.warns(UserWarning, match=str(exc).split(":")[0]):
        backend = get_compute_backend()
    assert backend.name != "cupy"


def test_no_precision_notice_when_fallback_supports_float64(monkeypatch):
    monkeypatch.setattr(_backend, "CuPyBackend", _raising(RuntimeError("boom")))
    monkeypatch.setattr(_backend, "MLXBackend", _Float64Fallback)
    with pytest.warns(UserWarning, match="boom") as record:
        backend = get_compute_backend()
    assert backend.name == "double"
    assert not any("float32" in str(w.message) for w in record)


def test_both_backends_failing_raises_with_cupy_cause(monkeypatch):
    monkeypatch.setattr(_backend, "CuPyBackend", _raising(RuntimeError("boom")))
    monkeypatch.setattr(_backend, "MLXBackend", _raising(ImportError("no mlx")))
    with pytest.raises(DependencyError, match="boom") as info:
        get_compute_backend()
    assert isinstance(info.value.__cause__, RuntimeError)


def test_available_is_quiet_when_cupy_import_raises_runtime_error(monkeypatch):
    def fake_import(name):
        if name == "cupy":
            raise RuntimeError("CUDA driver version")
        return types.ModuleType(name)

    monkeypatch.setattr(_backend.importlib, "import_module", fake_import)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compute_backend_available() is True

    def only_broken(name):
        raise RuntimeError("CUDA driver version")

    monkeypatch.setattr(_backend.importlib, "import_module", only_broken)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compute_backend_available() is False


def test_available_is_quiet_when_nothing_installed(monkeypatch):
    def fake_import(name):
        raise ImportError(name)

    monkeypatch.setattr(_backend.importlib, "import_module", fake_import)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compute_backend_available() is False
