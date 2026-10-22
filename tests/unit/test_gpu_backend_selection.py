"""Backend auto-selection must not hide a CUDA failure behind MLX float32."""

import types
import warnings

import pytest

from pytcl.core.exceptions import DependencyError
from pytcl.gpu import _backend
from pytcl.gpu._backend import compute_backend_available, get_compute_backend

pytestmark = pytest.mark.skipif(
    not compute_backend_available(), reason="no GPU compute backend installed"
)


def _raising(exc):
    def ctor():
        raise exc

    return ctor


class _Float64Fallback:
    name = "double"
    supports_float64 = True


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
    monkeypatch.setattr(
        _backend, "CuPyBackend", _raising(ImportError("No module named 'cupy'"))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_available_warns_on_installed_but_broken_cupy(monkeypatch):
    class Broken(types.ModuleType):
        def __getattr__(self, name):
            raise AttributeError(name)

    def fake_import(name):
        if name == "cupy":
            raise RuntimeError("CUDA driver version")
        return Broken(name)

    monkeypatch.setattr(_backend.importlib, "import_module", fake_import)
    with pytest.warns(UserWarning, match="CUDA driver version"):
        assert compute_backend_available() is True


def test_available_is_quiet_when_nothing_installed(monkeypatch):
    def fake_import(name):
        raise ImportError(name)

    monkeypatch.setattr(_backend.importlib, "import_module", fake_import)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compute_backend_available() is False
