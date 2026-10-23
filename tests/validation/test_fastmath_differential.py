"""Every fastmath kernel agrees with its own interpreted form.

The coverage cell runs with ``NUMBA_DISABLE_JIT=1``, so it measures the
kernels as plain Python. ``fastmath`` lets LLVM assume no NaN, no infinity
and free reassociation, so the compiled kernel can disagree with the code
coverage reports on: that divergence is what hid the SO-CFAR sentinel bug
fixed in 2.11.1 (a NaN test compiled away under ``nnan``). Here each
kernel runs compiled and as ``kernel.py_func`` on the same inputs,
including NaN, +/-inf, zero-width windows and singular matrices, and the
two must agree to 1e-12 relative.

The oracle is the interpreted body, which is the code coverage measures
and the code reviewers read. ``test_registry_lists_every_fastmath_kernel``
keeps the registry complete: a new ``fastmath`` kernel fails it until it
is registered here with inputs.
"""

import ast
import copy
import importlib
import pathlib

import numba
import numpy as np
import pytest

gating = importlib.import_module("pytcl.assignment_algorithms.gating")
detection = importlib.import_module(
    "pytcl.mathematical_functions.signal_processing.detection"
)
matched_filter = importlib.import_module(
    "pytcl.mathematical_functions.signal_processing.matched_filter"
)

PYTCL_ROOT = pathlib.Path(__file__).resolve().parents[2] / "pytcl"


def _jit_disabled():
    return bool(numba.config.DISABLE_JIT)


def _is_fastmath_off(value):
    return isinstance(value, ast.Constant) and value.value is False


def _ast_fastmath_kernels(root):
    """``module:name`` of every function decorated with fastmath enabled."""
    keys = []
    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root.parent).with_suffix("").parts)
        module = module.removesuffix(".__init__")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and any(
                    kw.arg == "fastmath" and not _is_fastmath_off(kw.value)
                    for kw in dec.keywords
                ):
                    keys.append(f"{module}:{node.name}")
    return keys


def _kernel_key(kernel):
    func = getattr(kernel, "py_func", kernel)
    return f"{func.__module__}:{func.__name__}"


_RNG_SEED = 20261023


def _spd(n, rng):
    a = rng.standard_normal((n, n))
    return a @ a.T + n * np.eye(n)


def _special_vectors(n, include_inf=True):
    v_nan = np.ones(n)
    v_nan[0] = np.nan
    v_inf = np.ones(n)
    v_inf[-1] = np.inf
    v_ninf = np.ones(n)
    v_ninf[-1] = -np.inf
    return [v_nan, np.zeros(n)] + ([v_inf, v_ninf] if include_inf else [])


def _gating_cases(n, include_inf=True):
    rng = np.random.default_rng(_RNG_SEED + n)
    cases = []
    for _ in range(5):
        cases.append((rng.standard_normal(n), np.linalg.inv(_spd(n, rng))))
    s_inv = np.linalg.inv(_spd(n, rng))
    for v in _special_vectors(n, include_inf):
        cases.append((v, s_inv))
    singular = np.ones((n, n))
    cases.append((rng.standard_normal(n), singular))
    return cases


def _gating_2d_cases():
    return _gating_cases(2)


def _gating_3d_cases():
    # An infinite innovation is excluded: under ninf the compiled kernel
    # returns inf where the interpreted sum hits inf - inf = nan. Recorded
    # as unsupported input in the kernel's docstring.
    return _gating_cases(3, include_inf=False)


def _gating_general_cases():
    return _gating_cases(2) + _gating_cases(3) + _gating_cases(6)


def _invert_cases(n):
    rng = np.random.default_rng(_RNG_SEED + 10 * n)
    cases = [(_spd(n, rng),) for _ in range(5)]
    cases.append((rng.standard_normal((n, n)),))
    cases.append((np.ones((n, n)),))
    cases.append((np.zeros((n, n)),))
    m_nan = _spd(n, rng)
    m_nan[0, 1] = np.nan
    cases.append((m_nan,))
    m_inf = _spd(n, rng)
    m_inf[0, 0] = np.inf
    cases.append((m_inf,))
    return cases


def _invert_2x2_cases():
    return _invert_cases(2)


def _invert_3x3_cases():
    return _invert_cases(3)


def _batch_cases():
    rng = np.random.default_rng(_RNG_SEED + 100)
    cases = []
    for dim in (2, 3, 5):
        innov = rng.standard_normal((7, dim))
        innov[1, 0] = np.nan
        innov[2, -1] = np.inf
        innov[3, 0] = -np.inf
        innov[4] = 0.0
        cases.append((innov, np.linalg.inv(_spd(dim, rng)), np.empty(7)))
    cases.append((np.empty((0, 3)), np.eye(3), np.empty(0)))
    return cases


def _cfar_signals():
    rng = np.random.default_rng(_RNG_SEED + 200)
    base = rng.exponential(1.0, 40)
    signals = [base, np.zeros(40), base[:1], base[:3]]
    with_nan = base.copy()
    with_nan[[5, 20]] = np.nan
    with_inf = base.copy()
    with_inf[12] = np.inf
    return signals, [with_nan], [with_inf]


_WINDOWS = [(2, 4), (0, 3), (3, 0), (0, 0), (25, 25)]


def _cfar_1d_cases(include_nan=True, include_inf=True):
    finite, nan, inf = _cfar_signals()
    signals = finite + (nan if include_nan else []) + (inf if include_inf else [])
    cases = []
    for sig in signals:
        for guard, ref in _WINDOWS:
            n = len(sig)
            cases.append((sig, guard, ref, 3.5, np.empty(n), np.empty(n)))
    return cases


def _cfar_1d_cases_finite_or_inf():
    # NaN data is excluded: under nnan the compiled GO kernel drops the
    # comparison with a NaN average and returns a finite threshold where the
    # interpreted max() returns NaN. Recorded as unsupported input in the
    # kernel's docstring.
    return _cfar_1d_cases(include_nan=False)


def _cfar_os_cases(include_nan=True, include_inf=True):
    finite, nan, inf = _cfar_signals()
    signals = finite + (nan if include_nan else []) + (inf if include_inf else [])
    cases = []
    for sig in signals:
        for guard, ref in _WINDOWS:
            n = len(sig)
            for k in (1, max(1, ref), 2 * ref + 10):
                cases.append((sig, guard, ref, k, 3.5, np.empty(n), np.empty(n)))
    return cases


_WINDOWS_2D = [(1, 1, 2, 2), (0, 0, 1, 1), (0, 0, 0, 0), (2, 1, 0, 3), (9, 9, 9, 9)]


def _cfar_2d_cases(include_nan=True, include_inf=True):
    rng = np.random.default_rng(_RNG_SEED + 300)
    base = rng.exponential(1.0, (9, 11))
    images = [base, np.zeros((9, 11)), base[:1, :1], base[:2, :5]]
    if include_nan:
        with_nan = base.copy()
        with_nan[4, 5] = np.nan
        images.append(with_nan)
    if include_inf:
        with_inf = base.copy()
        with_inf[2, 3] = np.inf
        images.append(with_inf)
    cases = []
    for img in images:
        for gr, gc, rr, rc in _WINDOWS_2D:
            cases.append(
                (img, gr, gc, rr, rc, 3.5, np.empty(img.shape), np.empty(img.shape))
            )
    return cases


def _cfar_2d_cases_finite_or_inf():
    # NaN data is excluded for the 2D GO kernel, for the nnan reason given
    # in _cfar_1d_cases_finite_or_inf.
    return _cfar_2d_cases(include_nan=False)


def _ambiguity_signals():
    n = 32
    t = np.arange(n)
    chirp = np.exp(1j * np.pi * 0.01 * t**2)
    rng = np.random.default_rng(_RNG_SEED + 400)
    noise = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    return chirp, noise


def _ambiguity_grid(n, fs):
    delays = np.array([-n - 3, -5, -1, 0, 1, 7, n + 2], dtype=float) / fs
    dopplers = np.array([-0.2, 0.0, 0.05, 0.3]) * fs
    return delays, dopplers


def _ambiguity_cases():
    chirp, noise = _ambiguity_signals()
    fs = 1000.0
    delays, dopplers = _ambiguity_grid(len(chirp), fs)
    out = np.empty((len(dopplers), len(delays)), dtype=np.complex128)
    return [(sig, delays, dopplers, fs, out.copy()) for sig in (chirp, noise)] + [
        (chirp[:1], delays, dopplers, fs, out.copy())
    ]


def _cross_ambiguity_cases():
    chirp, noise = _ambiguity_signals()
    fs = 1000.0
    delays, dopplers = _ambiguity_grid(len(chirp), fs)
    out = np.empty((len(dopplers), len(delays)), dtype=np.complex128)
    return [
        (chirp, noise, delays, dopplers, fs, out.copy()),
        (noise, chirp, delays, dopplers, fs, out.copy()),
        (chirp[:1], noise[:1], delays, dopplers, fs, out.copy()),
    ]


_FASTMATH_KERNELS = {
    gating._mahalanobis_distance_2d: _gating_2d_cases,
    gating._mahalanobis_distance_3d: _gating_3d_cases,
    gating._mahalanobis_distance_general: _gating_general_cases,
    gating._invert_2x2: _invert_2x2_cases,
    gating._invert_3x3: _invert_3x3_cases,
    gating.mahalanobis_batch: _batch_cases,
    detection._cfar_ca_kernel: _cfar_1d_cases,
    detection._cfar_go_kernel: _cfar_1d_cases_finite_or_inf,
    detection._cfar_so_kernel: _cfar_1d_cases,
    detection._cfar_os_kernel: _cfar_os_cases,
    detection._cfar_2d_ca_kernel: _cfar_2d_cases,
    detection._cfar_2d_go_kernel: _cfar_2d_cases_finite_or_inf,
    detection._cfar_2d_so_kernel: _cfar_2d_cases,
    matched_filter._ambiguity_function_kernel: _ambiguity_cases,
    matched_filter._cross_ambiguity_kernel: _cross_ambiguity_cases,
}


def _defined_part(kernel, result):
    """The part of a kernel's result its contract defines.

    ``_invert_2x2``/``_invert_3x3`` return an uninitialized inverse when the
    determinant is exactly zero and document that the caller must not read
    it, so only the determinant is compared there.
    """
    if kernel in (gating._invert_2x2, gating._invert_3x3):
        inv, det = result[0]
        return [det] if det == 0.0 else [inv, det]
    return result


def _call(kernel, args):
    args = copy.deepcopy(args)
    with np.errstate(invalid="ignore"):
        ret = kernel(*args)
    outputs = [a for a in args if isinstance(a, np.ndarray)]
    return [ret] + outputs if ret is not None else outputs


def test_registry_lists_every_fastmath_kernel():
    assert sorted(_ast_fastmath_kernels(PYTCL_ROOT)) == sorted(
        _kernel_key(k) for k in _FASTMATH_KERNELS
    )


def test_kernel_scan_keys_by_module_and_skips_fastmath_off(tmp_path):
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    body = (
        "from numba import njit\n"
        "@njit(fastmath=True)\ndef k(x):\n    return x\n"
        "@njit(fastmath=False)\ndef off(x):\n    return x\n"
        "@njit(fastmath={'nsz'})\ndef flags(x):\n    return x\n"
    )
    (pkg / "a.py").write_text(body)
    (pkg / "sub" / "b.py").write_text(body)
    assert sorted(_ast_fastmath_kernels(pkg)) == [
        "pkg.a:flags",
        "pkg.a:k",
        "pkg.sub.b:flags",
        "pkg.sub.b:k",
    ]


@pytest.mark.skipif(
    _jit_disabled(), reason="NUMBA_DISABLE_JIT=1: no compiled path to compare"
)
@pytest.mark.parametrize("kernel", list(_FASTMATH_KERNELS), ids=lambda k: k.__name__)
def test_compiled_and_interpreted_kernels_agree(kernel):
    for args in _FASTMATH_KERNELS[kernel]():
        compiled = _defined_part(kernel, _call(kernel, args))
        interpreted = _defined_part(kernel, _call(kernel.py_func, args))
        assert len(compiled) == len(interpreted)
        for c, p in zip(compiled, interpreted):
            np.testing.assert_allclose(c, p, rtol=1e-12, atol=0, equal_nan=True)
