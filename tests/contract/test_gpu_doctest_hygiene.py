"""A CPU-runnable gate on the ``pytcl.gpu`` docstring examples.

The GPU examples cannot be executed by CI: ``conftest`` drops ``pytcl/gpu``
from collection when no compute backend is installed, and no runner has a GPU.
So the doctests in that package are checked by nothing, and three distinct
bugs lived there until the suite was run on an RTX 5080:

- five examples handed a device array straight back to numpy. ``gpu_solve``
  and ``gpu_inv`` raised ``TypeError`` outright (``numpy @ cupy``); the other
  three returned ``array(True)`` instead of ``True`` because ``np.allclose``
  dispatched to CuPy;
- ``is_cupy_available`` printed under an ``if`` while declaring no expected
  output, so it passed exactly on machines *without* CUDA;
- ``get_array_module`` imported ``mlx.core`` unconditionally, which fails off
  Apple Silicon.

None of those need a GPU to detect -- they are visible in the source of the
example. This module reads the examples and applies the three rules
statically, so the class of bug is caught on any runner.

The checks are deliberately paired with negative controls further down: each
rule is run against the original broken example it was written for. A gate
nobody has watched fail is a gate nobody should trust.
"""

import ast
import doctest
import importlib
import pkgutil

import pytest

import pytcl.gpu

# Calls whose result lives on the device when a backend is active.
_DEVICE_RETURNING = ("to_gpu", "ensure_gpu_array")
_DEVICE_PREFIX = "gpu_"

# Importing either of these off its own platform is an error, so an example
# that does it must be skipped rather than executed.
_PLATFORM_ONLY = ("cupy", "mlx")


def _is_device_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    return name.startswith(_DEVICE_PREFIX) or name in _DEVICE_RETURNING


def _root_name(node: ast.AST):
    """Walk an attribute chain to its base name: np.random.randn -> 'np'."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _is_numpy_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _root_name(node.func) in ("np", "numpy")


def _contains_numpy_call(node: ast.AST) -> bool:
    return any(_is_numpy_call(n) for n in ast.walk(node))


def _wrapped_in_to_cpu(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and getattr(node.func, "id", getattr(node.func, "attr", "")) == "to_cpu"
    )


def _names(node: ast.AST) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _names_converted(node: ast.AST) -> set:
    """Names sitting inside a ``to_cpu(...)`` anywhere in this subtree.

    The conversion is frequently nested rather than outermost --
    ``np.asarray(to_cpu(eigvals))``, ``to_cpu(Q) @ to_cpu(R)`` -- so testing
    only the top-level node reports correct examples as broken.
    """
    out = set()
    for n in ast.walk(node):
        if _wrapped_in_to_cpu(n):
            out |= _names(n)
    return out


# Where a user-supplied callback is handed device arrays. The two families
# differ: batch_ekf_predict/batch_ukf_predict call to_cpu first and invoke the
# dynamics callback per track with a numpy array, so an example that mixes
# numpy into it is correct. CuPyParticleFilter.predict passes self.particles
# straight through, so the same expression there is the bug found on the 5080.
_CALLBACKS_RECEIVE_DEVICE_ARRAYS = frozenset({"pytcl.gpu.particle_filter"})


def check_device_arrays_reach_numpy(
    source: str, callbacks_are_device: bool = False
) -> list:
    """Flag a device array used by numpy without ``to_cpu``.

    This is the ``gpu_solve``/``gpu_inv``/``gpu_cholesky`` family of bugs.

    ``callbacks_are_device`` enables the stricter rule about names of unknown
    origin, which only holds where callbacks receive device arrays.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    device, numpy_names, problems = set(), set(), []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = set()
        for t in node.targets:
            targets |= _names(t)
        if _wrapped_in_to_cpu(node.value):
            device -= targets
            numpy_names |= targets
        elif _is_device_call(node.value):
            device |= targets
        elif _is_numpy_call(node.value):
            numpy_names |= targets

    for node in ast.walk(tree):
        # A device value passed into any numpy function.
        if _is_numpy_call(node):
            for arg in node.args:
                touched = (_names(arg) - _names_converted(arg)) & device
                if touched:
                    problems.append(
                        f"device array {sorted(touched)} passed to "
                        f"np.{node.func.attr}() without to_cpu()"
                    )
        # A device value combined with a host value in one expression.
        if isinstance(node, ast.BinOp):
            left = _names(node.left) - _names_converted(node.left)
            right = _names(node.right) - _names_converted(node.right)
            if (left & device and right & numpy_names) or (
                right & device and left & numpy_names
            ):
                problems.append(
                    "device array combined with a numpy array in a binary op; "
                    "CuPy rejects a host operand"
                )
            # `particles + np.random.randn(*particles.shape) * 0.1`: one side
            # builds a host array, the other is a name of unknown origin -- in
            # a pytcl.gpu example that name is holding device data, and CuPy
            # rejects the mix. The numpy call can be nested (here it sits under
            # a further multiply), so look through the whole subtree.
            for a, b in ((node.left, node.right), (node.right, node.left)):
                if not callbacks_are_device:
                    continue
                if not _contains_numpy_call(a) or _contains_numpy_call(b):
                    continue
                unknown = _names(b) - numpy_names - device
                if unknown:
                    problems.append(
                        f"host array combined with {sorted(unknown)}, whose "
                        "origin is unknown; if it holds device data CuPy "
                        "rejects this"
                    )
    return problems


def check_conditional_output_is_declared(source: str, want: str) -> list:
    """Flag an example that prints under a condition but expects no output.

    ``is_cupy_available`` did this: green if and only if the branch it
    documents does not run.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if want.strip():
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and getattr(inner.func, "id", "") in (
                    "print",
                ):
                    return [
                        "prints inside an if but declares no expected output, "
                        "so it passes only when the branch does not run"
                    ]
    return []


def check_platform_imports_are_skipped(source: str, is_skipped: bool) -> list:
    """Flag an unconditional import of a platform-only backend."""
    if is_skipped:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    problems = []
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            if m.split(".")[0] in _PLATFORM_ONLY:
                problems.append(
                    f"imports {m!r} without '# doctest: +SKIP'; it is not "
                    "installable on every platform"
                )
    return problems


def _gpu_doctests():
    """(name, examples) for every docstring in pytcl.gpu that has examples.

    Grouped per docstring, not per example. doctest splits every ``>>>`` line
    into its own Example, so ``A_inv = gpu_inv(A)`` and the ``np.allclose``
    line that misuses it arrive separately -- checking them one at a time
    cannot see an assignment reaching a later use, which is exactly the bug
    this module exists to catch.
    """
    finder = doctest.DocTestFinder(recurse=True, exclude_empty=True)
    out = []
    for info in pkgutil.walk_packages(pytcl.gpu.__path__, "pytcl.gpu."):
        module = importlib.import_module(info.name)
        for test in finder.find(module):
            if test.examples:
                out.append((test.name, test.examples))
    return out


_DOCTESTS = _gpu_doctests()
_EXAMPLES = [(n, e) for n, exs in _DOCTESTS for e in exs]


def test_the_scan_found_examples():
    """Guard the guard: an empty scan would make every check below vacuous."""
    assert len(_EXAMPLES) > 50, f"only {len(_EXAMPLES)} examples found"


def _joined(examples):
    """The executable lines of one docstring, as a single program.

    +SKIP lines are dropped: they never run, so a device misuse inside one
    cannot fail anybody.
    """
    return "".join(e.source for e in examples if doctest.SKIP not in e.options)


@pytest.mark.parametrize("name,examples", _DOCTESTS, ids=[n for n, _ in _DOCTESTS])
def test_examples_are_backend_correct(name, examples):
    module = name.rsplit(".", 1)[0] if "." in name else name
    callbacks_are_device = any(
        name == m or name.startswith(m + ".") or module == m
        for m in _CALLBACKS_RECEIVE_DEVICE_ARRAYS
    )
    problems = check_device_arrays_reach_numpy(_joined(examples), callbacks_are_device)
    for example in examples:
        problems += check_conditional_output_is_declared(example.source, example.want)
        problems += check_platform_imports_are_skipped(
            example.source, doctest.SKIP in example.options
        )
    assert not problems, f"{name}: " + "; ".join(problems)


class TestTheChecksCatchTheRealBugs:
    """Negative controls: the exact examples that shipped broken.

    Each is the source as it stood before the RTX 5080 run, so a regression in
    the checker shows up here rather than silently passing everything.
    """

    @staticmethod
    def _as_doctest_would(lines):
        """Rebuild source the way the real scan does: one Example per line.

        The first version of these controls passed a single multi-line string,
        which no real docstring ever produces -- doctest emits one Example per
        ``>>>`` line. That difference is why the checker shipped unable to
        catch the very bug it was written for.
        """
        return "".join(line if line.endswith("\n") else line + "\n" for line in lines)

    def test_catches_gpu_solve(self):
        source = self._as_doctest_would(
            ["x = gpu_solve(A, b)", "np.allclose(A @ x, b)"]
        )
        assert check_device_arrays_reach_numpy(source)

    def test_catches_gpu_inv(self):
        source = "A_inv = gpu_inv(A)\nnp.allclose(A @ A_inv, np.eye(2))\n"
        assert check_device_arrays_reach_numpy(source)

    def test_catches_gpu_cholesky(self):
        source = "L = gpu_cholesky(A)\nnp.allclose(L @ L.T, A)\n"
        assert check_device_arrays_reach_numpy(source)

    def test_catches_the_particle_filter_noise(self):
        source = "particles + np.random.randn(*particles.shape) * 0.1\n"
        assert check_device_arrays_reach_numpy(source, callbacks_are_device=True)

    def test_allows_the_same_shape_where_callbacks_get_host_arrays(self):
        """batch_ekf_predict calls its dynamics callback with numpy, so the
        identical expression is correct there. Without this distinction the
        rule would fire on every EKF and UKF example."""
        source = "np.array([x[0] + np.cos(w) * x[2]])\n"
        assert not check_device_arrays_reach_numpy(source, callbacks_are_device=False)

    def test_catches_the_conditional_print(self):
        source = 'if is_cupy_available():\n    print("CUDA GPU available")\n'
        assert check_conditional_output_is_declared(source, want="")

    def test_conditional_print_is_fine_when_output_is_declared(self):
        source = 'if True:\n    print("x")\n'
        assert not check_conditional_output_is_declared(source, want="x\n")

    def test_catches_the_unconditional_mlx_import(self):
        assert check_platform_imports_are_skipped(
            "import mlx.core as mx\n", is_skipped=False
        )

    def test_platform_import_is_fine_when_skipped(self):
        assert not check_platform_imports_are_skipped(
            "import mlx.core as mx\n", is_skipped=True
        )

    def test_nested_to_cpu_is_recognized(self):
        """The forms that the first version of this checker misreported."""
        assert not check_device_arrays_reach_numpy(
            self._as_doctest_would(
                [
                    "eigvals, eigvecs = gpu_eigh(A)",
                    "bool(np.allclose(np.asarray(to_cpu(eigvals)), [1.0, 3.0]))",
                ]
            )
        )
        assert not check_device_arrays_reach_numpy(
            self._as_doctest_would(
                ["Q, R = gpu_qr(A)", "np.allclose(to_cpu(Q) @ to_cpu(R), A)"]
            )
        )

    def test_the_fixed_form_passes(self):
        """The corrected examples must not trip the checker."""
        assert not check_device_arrays_reach_numpy(
            "x = to_cpu(gpu_solve(A, b))\nnp.allclose(A @ x, b)\n"
        )
        assert not check_device_arrays_reach_numpy(
            "L = to_cpu(gpu_cholesky(A))\nnp.allclose(L @ L.T, A)\n"
        )
