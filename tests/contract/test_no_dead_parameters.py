"""Every public parameter must be read by its function's body.

A parameter that is accepted, documented, and never read is worse than a
missing feature: callers tune it, tests pass it, and nothing moves. The
post-v2.5.0 audit (2026-08) found four of these the hard way (``KDTree.leaf_size``,
``MultiTargetTracker.confirm_window``, ``RTree.min_entries``, RBPF's ``G``),
and this gate's first sweep found five more that had survived every previous
review (``marcum_q_inv``'s ``tol``/``max_iter`` beside a Notes section
claiming an iteration no code performed, the matched filters' ``fs``
documented as "used for output scaling", ``parse_earth2014_binary``'s
``layer``, ``fisher_information_exponential_family``'s ``h``,
``associated_legendre_scaled``'s ``scale``).

Exemptions are structural where they can be (interface stubs whose body only
raises, ``lru_cache``-decorated functions whose parameters are cache keys,
nested callback adapters) and an explicit allowlist with reasons where they
cannot. An allowlist entry naming a parameter that no longer exists fails,
so the list cannot outlive the code it excuses.
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PACKAGE = REPO_ROOT / "pytcl"

# (module-relative-path, function, parameter): why the parameter is
# legitimately unread. Every entry must correspond to a real, still-unread
# parameter -- stale entries fail the gate from the other direction.
ALLOWED: dict[tuple[str, str, str], str] = {
    # SDE callback contract: drift f(x, t) and diffusion g(x, t) must share a
    # signature for the integrators even when a model is autonomous (t unused)
    # or state-independent (x unused).
    (
        "dynamic_models/continuous_time/dynamics.py",
        "drift_constant_velocity",
        "t",
    ): "autonomous drift, f(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "drift_constant_acceleration",
        "t",
    ): "autonomous drift, f(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "drift_singer",
        "t",
    ): "autonomous drift, f(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "drift_coordinated_turn_2d",
        "t",
    ): "autonomous drift, f(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "diffusion_constant_velocity",
        "x",
    ): "state-independent diffusion, g(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "diffusion_constant_velocity",
        "t",
    ): "autonomous diffusion, g(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "diffusion_constant_acceleration",
        "x",
    ): "state-independent diffusion, g(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "diffusion_constant_acceleration",
        "t",
    ): "autonomous diffusion, g(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "diffusion_singer",
        "x",
    ): "state-independent diffusion, g(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "diffusion_singer",
        "t",
    ): "autonomous diffusion, g(x, t) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "state_jacobian_cv",
        "x",
    ): "constant Jacobian, J(x) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "state_jacobian_ca",
        "x",
    ): "constant Jacobian, J(x) contract",
    (
        "dynamic_models/continuous_time/dynamics.py",
        "state_jacobian_singer",
        "x",
    ): "constant Jacobian, J(x) contract",
    # Physics: the docstrings state the independence explicitly.
    (
        "astronomical/relativity.py",
        "geodetic_precession",
        "inclination",
    ): "docstring: magnitude independent of inclination; kept for orbital-elements signature parity",
    (
        "astronomical/relativity.py",
        "lense_thirring_precession",
        "inclination",
    ): "docstring: rate independent of inclination",
    (
        "astronomical/relativity.py",
        "lense_thirring_precession",
        "gm",
    ): "formula uses G*J, not GM; kept for signature parity with geodetic_precession",
    (
        "gravity/models.py",
        "gravity_wgs84",
        "lon",
    ): "normal gravity is longitude-independent by rotational symmetry; uniform lat/lon signature",
    (
        "gravity/models.py",
        "gravity_j2",
        "lon",
    ): "normal gravity is longitude-independent by rotational symmetry; uniform lat/lon signature",
    (
        "gravity/tides.py",
        "atmospheric_pressure_loading",
        "lat",
    ): "simple admittance model is position-independent; uniform signature with the other loading functions",
    (
        "gravity/tides.py",
        "atmospheric_pressure_loading",
        "lon",
    ): "simple admittance model is position-independent; uniform signature with the other loading functions",
    # Leveling from accelerometers needs only the gravity direction; lat is
    # kept for signature parity with gyrocompass_alignment, which needs it.
    (
        "navigation/ins.py",
        "coarse_alignment",
        "lat",
    ): "leveling uses gravity direction only; signature parity with gyrocompass_alignment",
}


def _body_is_stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Docstring plus nothing, or docstring plus a bare raise."""
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        body = body[1:]
    if not body:
        return True
    if len(body) == 1 and isinstance(body[0], (ast.Raise, ast.Pass)):
        return True
    if (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
    ):
        return True  # Ellipsis
    return False


def _exempt_by_decorator(node) -> bool:
    for dec in node.decorator_list:
        text = ast.unparse(dec)
        if "abstractmethod" in text or "overload" in text:
            return True
        if "lru_cache" in text or "cache" == text.split("(")[0].split(".")[-1]:
            return True  # parameters are cache keys even when unread
    return False


def _find_dead_parameters():
    findings = []
    for path in sorted(PACKAGE.rglob("*.py")):
        rel = str(path.relative_to(PACKAGE))
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))

        # module-level functions and methods of module-level classes only:
        # nested defs are callback adapters whose signatures a caller owns
        scopes: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scopes.append(node)
            elif isinstance(node, ast.ClassDef):
                scopes.extend(
                    n
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                )

        for node in scopes:
            if node.name.startswith("_") and node.name != "__init__":
                continue
            if _exempt_by_decorator(node) or _body_is_stub(node):
                continue

            params = [
                a.arg
                for a in (node.args.posonlyargs + node.args.args + node.args.kwonlyargs)
                if a.arg not in ("self", "cls") and not a.arg.startswith("_")
            ]
            read = {
                n.id
                for stmt in node.body
                for n in ast.walk(stmt)
                if isinstance(n, ast.Name)
            }
            for param in params:
                if param not in read:
                    findings.append((rel, node.name, param))
    return findings


def test_every_public_parameter_is_read():
    dead = [
        f"pytcl/{rel}: {fn}({param})"
        for rel, fn, param in _find_dead_parameters()
        if (rel, fn, param) not in ALLOWED
    ]
    assert not dead, (
        "parameters accepted but never read -- implement, remove, or add to "
        "ALLOWED with a reason:\n  " + "\n  ".join(sorted(dead))
    )


def test_the_allowlist_is_not_stale():
    """An entry excusing a parameter that was removed or is now read must go."""
    current = set(_find_dead_parameters())
    stale = [k for k in ALLOWED if k not in current]
    assert not stale, (
        "ALLOWED entries no longer matching an unread parameter: "
        + ", ".join(f"{r}:{f}({p})" for r, f, p in stale)
    )


class TestTheGateActuallyFires:
    """Negative control on a synthetic module."""

    def test_a_dead_parameter_is_found(self, tmp_path, monkeypatch):
        mod = tmp_path / "synthetic.py"
        mod.write_text("def f(used, ignored):\n    return used + 1\n", encoding="utf-8")
        import tests.contract.test_no_dead_parameters as gate

        monkeypatch.setattr(gate, "PACKAGE", tmp_path)
        found = gate._find_dead_parameters()
        assert ("synthetic.py", "f", "ignored") in found
        assert ("synthetic.py", "f", "used") not in found
