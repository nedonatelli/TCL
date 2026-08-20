"""No new public function without an executed example.

797 of 1,563 public functions carry no doctest, which means their
docstrings' factual claims are checked by nothing -- the post-v2.5.0 audit
found ~60 wrong claims in exactly that unexecuted prose. The stock cannot
be fixed at once, but it can be stopped from growing: this ratchet fails
when the count of example-less public functions EXCEEDS the recorded
baseline, and invites lowering the baseline whenever work reduces it.

The other direction fails too: if the real count drops below the baseline,
the baseline must be lowered to match, so progress is locked in rather
than silently spendable by the next example-less addition.
"""

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PACKAGE = REPO_ROOT / "pytcl"

# Lower this whenever the measured count drops; never raise it.
# History: 797 recorded 2026-08-20 (post-v2.5.0 audit).
BASELINE = 797


def _count_example_less_public_functions() -> int:
    count = 0
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_"):
                    continue
                docstring = ast.get_docstring(node) or ""
                if ">>>" not in docstring:
                    count += 1
    return count


def test_no_new_public_function_without_a_doctest():
    count = _count_example_less_public_functions()
    assert count <= BASELINE, (
        f"{count} public functions lack a doctest, up from the recorded "
        f"baseline of {BASELINE}. New public functions must carry an "
        "executed example (their prose is otherwise checked by nothing); "
        "if the addition is deliberate, the right fix is an example, not a "
        "baseline bump."
    )


def test_the_baseline_is_not_stale():
    """Progress gets locked in: a count below baseline lowers the baseline."""
    count = _count_example_less_public_functions()
    assert count >= BASELINE, (
        f"only {count} public functions lack a doctest -- lower BASELINE "
        f"from {BASELINE} to {count} so the improvement cannot be spent by "
        "the next example-less addition."
    )
