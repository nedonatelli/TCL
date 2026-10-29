"""CI steps report the status of the command that matters.

GitHub runs an unannotated ``run:`` step as ``bash -e {0}``, without
``pipefail``, so ``cmd | tee log`` takes ``tee``'s status and a failed
``cmd`` passes. The docs job did exactly that: a Sphinx ``Configuration
error`` exited 2 and the step exited 0. ``shell: bash`` runs
``bash --noprofile --norc -eo pipefail {0}`` instead.

``|| true`` and ``|| echo`` discard a status by construction; each one kept
is listed below with its reason. ``pytest -q`` on a CI line stacks with the
``-q`` already in ``addopts`` and hides the summary line in the logs.
"""

import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.yml"))
CI_YML = WORKFLOW_DIR / "ci.yml"

ALLOWED_OR_TRUE = {
    # grep -c exits 1 on zero matches, which is the passing case here.
    ("ci.yml", 'count=$(grep -c "WARNING:" /tmp/sphinx.log || true)'),
    # Retry-loop guard: clears a half-finished rebase before each attempt;
    # the attempt itself is checked.
    ("benchmark-full.yml", "git rebase --abort >/dev/null 2>&1 || true"),
}


def _run_steps(workflows):
    for wf in workflows:
        doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        for job_id, job in doc.get("jobs", {}).items():
            for step in job.get("steps", []):
                if "run" in step:
                    yield wf.name, job_id, step


def _pipes_only(script: str) -> str:
    """The script with everything that can hold a ``|`` but not a pipe removed."""
    lines = [ln for ln in script.splitlines() if not ln.lstrip().startswith("#")]
    text = "\n".join(lines)
    text = re.sub(r"\$\{\{.*?\}\}", "", text, flags=re.S)
    text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
    text = re.sub(r"'[^']*'", "''", text)
    return text.replace("||", "")


def _lines_matching(paths, pattern):
    rx = re.compile(pattern)
    found = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#"):
                continue
            if rx.search(line):
                found.append((path.name, line))
    return found


def test_every_piped_run_step_has_pipefail():
    for wf, job, step in _run_steps(WORKFLOWS):
        if "|" in _pipes_only(step["run"]):
            assert step.get("shell") == "bash" or "set -o pipefail" in step["run"], (
                f"{wf}:{job}:{step.get('name')}"
            )


STATUS_DISCARD = (
    r"\|\|\s*(?:true\b|echo\b|:(?=\s|;|\)|$)|exit\s+0\b)"
    r"|continue-on-error:\s*true\b"
)


def test_no_unreviewed_or_true():
    found = {
        (wf, line.strip()) for wf, line in _lines_matching(WORKFLOWS, STATUS_DISCARD)
    }
    assert found <= ALLOWED_OR_TRUE


def test_status_discard_pattern_catches_every_form():
    rx = re.compile(STATUS_DISCARD)
    for line in (
        "cmd || true",
        'cmd || echo "x"',
        "cmd || :",
        "cmd || exit 0",
        "        continue-on-error: true",
    ):
        assert rx.search(line), line
    for line in ("cmd || status=$?", "cmd || exit 1", "a || b"):
        assert not rx.search(line), line


def test_ci_pytest_lines_do_not_repeat_addopts_quiet():
    assert not _lines_matching([CI_YML], r"pytest .* -q(\s|$)")


def test_pipes_only_ignores_or_quoted_alternation_and_expressions():
    script = (
        "# a | b\ngrep -qE \"A|B\" log\nx || y\necho ${{ a || 'b' }}\nsed 's/a|b//' f\n"
    )
    assert "|" not in _pipes_only(script)
    assert "|" in _pipes_only("cmd 2>&1 | tee log")
