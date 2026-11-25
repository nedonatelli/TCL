"""The coverage-claim and CHANGELOG-entry CI checks must trip.

Both scripts guard claims that used to be kept up to date by hand: v2.12.0's
release PR merged with the previous release's coverage figure, and its
whole-branch review found four behavior changes with no CHANGELOG entry.
"""

import os
import subprocess
from pathlib import Path

import pytest

from scripts import check_changelog_entry, check_coverage_claim


def _cobertura(lines_valid, lines_covered, branches_valid, branches_covered):
    return (
        f'<?xml version="1.0" ?><coverage lines-valid="{lines_valid}" '
        f'lines-covered="{lines_covered}" branches-valid="{branches_valid}" '
        f'branches-covered="{branches_covered}" line-rate="0" branch-rate="0" '
        'version="7" timestamp="0"><packages/></coverage>'
    )


class TestCoverageClaim:
    def test_total_counts_statements_and_branches(self):
        # 900 + 50 of 1000 + 100: 950 / 1100, not the 90% line rate.
        xml = _cobertura(1000, 900, 100, 50)
        assert check_coverage_claim.measured_percent(xml) == pytest.approx(
            100 * 950 / 1100, rel=1e-15, abs=0
        )

    def test_recorded_figure_is_read_from_the_metrics_line(self):
        text = "- **Coverage:** 95.25% as CI's gate measures it"
        assert check_coverage_claim.recorded_percent(text) == 95.25

    def test_recorded_figure_must_exist(self):
        with pytest.raises(ValueError, match="Coverage"):
            check_coverage_claim.recorded_percent("no metrics here")

    @pytest.mark.parametrize(
        "covered, ok",
        [(9525, True), (9426, True), (9624, True), (9420, False), (9630, False)],
    )
    def test_main_fails_beyond_the_tolerance(self, tmp_path, monkeypatch, covered, ok):
        contributing = tmp_path / "CONTRIBUTING.md"
        contributing.write_text("- **Coverage:** 95.25% measured\n", encoding="utf-8")
        monkeypatch.setattr(check_coverage_claim, "CONTRIBUTING", contributing)
        xml = tmp_path / "coverage.xml"
        xml.write_text(_cobertura(10000, covered, 0, 0), encoding="utf-8")
        assert check_coverage_claim.main([str(xml)]) == (0 if ok else 1)

    def test_contributing_carries_a_readable_figure(self):
        text = check_coverage_claim.CONTRIBUTING.read_text(encoding="utf-8")
        assert 50.0 < check_coverage_claim.recorded_percent(text) <= 100.0


CHANGELOG_BASE = """# Changelog

## [Unreleased]

## [1.0.0] - 2026-01-01

- Old entry.
"""


def _git(repo, *args):
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "pytcl").mkdir()
    (tmp_path / "pytcl" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG_BASE, encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "switch", "-q", "-c", "feature")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _commit(repo, files):
    for name, text in files.items():
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "change")


class TestChangelogEntry:
    def test_package_change_without_an_entry_fails(self, repo):
        _commit(repo, {"pytcl/mod.py": "x = 2\n"})
        assert check_changelog_entry.main(["main", "feature"]) == 1

    def test_package_change_with_an_unreleased_entry_passes(self, repo):
        text = CHANGELOG_BASE.replace(
            "## [Unreleased]\n", "## [Unreleased]\n\n- x is 2. What to do: nothing.\n"
        )
        _commit(repo, {"pytcl/mod.py": "x = 2\n", "CHANGELOG.md": text})
        assert check_changelog_entry.main(["main", "feature"]) == 0

    def test_an_entry_under_a_released_version_does_not_count(self, repo):
        text = CHANGELOG_BASE.replace("- Old entry.\n", "- Old entry.\n- Backdated.\n")
        _commit(repo, {"pytcl/mod.py": "x = 2\n", "CHANGELOG.md": text})
        assert check_changelog_entry.main(["main", "feature"]) == 1

    def test_a_blank_line_is_not_an_entry(self, repo):
        text = CHANGELOG_BASE.replace("## [Unreleased]\n", "## [Unreleased]\n\n\n")
        _commit(repo, {"pytcl/mod.py": "x = 2\n", "CHANGELOG.md": text})
        assert check_changelog_entry.main(["main", "feature"]) == 1

    def test_a_change_outside_the_package_needs_no_entry(self, repo):
        _commit(repo, {"README.md": "changed\n"})
        assert check_changelog_entry.main(["main", "feature"]) == 0

    def test_unreleased_section_bounds(self):
        section = check_changelog_entry.unreleased_lines(CHANGELOG_BASE)
        lines = CHANGELOG_BASE.splitlines()
        assert [lines[n - 1] for n in section] == [""]

    def test_unreleased_heading_must_exist(self):
        with pytest.raises(ValueError, match="Unreleased"):
            check_changelog_entry.unreleased_lines("# Changelog\n\n## [1.0.0]\n")

    def test_repository_changelog_has_an_unreleased_section(self):
        text = (Path(__file__).resolve().parents[2] / "CHANGELOG.md").read_text(
            encoding="utf-8"
        )
        check_changelog_entry.unreleased_lines(text)
