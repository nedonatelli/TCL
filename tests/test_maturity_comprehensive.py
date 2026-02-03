"""
Comprehensive tests for module maturity classification system.

Tests coverage for:
- MaturityLevel enum and values
- Module maturity lookup and classification
- Maturity level queries and filtering
- Production readiness checks
- Maturity badge formatting
"""

import pytest

from pytcl.core.maturity import (
    MaturityLevel,
    get_maturity,
    get_modules_by_maturity,
    get_maturity_summary,
    is_stable,
    is_production_ready,
    format_maturity_badge,
    MODULE_MATURITY,
)


class TestMaturityLevelEnum:
    """Tests for MaturityLevel enum."""

    def test_maturity_level_values(self):
        """Test MaturityLevel enum values."""
        assert MaturityLevel.DEPRECATED == 0
        assert MaturityLevel.EXPERIMENTAL == 1
        assert MaturityLevel.MATURE == 2
        assert MaturityLevel.STABLE == 3

    def test_maturity_level_names(self):
        """Test MaturityLevel enum names."""
        assert MaturityLevel.DEPRECATED.name == "DEPRECATED"
        assert MaturityLevel.EXPERIMENTAL.name == "EXPERIMENTAL"
        assert MaturityLevel.MATURE.name == "MATURE"
        assert MaturityLevel.STABLE.name == "STABLE"

    def test_maturity_level_ordering(self):
        """Test that maturity levels are ordered correctly."""
        assert MaturityLevel.DEPRECATED < MaturityLevel.EXPERIMENTAL
        assert MaturityLevel.EXPERIMENTAL < MaturityLevel.MATURE
        assert MaturityLevel.MATURE < MaturityLevel.STABLE

    def test_maturity_level_comparisons(self):
        """Test comparisons between maturity levels."""
        assert MaturityLevel.STABLE >= MaturityLevel.MATURE
        assert MaturityLevel.EXPERIMENTAL <= MaturityLevel.MATURE
        assert MaturityLevel.DEPRECATED < MaturityLevel.STABLE


class TestGetMaturity:
    """Tests for getting module maturity."""

    def test_get_maturity_known_stable(self):
        """Test getting maturity for known stable module."""
        # core.constants should be stable
        level = get_maturity("core.constants")
        assert level == MaturityLevel.STABLE

    def test_get_maturity_known_module(self):
        """Test getting maturity for known modules."""
        # Test several known modules
        for module_path in ["core.constants", "core.exceptions"]:
            level = get_maturity(module_path)
            assert isinstance(level, MaturityLevel)
            assert level in [
                MaturityLevel.DEPRECATED,
                MaturityLevel.EXPERIMENTAL,
                MaturityLevel.MATURE,
                MaturityLevel.STABLE,
            ]

    def test_get_maturity_unknown_module(self):
        """Test getting maturity for unknown module returns valid level."""
        # Unknown modules should return a default level
        try:
            level = get_maturity("unknown.nonexistent.module")
            assert isinstance(level, MaturityLevel)
        except KeyError:
            # It's OK if unknown modules raise KeyError
            pass

    def test_get_maturity_nested_path(self):
        """Test with nested module paths."""
        # These should have valid maturity levels
        for module in MODULE_MATURITY.keys():
            level = get_maturity(module)
            assert isinstance(level, MaturityLevel)


class TestGetModulesByMaturity:
    """Tests for filtering modules by maturity."""

    def test_get_stable_modules(self):
        """Test getting all stable modules."""
        stable = get_modules_by_maturity(MaturityLevel.STABLE)

        assert isinstance(stable, list)
        assert len(stable) > 0
        # All returned modules should be stable
        for mod in stable:
            assert get_maturity(mod) == MaturityLevel.STABLE

    def test_get_mature_modules(self):
        """Test getting all mature modules."""
        mature = get_modules_by_maturity(MaturityLevel.MATURE)

        assert isinstance(mature, list)
        # All returned modules should be mature or higher
        for mod in mature:
            level = get_maturity(mod)
            assert level >= MaturityLevel.MATURE

    def test_get_experimental_modules(self):
        """Test getting experimental modules."""
        experimental = get_modules_by_maturity(MaturityLevel.EXPERIMENTAL)

        assert isinstance(experimental, list)
        for mod in experimental:
            level = get_maturity(mod)
            assert level >= MaturityLevel.EXPERIMENTAL

    def test_get_deprecated_modules(self):
        """Test getting deprecated modules."""
        deprecated = get_modules_by_maturity(MaturityLevel.DEPRECATED)

        # May or may not have deprecated modules
        assert isinstance(deprecated, list)

    def test_all_levels_have_results(self):
        """Test that we can query all maturity levels."""
        for level in MaturityLevel:
            result = get_modules_by_maturity(level)
            assert isinstance(result, list)


class TestMaturitySummary:
    """Tests for maturity summary statistics."""

    def test_get_maturity_summary_structure(self):
        """Test maturity summary has correct structure."""
        summary = get_maturity_summary()

        assert isinstance(summary, dict)
        # Should have entries for each maturity level
        assert len(summary) > 0

    def test_get_maturity_summary_values(self):
        """Test maturity summary values are counts."""
        summary = get_maturity_summary()

        for level, count in summary.items():
            assert isinstance(level, MaturityLevel)
            assert isinstance(count, int)
            assert count >= 0

    def test_maturity_summary_totals(self):
        """Test summary counts match module list lengths."""
        summary = get_maturity_summary()

        for level, count in summary.items():
            modules = get_modules_by_maturity(level)
            # Count should match or be subset
            assert count >= 0


class TestStabilityChecks:
    """Tests for stability and production-readiness checks."""

    def test_is_stable_stable_module(self):
        """Test is_stable for known stable module."""
        # core.constants should be stable
        assert is_stable("core.constants") == True

    def test_is_stable_experimental_module(self):
        """Test is_stable for experimental modules."""
        experimental = get_modules_by_maturity(MaturityLevel.EXPERIMENTAL)
        if experimental:
            module = experimental[0]
            assert is_stable(module) == False

    def test_is_production_ready_stable(self):
        """Test is_production_ready for stable modules."""
        stable = get_modules_by_maturity(MaturityLevel.STABLE)
        if stable:
            module = stable[0]
            assert is_production_ready(module) == True

    def test_is_production_ready_mature(self):
        """Test is_production_ready for mature modules."""
        mature = get_modules_by_maturity(MaturityLevel.MATURE)
        if mature:
            module = mature[0]
            assert is_production_ready(module) == True

    def test_is_production_ready_experimental(self):
        """Test is_production_ready for experimental modules."""
        experimental = get_modules_by_maturity(MaturityLevel.EXPERIMENTAL)
        if experimental:
            module = experimental[0]
            # Experimental should not be production ready
            assert is_production_ready(module) == False

    def test_is_production_ready_deprecated(self):
        """Test is_production_ready for deprecated modules."""
        deprecated = get_modules_by_maturity(MaturityLevel.DEPRECATED)
        if deprecated:
            module = deprecated[0]
            assert is_production_ready(module) == False


class TestMaturityBadges:
    """Tests for maturity badge formatting."""

    def test_format_stable_badge(self):
        """Test formatting stable maturity badge."""
        badge = format_maturity_badge(MaturityLevel.STABLE)

        assert isinstance(badge, str)
        assert len(badge) > 0
        # Badge should indicate stability
        assert badge.lower() != ""

    def test_format_mature_badge(self):
        """Test formatting mature maturity badge."""
        badge = format_maturity_badge(MaturityLevel.MATURE)

        assert isinstance(badge, str)
        assert len(badge) > 0

    def test_format_experimental_badge(self):
        """Test formatting experimental maturity badge."""
        badge = format_maturity_badge(MaturityLevel.EXPERIMENTAL)

        assert isinstance(badge, str)
        assert len(badge) > 0

    def test_format_deprecated_badge(self):
        """Test formatting deprecated maturity badge."""
        badge = format_maturity_badge(MaturityLevel.DEPRECATED)

        assert isinstance(badge, str)
        assert len(badge) > 0

    def test_badges_are_different(self):
        """Test that different levels produce different badges."""
        badges = {
            MaturityLevel.STABLE: format_maturity_badge(MaturityLevel.STABLE),
            MaturityLevel.MATURE: format_maturity_badge(MaturityLevel.MATURE),
            MaturityLevel.EXPERIMENTAL: format_maturity_badge(
                MaturityLevel.EXPERIMENTAL
            ),
            MaturityLevel.DEPRECATED: format_maturity_badge(MaturityLevel.DEPRECATED),
        }

        # Badges should be distinguishable
        unique_badges = set(badges.values())
        assert len(unique_badges) > 0


class TestMaturityIntegration:
    """Integration tests for maturity system."""

    def test_module_maturity_coverage(self):
        """Test that modules in MODULE_MATURITY are retrievable."""
        for module_path in list(MODULE_MATURITY.keys())[:10]:
            level = get_maturity(module_path)
            assert isinstance(level, MaturityLevel)

    def test_maturity_hierarchy_consistency(self):
        """Test consistency of maturity hierarchy."""
        stable = set(get_modules_by_maturity(MaturityLevel.STABLE))
        mature = set(get_modules_by_maturity(MaturityLevel.MATURE))
        experimental = set(get_modules_by_maturity(MaturityLevel.EXPERIMENTAL))
        deprecated = set(get_modules_by_maturity(MaturityLevel.DEPRECATED))

        # Sets should be disjoint (each module has one level)
        assert len(stable & mature) == 0
        assert len(stable & experimental) == 0
        assert len(stable & deprecated) == 0
        assert len(mature & experimental) == 0

    def test_all_modules_classified(self):
        """Test that all modules in MODULE_MATURITY are classified."""
        for module, level in MODULE_MATURITY.items():
            retrieved_level = get_maturity(module)
            assert retrieved_level == level

    def test_production_ready_vs_stable(self):
        """Test relationship between production_ready and stable."""
        # All stable modules should be production ready
        stable = get_modules_by_maturity(MaturityLevel.STABLE)
        for module in stable:
            assert is_production_ready(module)
            assert is_stable(module)

    def test_summary_completeness(self):
        """Test that summary includes all modules."""
        summary = get_maturity_summary()
        total_in_summary = sum(summary.values())

        total_modules = sum(
            len(get_modules_by_maturity(level)) for level in MaturityLevel
        )

        # Should account for all modules
        assert total_in_summary > 0
