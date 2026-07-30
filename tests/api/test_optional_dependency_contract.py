"""The error contract of the optional-dependency loader.

Extracted from tests/validation/test_optional_deps.py: these assert the shape
of what ``import_optional`` raises -- that a DependencyError carries package,
feature and install_command -- rather than checking a computation against a
reference. That is an API-surface concern, and it was the one cohesive group
in that file, so it moves cleanly.
"""

import pytest

from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import _clear_cache, import_optional


class TestDependencyErrorAttributes:
    """Tests for DependencyError attributes set by optional_deps."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_error_has_package_attribute(self):
        """Test that DependencyError has package attribute."""
        with pytest.raises(DependencyError) as exc_info:
            import_optional("nonexistent_pkg", package="my_package")

        assert exc_info.value.package == "my_package"

    def test_error_has_feature_attribute(self):
        """Test that DependencyError has feature attribute."""
        with pytest.raises(DependencyError) as exc_info:
            import_optional("nonexistent_pkg", feature="my_feature")

        assert exc_info.value.feature == "my_feature"

    def test_error_has_install_command_attribute(self):
        """Test that DependencyError has install_command attribute."""
        with pytest.raises(DependencyError) as exc_info:
            import_optional("nonexistent_pkg", extra="myextra")

        assert "pip install" in exc_info.value.install_command
