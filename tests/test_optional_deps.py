"""
Tests for optional dependencies management system.
"""

from unittest.mock import MagicMock, patch

import pytest

from pytcl.core.exceptions import DependencyError
from pytcl.core.optional_deps import (
    PACKAGE_EXTRAS,
    PACKAGE_FEATURES,
    LazyModule,
    _clear_cache,
    check_dependencies,
    import_optional,
    is_available,
    requires,
)


class TestIsAvailable:
    """Tests for is_available function."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_available_package(self):
        """Test that numpy (a required package) is detected as available."""
        assert is_available("numpy") is True

    def test_unavailable_package(self):
        """Test that a non-existent package is detected as unavailable."""
        assert is_available("definitely_not_a_real_package_xyz123") is False

    def test_result_is_cached(self):
        """Test that availability results are cached."""
        _clear_cache()

        # First call should import
        result1 = is_available("numpy")

        # Patch importlib to verify it's not called again
        with patch("pytcl.core.optional_deps.importlib") as mock_importlib:
            result2 = is_available("numpy")

        # Should return same result without importing again
        assert result1 == result2
        mock_importlib.import_module.assert_not_called()

    def test_clear_cache(self):
        """Test that _clear_cache clears the cache."""
        is_available("numpy")
        _clear_cache()

        with patch("pytcl.core.optional_deps.importlib.import_module") as mock_import:
            mock_import.return_value = MagicMock()
            is_available("numpy")

        # After clearing cache, import should be called again
        mock_import.assert_called_once_with("numpy")


class TestImportOptional:
    """Tests for import_optional function."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_import_available_module(self):
        """Test importing an available module."""
        np = import_optional("numpy")
        assert hasattr(np, "array")

    def test_import_unavailable_module(self):
        """Test importing an unavailable module raises DependencyError."""
        with pytest.raises(DependencyError) as exc_info:
            import_optional("nonexistent_package_xyz")

        assert "nonexistent_package_xyz" in str(exc_info.value)
        assert exc_info.value.package == "nonexistent_package_xyz"

    def test_import_with_extra(self):
        """Test that extra is included in install command."""
        with pytest.raises(DependencyError) as exc_info:
            import_optional(
                "nonexistent_package_xyz",
                extra="visualization",
            )

        assert "pip install pytcl[visualization]" in exc_info.value.install_command

    def test_import_with_feature(self):
        """Test that feature is included in error message."""
        with pytest.raises(DependencyError) as exc_info:
            import_optional(
                "nonexistent_package_xyz",
                feature="amazing feature",
            )

        assert "amazing feature" in str(exc_info.value)
        assert exc_info.value.feature == "amazing feature"

    def test_import_submodule(self):
        """Test importing a submodule."""
        np_random = import_optional("numpy.random")
        assert hasattr(np_random, "random")

    def test_package_extracted_from_module_name(self):
        """Test that package is extracted from module name if not provided."""
        with pytest.raises(DependencyError) as exc_info:
            import_optional("nonexistent.submodule.deep")

        assert exc_info.value.package == "nonexistent"


class TestRequiresDecorator:
    """Tests for the @requires decorator."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_decorator_with_available_package(self):
        """Test decorator passes when package is available."""

        @requires("numpy")
        def my_function():
            return "success"

        assert my_function() == "success"

    def test_decorator_with_unavailable_package(self):
        """Test decorator raises when package is unavailable."""

        @requires("nonexistent_package_xyz")
        def my_function():
            return "success"

        with pytest.raises(DependencyError) as exc_info:
            my_function()

        assert "nonexistent_package_xyz" in str(exc_info.value)

    def test_decorator_with_multiple_packages(self):
        """Test decorator with multiple packages."""

        @requires("numpy", "nonexistent_package_xyz")
        def my_function():
            return "success"

        with pytest.raises(DependencyError) as exc_info:
            my_function()

        # Should report the missing package
        assert "nonexistent_package_xyz" in str(exc_info.value)

    def test_decorator_with_extra(self):
        """Test decorator includes extra in install command."""

        @requires("nonexistent_package_xyz", extra="visualization")
        def my_function():
            return "success"

        with pytest.raises(DependencyError) as exc_info:
            my_function()

        assert "pip install pytcl[visualization]" in exc_info.value.install_command

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""

        @requires("numpy")
        def my_function():
            """My docstring."""
            return "success"

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_decorator_with_arguments(self):
        """Test decorator works with function arguments."""

        @requires("numpy")
        def add_numbers(a, b):
            return a + b

        assert add_numbers(1, 2) == 3

    def test_decorator_with_kwargs(self):
        """Test decorator works with keyword arguments."""

        @requires("numpy")
        def add_numbers(a, b=10):
            return a + b

        assert add_numbers(5, b=20) == 25


class TestCheckDependencies:
    """Tests for check_dependencies function."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_check_available_package(self):
        """Test checking an available package doesn't raise."""
        check_dependencies("numpy")  # Should not raise

    def test_check_unavailable_package(self):
        """Test checking an unavailable package raises."""
        with pytest.raises(DependencyError) as exc_info:
            check_dependencies("nonexistent_package_xyz")

        assert "nonexistent_package_xyz" in str(exc_info.value)

    def test_check_multiple_packages_all_available(self):
        """Test checking multiple available packages."""
        check_dependencies("numpy", "sys")  # Should not raise

    def test_check_multiple_packages_some_missing(self):
        """Test checking multiple packages when some are missing."""
        with pytest.raises(DependencyError) as exc_info:
            check_dependencies("numpy", "nonexistent_package_xyz")

        assert "nonexistent_package_xyz" in str(exc_info.value)

    def test_check_with_extra(self):
        """Test extra is included in install command."""
        with pytest.raises(DependencyError) as exc_info:
            check_dependencies("nonexistent_package_xyz", extra="astronomy")

        assert "pip install pytcl[astronomy]" in exc_info.value.install_command


class TestLazyModule:
    """Tests for LazyModule class."""

    def setup_method(self):
        """Clear cache before each test."""
        _clear_cache()

    def test_lazy_load_available_module(self):
        """Test lazy loading an available module."""
        np_lazy = LazyModule("numpy")

        # Module should not be loaded yet
        assert np_lazy._module is None

        # Access an attribute to trigger load
        result = np_lazy.array([1, 2, 3])

        # Module should now be loaded
        assert np_lazy._module is not None
        assert list(result) == [1, 2, 3]

    def test_lazy_load_unavailable_module(self):
        """Test lazy loading an unavailable module raises on access."""
        lazy = LazyModule("nonexistent_package_xyz", feature="testing")

        # Should not raise yet
        assert lazy._module is None

        # Should raise when accessing attribute
        with pytest.raises(DependencyError):
            _ = lazy.some_attribute

    def test_lazy_module_dir(self):
        """Test __dir__ returns module attributes when available."""
        np_lazy = LazyModule("numpy")
        dir_result = dir(np_lazy)

        assert "array" in dir_result
        assert "zeros" in dir_result

    def test_lazy_module_dir_unavailable(self):
        """Test __dir__ returns empty list when module unavailable."""
        lazy = LazyModule("nonexistent_package_xyz")
        dir_result = dir(lazy)

        assert dir_result == []


class TestPackageConfiguration:
    """Tests for package configuration."""

    def test_package_extras_has_known_packages(self):
        """Test that PACKAGE_EXTRAS contains expected packages."""
        assert "plotly" in PACKAGE_EXTRAS
        assert "pywt" in PACKAGE_EXTRAS
        assert "jplephem" in PACKAGE_EXTRAS

    def test_package_extras_format(self):
        """Test that PACKAGE_EXTRAS values are (extra, pip_package) tuples."""
        for package, (extra, pip_package) in PACKAGE_EXTRAS.items():
            assert isinstance(extra, str)
            assert isinstance(pip_package, str)

    def test_package_features_has_known_packages(self):
        """Test that PACKAGE_FEATURES contains expected packages."""
        assert "plotly" in PACKAGE_FEATURES
        assert "pywt" in PACKAGE_FEATURES
        assert "jplephem" in PACKAGE_FEATURES

    def test_package_features_values_are_descriptive(self):
        """Test that PACKAGE_FEATURES values are descriptive strings."""
        for package, feature in PACKAGE_FEATURES.items():
            assert isinstance(feature, str)
            assert len(feature) > 0


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


class TestIntegrationWithCore:
    """Test integration with pytcl.core module."""

    def test_exports_from_core(self):
        """Test that optional deps are exported from pytcl.core."""
        from pytcl.core import (
            LazyModule,
            check_dependencies,
            import_optional,
            is_available,
            requires,
        )

        assert callable(is_available)
        assert callable(import_optional)
        assert callable(requires)
        assert callable(check_dependencies)
        assert LazyModule is not None
