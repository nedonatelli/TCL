"""Tests for the custom exception hierarchy in pytcl.core.exceptions."""

import pytest

from pytcl.core.exceptions import (
    ComputationError,
    ConfigurationError,
    ConvergenceError,
    DataError,
    DependencyError,
    DimensionError,
    EmptyContainerError,
    FormatError,
    MethodError,
    NumericalError,
    ParameterError,
    ParseError,
    RangeError,
    SingularMatrixError,
    StateError,
    TCLError,
    UninitializedError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test the exception class hierarchy."""

    def test_tcl_error_is_base(self) -> None:
        """TCLError should be the base for all TCL exceptions."""
        assert issubclass(ValidationError, TCLError)
        assert issubclass(ComputationError, TCLError)
        assert issubclass(StateError, TCLError)
        assert issubclass(ConfigurationError, TCLError)
        assert issubclass(DataError, TCLError)

    def test_validation_error_hierarchy(self) -> None:
        """ValidationError should have proper subclasses."""
        assert issubclass(ValidationError, ValueError)
        assert issubclass(DimensionError, ValidationError)
        assert issubclass(ParameterError, ValidationError)
        assert issubclass(RangeError, ValidationError)

    def test_computation_error_hierarchy(self) -> None:
        """ComputationError should have proper subclasses."""
        assert issubclass(ComputationError, RuntimeError)
        assert issubclass(ConvergenceError, ComputationError)
        assert issubclass(NumericalError, ComputationError)
        assert issubclass(SingularMatrixError, ComputationError)

    def test_state_error_hierarchy(self) -> None:
        """StateError should have proper subclasses."""
        assert issubclass(UninitializedError, StateError)
        assert issubclass(EmptyContainerError, StateError)

    def test_configuration_error_hierarchy(self) -> None:
        """ConfigurationError should have proper subclasses."""
        assert issubclass(MethodError, ConfigurationError)
        assert issubclass(MethodError, ValueError)
        assert issubclass(DependencyError, ConfigurationError)
        assert issubclass(DependencyError, ImportError)

    def test_data_error_hierarchy(self) -> None:
        """DataError should have proper subclasses."""
        assert issubclass(FormatError, DataError)
        assert issubclass(FormatError, ValueError)
        assert issubclass(ParseError, DataError)
        assert issubclass(ParseError, ValueError)


class TestTCLError:
    """Tests for the base TCLError class."""

    def test_basic_message(self) -> None:
        """TCLError should store and display message."""
        err = TCLError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"

    def test_with_details(self) -> None:
        """TCLError should include details in string representation."""
        err = TCLError("Error occurred", details={"value": 42, "file": "test.py"})
        assert "value=42" in str(err)
        assert "file=test.py" in str(err)
        assert err.details == {"value": 42, "file": "test.py"}

    def test_can_be_caught_as_exception(self) -> None:
        """TCLError should be catchable as Exception."""
        with pytest.raises(Exception):
            raise TCLError("test")


class TestValidationError:
    """Tests for ValidationError and subclasses."""

    def test_validation_error_basic(self) -> None:
        """ValidationError should work with basic parameters."""
        err = ValidationError("Invalid input")
        assert str(err) == "Invalid input"
        assert isinstance(err, ValueError)
        assert isinstance(err, TCLError)

    def test_validation_error_with_parameters(self) -> None:
        """ValidationError should include parameter info."""
        err = ValidationError(
            "Invalid matrix",
            parameter="P",
            expected="3x3 matrix",
            actual="2x4 array",
        )
        assert err.parameter == "P"
        assert err.expected == "3x3 matrix"
        assert err.actual == "2x4 array"

    def test_dimension_error(self) -> None:
        """DimensionError should include shape info."""
        err = DimensionError(
            "Shape mismatch",
            expected_shape=(3, 3),
            actual_shape=(2, 4),
            parameter="P",
        )
        assert err.expected_shape == (3, 3)
        assert err.actual_shape == (2, 4)
        assert err.parameter == "P"

    def test_parameter_error(self) -> None:
        """ParameterError should include constraint info."""
        err = ParameterError(
            "Invalid value",
            parameter="variance",
            value=-1.0,
            constraint="must be > 0",
        )
        assert err.parameter == "variance"
        assert err.value == -1.0
        assert err.constraint == "must be > 0"

    def test_range_error(self) -> None:
        """RangeError should include range info."""
        err = RangeError(
            "Out of range",
            parameter="e",
            value=1.5,
            min_value=0.0,
            max_value=1.0,
        )
        assert err.value == 1.5
        assert err.min_value == 0.0
        assert err.max_value == 1.0


class TestComputationError:
    """Tests for ComputationError and subclasses."""

    def test_computation_error_basic(self) -> None:
        """ComputationError should work with basic parameters."""
        err = ComputationError("Computation failed", algorithm="eigenvalue")
        assert err.algorithm == "eigenvalue"
        assert isinstance(err, RuntimeError)

    def test_convergence_error(self) -> None:
        """ConvergenceError should include iteration info."""
        err = ConvergenceError(
            "Did not converge",
            algorithm="Newton-Raphson",
            iterations=100,
            max_iterations=100,
            residual=1e-5,
            tolerance=1e-12,
        )
        assert err.algorithm == "Newton-Raphson"
        assert err.iterations == 100
        assert err.max_iterations == 100
        assert err.residual == 1e-5
        assert err.tolerance == 1e-12

    def test_numerical_error(self) -> None:
        """NumericalError should include numerical info."""
        err = NumericalError(
            "Ill-conditioned",
            operation="matrix inversion",
            condition_number=1e16,
        )
        assert err.operation == "matrix inversion"
        assert err.condition_number == 1e16

    def test_singular_matrix_error(self) -> None:
        """SingularMatrixError should include matrix info."""
        err = SingularMatrixError(
            "Matrix is singular",
            matrix_name="P",
            determinant=1e-20,
        )
        assert err.matrix_name == "P"
        assert err.determinant == 1e-20


class TestStateError:
    """Tests for StateError and subclasses."""

    def test_state_error_basic(self) -> None:
        """StateError should work with state info."""
        err = StateError(
            "Invalid state",
            object_type="KalmanFilter",
            current_state="uninitialized",
            required_state="predicted",
        )
        assert err.object_type == "KalmanFilter"
        assert err.current_state == "uninitialized"
        assert err.required_state == "predicted"

    def test_uninitialized_error(self) -> None:
        """UninitializedError should set uninitialized state."""
        err = UninitializedError(
            "Not initialized",
            object_type="Tracker",
            required_initialization="call init()",
        )
        assert err.object_type == "Tracker"
        assert err.current_state == "uninitialized"

    def test_empty_container_error(self) -> None:
        """EmptyContainerError should indicate empty state."""
        err = EmptyContainerError(
            "Container is empty",
            container_type="RTree",
            operation="nearest query",
        )
        assert err.current_state == "empty"
        assert err.object_type == "RTree"


class TestConfigurationError:
    """Tests for ConfigurationError and subclasses."""

    def test_method_error(self) -> None:
        """MethodError should include method info."""
        err = MethodError(
            "Unknown method",
            method="invalid",
            valid_methods=["hungarian", "auction", "greedy"],
        )
        assert err.method == "invalid"
        assert err.valid_methods == ["hungarian", "auction", "greedy"]
        assert isinstance(err, ValueError)

    def test_dependency_error(self) -> None:
        """DependencyError should include dependency info."""
        err = DependencyError(
            "Missing dependency",
            package="plotly",
            feature="3D visualization",
            install_command="pip install plotly",
        )
        assert err.package == "plotly"
        assert err.feature == "3D visualization"
        assert err.install_command == "pip install plotly"
        assert isinstance(err, ImportError)


class TestDataError:
    """Tests for DataError and subclasses."""

    def test_format_error(self) -> None:
        """FormatError should include format info."""
        err = FormatError(
            "Invalid format",
            expected_format="69 characters",
            actual_format="65 characters",
        )
        assert err.expected_format == "69 characters"
        assert err.actual_format == "65 characters"

    def test_parse_error(self) -> None:
        """ParseError should include parse info."""
        err = ParseError(
            "Parse failed",
            data_type="TLE",
            position=68,
            reason="invalid checksum",
        )
        assert err.data_type == "TLE"
        assert err.position == 68
        assert err.reason == "invalid checksum"


class TestCatchingPatterns:
    """Test various exception catching patterns."""

    def test_catch_all_tcl_errors(self) -> None:
        """All exceptions should be catchable with TCLError."""
        exceptions = [
            ValidationError("test"),
            DimensionError("test"),
            ConvergenceError("test"),
            EmptyContainerError("test"),
            MethodError("test"),
            FormatError("test"),
        ]
        for exc in exceptions:
            with pytest.raises(TCLError):
                raise exc

    def test_catch_validation_as_value_error(self) -> None:
        """ValidationError subtypes should be catchable as ValueError."""
        with pytest.raises(ValueError):
            raise DimensionError("test")
        with pytest.raises(ValueError):
            raise ParameterError("test")
        with pytest.raises(ValueError):
            raise RangeError("test")

    def test_catch_computation_as_runtime_error(self) -> None:
        """ComputationError subtypes should be catchable as RuntimeError."""
        with pytest.raises(RuntimeError):
            raise ConvergenceError("test")
        with pytest.raises(RuntimeError):
            raise NumericalError("test")
        with pytest.raises(RuntimeError):
            raise SingularMatrixError("test")

    def test_catch_dependency_as_import_error(self) -> None:
        """DependencyError should be catchable as ImportError."""
        with pytest.raises(ImportError):
            raise DependencyError("test")
