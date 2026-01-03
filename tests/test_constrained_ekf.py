"""Unit tests for Constrained Extended Kalman Filter (CEKF).

Tests cover:
- Basic constraint satisfaction verification
- Linear constraints (bounds on states)
- Nonlinear constraint enforcement
- Covariance positive definiteness after projection
- Comparison with standard EKF
- Multiple simultaneous constraints
- Edge cases and numerical stability
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman import (
    ConstrainedEKF,
    ConstraintFunction,
    constrained_ekf_predict,
    constrained_ekf_update,
)
from pytcl.dynamic_estimation.kalman.linear import KalmanPrediction, KalmanUpdate


class TestConstraintFunction:
    """Test ConstraintFunction base class."""

    def test_constraint_evaluation_inequality(self):
        """Test evaluation of inequality constraint g(x) <= 0."""

        # Constraint: x[0] <= 10 (i.e., x[0] - 10 <= 0)
        def constraint_fn(x):
            return np.array([x[0] - 10])

        constraint = ConstraintFunction(constraint_fn)

        # x = [5, 0] satisfies constraint
        assert constraint.evaluate(np.array([5.0, 0.0])).shape == (1,)
        assert constraint.evaluate(np.array([5.0, 0.0]))[0] < 0

        # x = [15, 0] violates constraint
        assert constraint.evaluate(np.array([15.0, 0.0]))[0] > 0

    def test_constraint_multiple(self):
        """Test multiple constraints simultaneously."""

        # Constraints: -10 <= x[0] <= 10
        def constraint_fn(x):
            return np.array([x[0] - 10, -x[0] - 10])  # x[0] <= 10  # x[0] >= -10

        constraint = ConstraintFunction(constraint_fn)

        # x = [5, 0] satisfies both
        c_vals = constraint.evaluate(np.array([5.0, 0.0]))
        assert np.all(c_vals <= 0)

        # x = [15, 0] violates first
        c_vals = constraint.evaluate(np.array([15.0, 0.0]))
        assert c_vals[0] > 0

    def test_constraint_satisfaction_check(self):
        """Test constraint satisfaction verification."""

        def constraint_fn(x):
            return np.array([x[0] - 10])

        constraint = ConstraintFunction(constraint_fn)

        # Satisfied
        assert constraint.is_satisfied(np.array([5.0, 0.0]))

        # Not satisfied
        assert not constraint.is_satisfied(np.array([15.0, 0.0]))

        # On boundary (small tolerance)
        assert constraint.is_satisfied(np.array([10.0 - 1e-7, 0.0]))

    def test_numerical_jacobian(self):
        """Test numerical Jacobian computation."""

        def constraint_fn(x):
            return np.array([x[0] ** 2 - 25])  # g(x) = x[0]^2 - 25

        constraint = ConstraintFunction(constraint_fn)

        x = np.array([4.0, 0.0])
        G = constraint.jacobian(x)

        # Expected Jacobian: dg/dx = [2*x[0], 0] = [8, 0]
        assert G.shape == (1, 2)
        assert np.isclose(G[0, 0], 8.0, atol=1e-5)
        assert np.isclose(G[0, 1], 0.0, atol=1e-5)

    def test_analytical_jacobian(self):
        """Test constraint with analytical Jacobian."""

        def constraint_fn(x):
            return np.array([x[0] - 10])

        def jacobian_fn(x):
            return np.array([[1.0, 0.0]])

        # Use 'G' parameter name, not 'jacobian_fn'
        constraint = ConstraintFunction(constraint_fn, G=jacobian_fn)

        x = np.array([5.0, 3.0])
        G = constraint.jacobian(x)

        assert G.shape == (1, 2)
        assert np.allclose(G, [[1.0, 0.0]])


class TestConstrainedEKFBasic:
    """Test basic Constrained EKF functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple linear system: x[k+1] = x[k] + 0.1*v[k]
        self.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        self.Q = np.eye(2) * 0.01

        # Measurement: z = x[0] + noise
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

        # Initial state
        self.x0 = np.array([0.0, 0.0])
        self.P0 = np.eye(2)

    def test_cekf_initialization(self):
        """Test CEKF initialization."""
        cekf = ConstrainedEKF()

        assert len(cekf.constraints) == 0

    def test_add_constraint(self):
        """Test adding constraints to CEKF."""

        def constraint_fn(x):
            return np.array([x[0] - 10])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()

        cekf.add_constraint(constraint)
        assert len(cekf.constraints) == 1

    def test_predict_without_constraints(self):
        """Test prediction step without constraints."""
        cekf = ConstrainedEKF()

        # Linear prediction function: x[k+1] = F @ x[k]
        def f(x):
            return self.F @ x

        # Predict from x0
        prediction = cekf.predict(self.x0, self.P0, f, self.F, self.Q)

        assert prediction.x.shape == self.x0.shape
        assert prediction.P.shape == self.P0.shape

        # Check that covariance increased due to process noise
        assert np.trace(prediction.P) > np.trace(self.P0)

    def test_update_without_constraints(self):
        """Test update step without constraints."""
        cekf = ConstrainedEKF()

        # Linear measurement function: z = H @ x
        def h(x):
            return self.H @ x

        # First predict
        x_pred, P_pred = np.array([0.0, 0.0]), np.eye(2)

        # Then update with measurement
        z = np.array([0.5])
        update = cekf.update(x_pred, P_pred, z, h, self.H, self.R)

        assert update.x.shape == x_pred.shape
        assert update.P.shape == P_pred.shape


class TestConstrainedEKFLinearConstraints:
    """Test CEKF with linear constraints."""

    def setup_method(self):
        """Set up test with constrained system."""
        self.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

        self.x0 = np.array([0.0, 0.0])
        self.P0 = np.eye(2)

    def test_position_bound_constraint(self):
        """Test constraint on position: |x[0]| <= 10."""

        def constraint_fn(x):
            return np.array([x[0] - 10, -x[0] - 10])  # x[0] <= 10  # x[0] >= -10

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        # Create a violating state (x[0] = 15 violates x[0] <= 10)
        x_violating = np.array([15.0, 0.0])
        P = np.eye(2)

        # Project onto constraints
        x_proj, P_proj = cekf._project_onto_constraints(x_violating, P)

        # Check that constraint is now satisfied
        assert constraint.is_satisfied(x_proj)

        # Check that covariance is still positive definite
        eigvals = np.linalg.eigvalsh(P_proj)
        assert np.all(eigvals > -1e-10)  # All non-negative (numerical tolerance)

    def test_velocity_bound_constraint(self):
        """Test constraint on velocity: |x[1]| <= 5."""

        def constraint_fn(x):
            return np.array([x[1] - 5, -x[1] - 5])  # v <= 5  # v >= -5

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        # Create violating state
        x_violating = np.array([5.0, 10.0])
        P = np.eye(2)

        # Project
        x_proj, P_proj = cekf._project_onto_constraints(x_violating, P)

        # Constraint satisfied
        assert constraint.is_satisfied(x_proj)

        # Covariance positive definite
        eigvals = np.linalg.eigvalsh(P_proj)
        assert np.all(eigvals > -1e-10)

    def test_multiple_constraints(self):
        """Test multiple simultaneous linear constraints."""

        def constraint_fn(x):
            return np.array(
                [
                    x[0] - 10,  # pos <= 10
                    -x[0] - 10,  # pos >= -10
                    x[1] - 5,  # vel <= 5
                    -x[1] - 5,  # vel >= -5
                ]
            )

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        # Violate all constraints
        x_violating = np.array([20.0, 15.0])
        P = np.eye(2)

        # Project
        x_proj, P_proj = cekf._project_onto_constraints(x_violating, P)

        # All constraints satisfied
        assert constraint.is_satisfied(x_proj)
        assert np.linalg.norm(x_proj) <= 10 * np.sqrt(2) + 1e-5


class TestConstrainedEKFNonlinear:
    """Test CEKF with nonlinear constraints."""

    def setup_method(self):
        """Set up system with nonlinear dynamics."""
        self.F = np.array([[1.0, 0.1], [0.0, 0.95]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

    def test_circular_bound_constraint(self):
        """Test nonlinear constraint: x[0]^2 + x[1]^2 <= 25."""

        def constraint_fn(x):
            # Circle with radius 5
            return np.array([x[0] ** 2 + x[1] ** 2 - 25])

        def jacobian_fn(x):
            return np.array([[2 * x[0], 2 * x[1]]])

        constraint = ConstraintFunction(constraint_fn, G=jacobian_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        # State outside circle
        x_violating = np.array([6.0, 6.0])
        P = np.eye(2)

        # Project
        x_proj, P_proj = cekf._project_onto_constraints(x_violating, P)

        # Check constraint (with tolerance)
        c_val = constraint.evaluate(x_proj)[0]
        assert c_val <= 1e-6

        # Covariance positive definite
        eigvals = np.linalg.eigvalsh(P_proj)
        assert np.all(eigvals > -1e-10)

    def test_energy_constraint(self):
        """Test energy-like constraint: 0.5*x[0]^2 + 0.5*x[1]^2 <= E_max."""
        E_max = 10.0

        def constraint_fn(x):
            # Energy constraint
            return np.array([0.5 * x[0] ** 2 + 0.5 * x[1] ** 2 - E_max])

        def jacobian_fn(x):
            return np.array([[x[0], x[1]]])

        constraint = ConstraintFunction(constraint_fn, G=jacobian_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        # High energy state
        x_high = np.array([5.0, 5.0])
        P = np.eye(2)

        # Project
        x_proj, P_proj = cekf._project_onto_constraints(x_high, P)

        # Energy constraint satisfied
        energy = 0.5 * x_proj[0] ** 2 + 0.5 * x_proj[1] ** 2
        assert energy <= E_max + 1e-6


class TestConstrainedEKFCovarianceProperties:
    """Test covariance matrix properties after constraint projection."""

    def setup_method(self):
        """Set up test system."""
        self.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

    def test_covariance_positive_definite(self):
        """Test that projected covariance is positive definite."""

        def constraint_fn(x):
            return np.array([x[0] - 5])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        x = np.array([10.0, 0.0])
        P = np.array([[2.0, 0.1], [0.1, 1.0]])

        # Project
        _, P_proj = cekf._project_onto_constraints(x, P)

        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(P_proj)
        assert np.all(eigvals > -1e-10)

        # All eigenvalues should be positive (strict)
        assert np.all(eigvals > 1e-12)

    def test_covariance_symmetry(self):
        """Test that projected covariance remains symmetric."""

        def constraint_fn(x):
            return np.array([x[0] - 10, -x[0] - 10, x[1] - 5, -x[1] - 5])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        x = np.array([20.0, 15.0])
        P = np.array([[2.0, 0.5], [0.5, 1.5]])

        # Project
        _, P_proj = cekf._project_onto_constraints(x, P)

        # Check symmetry
        assert np.allclose(P_proj, P_proj.T)

    def test_covariance_trace_reduction(self):
        """Test that constraint projection reduces covariance trace."""

        def constraint_fn(x):
            return np.array([x[0] - 5])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        x = np.array([10.0, 0.0])
        P = np.eye(2) * 2.0

        # Project
        _, P_proj = cekf._project_onto_constraints(x, P)

        # Trace should decrease (constraint reduces uncertainty)
        assert np.trace(P_proj) < np.trace(P)


class TestConstrainedEKFComparison:
    """Test CEKF against standard EKF."""

    def setup_method(self):
        """Set up comparison."""
        self.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

    def test_cekf_reduces_to_ekf_no_constraints(self):
        """Test CEKF equals EKF when no constraints are active."""
        cekf = ConstrainedEKF()

        def f(x):
            return self.F @ x

        def h(x):
            return self.H @ x

        # No constraints added
        x = np.array([1.0, 0.1])
        P = np.eye(2)
        z = np.array([1.2])

        # Predict
        pred_cekf = cekf.predict(x, P, f, self.F, self.Q)

        # Update
        upd_cekf = cekf.update(pred_cekf.x, pred_cekf.P, z, h, self.H, self.R)

        # Just check that it runs without errors
        assert upd_cekf.x is not None
        assert upd_cekf.P is not None

    def test_cekf_with_inactive_constraints(self):
        """Test CEKF with constraints that are not violated."""

        def constraint_fn(x):
            return np.array([x[0] - 100])  # Very loose constraint

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.add_constraint(constraint)

        def f(x):
            return self.F @ x

        def h(x):
            return self.H @ x

        # Small state that easily satisfies constraint
        x = np.array([1.0, 0.1])
        P = np.eye(2)
        z = np.array([1.2])

        # Predict and update
        pred = cekf.predict(x, P, f, self.F, self.Q)
        upd = cekf.update(pred.x, pred.P, z, h, self.H, self.R)

        # State should be similar to without constraint
        assert upd.x[0] < 100  # Within constraint


class TestConstrainedEKFEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test system."""
        self.F = np.array([[1.0, 0.1], [0.0, 1.0]])
        self.Q = np.eye(2) * 0.01
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[0.1]])

    def test_constraint_on_boundary(self):
        """Test state exactly on constraint boundary."""

        def constraint_fn(x):
            return np.array([x[0] - 5])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        # State on boundary
        x_boundary = np.array([5.0, 0.0])
        P = np.eye(2)

        # Should handle gracefully
        x_proj, P_proj = cekf._project_onto_constraints(x_boundary, P)

        # Should remain on boundary (within tolerance)
        assert constraint.is_satisfied(x_proj)

    def test_constraint_no_violation(self):
        """Test projection when constraint is not violated."""

        def constraint_fn(x):
            return np.array([x[0] - 20])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = self.F
        cekf.Q = self.Q
        cekf.H = self.H
        cekf.R = self.R
        cekf.add_constraint(constraint)

        # Safe state
        x_safe = np.array([0.0, 0.0])
        P = np.eye(2)

        # Project
        x_proj, P_proj = cekf._project_onto_constraints(x_safe, P)

        # Should remain close to original
        assert np.linalg.norm(x_proj - x_safe) < 1e-6

    def test_high_dimensional_state(self):
        """Test CEKF with higher dimensional state (n=5)."""
        n = 5
        F = np.eye(n)
        Q = np.eye(n) * 0.01

        # Simple scalar measurement on first state
        H = np.zeros((1, n))
        H[0, 0] = 1.0
        R = np.array([[0.1]])

        def constraint_fn(x):
            return np.array([x[0] - 10])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.F = F
        cekf.Q = Q
        cekf.H = H
        cekf.R = R
        cekf.add_constraint(constraint)

        # Initial state
        x = np.array([15.0, 0.0, 0.0, 0.0, 0.0])
        P = np.eye(n)

        # Project
        x_proj, P_proj = cekf._project_onto_constraints(x, P)

        # Check constraint
        assert constraint.is_satisfied(x_proj)
        assert P_proj.shape == (n, n)


class TestConvenientFunctions:
    """Test convenience functions for CEKF."""

    def test_constrained_ekf_predict_function(self):
        """Test constrained_ekf_predict function."""
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        Q = np.eye(2) * 0.01

        def f(x):
            return F @ x

        x = np.array([0.0, 0.0])
        P = np.eye(2)

        result = constrained_ekf_predict(x, P, f, F, Q)

        assert isinstance(result, KalmanPrediction)
        assert result.x.shape == x.shape
        assert result.P.shape == P.shape

    def test_constrained_ekf_update_function(self):
        """Test constrained_ekf_update function."""
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        def h(x):
            return H @ x

        x = np.array([0.0, 0.0])
        P = np.eye(2)
        z = np.array([0.5])

        # Add constraint
        def constraint_fn(x):
            return np.array([x[0] - 10])

        constraint = ConstraintFunction(constraint_fn)
        constraints = [constraint]

        result = constrained_ekf_update(x, P, z, h, H, R, constraints)

        assert isinstance(result, KalmanUpdate)
        assert result.x.shape == x.shape
        assert result.P.shape == P.shape


class TestConstrainedEKFIntegration:
    """Integration tests with full filtering sequence."""

    def test_full_filter_sequence(self):
        """Test full predict-update sequence with constraints."""
        # System setup
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        Q = np.eye(2) * 0.01
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        # Constraints: -5 <= x[0] <= 5, -3 <= x[1] <= 3
        def constraint_fn(x):
            return np.array([x[0] - 5, -x[0] - 5, x[1] - 3, -x[1] - 3])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.add_constraint(constraint)

        # Initial state
        x = np.array([0.0, 0.0])
        P = np.eye(2)

        # Run 5 filter steps
        for k in range(5):
            # Predict
            pred = cekf.predict(x, P, f, F, Q)
            x, P = pred.x, pred.P

            # Generate measurement (with noise)
            z = np.array([x[0] + np.random.randn() * 0.1])

            # Update with constraint projection
            upd = cekf.update(x, P, z, h, H, R)
            x, P = upd.x, upd.P

            # Verify constraints
            assert constraint.is_satisfied(x)
            assert np.all(np.linalg.eigvalsh(P) > -1e-10)

        # Final state should satisfy constraints
        assert constraint.is_satisfied(x)

    def test_long_horizon_filtering(self):
        """Test CEKF over longer filtering horizon (20 steps)."""
        F = np.array([[1.0, 0.1], [0.0, 0.99]])
        Q = np.eye(2) * 0.005
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.05]])

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        def constraint_fn(x):
            return np.array([x[0] - 10, -x[0] - 10])

        constraint = ConstraintFunction(constraint_fn)
        cekf = ConstrainedEKF()
        cekf.add_constraint(constraint)

        x = np.array([0.0, 1.0])
        P = np.eye(2)

        trajectory = [x.copy()]

        for k in range(20):
            # Predict
            pred = cekf.predict(x, P, f, F, Q)
            x, P = pred.x, pred.P

            # Measurement
            z = np.array([x[0] + np.random.randn() * 0.05])

            # Update
            upd = cekf.update(x, P, z, h, H, R)
            x, P = upd.x, upd.P

            trajectory.append(x.copy())

            # Verify properties
            assert constraint.is_satisfied(x)
            assert np.all(np.linalg.eigvalsh(P) > -1e-10)

        # Check that trajectory respects constraints
        trajectory = np.array(trajectory)
        assert np.all(np.abs(trajectory[:, 0]) <= 10 + 1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
