"""
Tests to boost coverage for low-coverage modules.

Tests cover:
- Matrix decompositions (chol_semi_def, tria, tria_sqrt, etc.)
- Continuous-time dynamics (drift/diffusion functions)
- Particle filters (bootstrap filter)
- Process noise models (Singer, coordinated turn)
- Interpolation functions
- Statistics distributions and estimators
- CFAR detection algorithms
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.basic_matrix.decompositions import (
    chol_semi_def,
    matrix_sqrt,
    null_space,
    pinv_truncated,
    range_space,
    rank_revealing_qr,
    tria,
    tria_sqrt,
)

# =============================================================================
# Matrix Decomposition Tests
# =============================================================================


class TestCholSemiDef:
    """Tests for semi-definite Cholesky decomposition."""

    def test_positive_definite(self):
        """Test with positive definite matrix."""
        A = np.array([[4, 2], [2, 3]])
        L = chol_semi_def(A)
        np.testing.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_positive_semidefinite(self):
        """Test with positive semi-definite (nearly singular) matrix."""
        # Near-singular matrix that triggers the eigenvalue decomposition path
        # but still has a well-defined Cholesky-like factor
        A = np.array([[4, 2, 0], [2, 1.0001, 0], [0, 0, 1]])
        L = chol_semi_def(A)
        # Check that L @ L.T approximates A
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, A, rtol=1e-3)

    def test_upper_triangular(self):
        """Test upper triangular output."""
        A = np.array([[4, 2], [2, 3]])
        R = chol_semi_def(A, upper=True)
        np.testing.assert_allclose(R.T @ R, A, rtol=1e-10)

    def test_non_square_raises(self):
        """Test non-square matrix raises error."""
        with pytest.raises(ValueError):
            chol_semi_def(np.array([[1, 2, 3], [4, 5, 6]]))


class TestTria:
    """Tests for triangular square root."""

    def test_tria_basic(self):
        """Test basic triangular factor."""
        A = np.array([[4, 2], [2, 3]])
        S = tria(A)
        np.testing.assert_allclose(S @ S.T, A, rtol=1e-10)
        # Should be lower triangular
        assert np.allclose(S, np.tril(S))


class TestTriaSqrt:
    """Tests for triangular square root of matrix products."""

    def test_single_matrix(self):
        """Test with single matrix."""
        A = np.random.randn(3, 4)
        S = tria_sqrt(A)
        np.testing.assert_allclose(S @ S.T, A @ A.T, rtol=1e-10)

    def test_two_matrices(self):
        """Test with two matrices."""
        A = np.random.randn(3, 4)
        B = np.random.randn(3, 2)
        S = tria_sqrt(A, B)
        expected = A @ A.T + B @ B.T
        np.testing.assert_allclose(S @ S.T, expected, rtol=1e-10)

    def test_shape_mismatch_raises(self):
        """Test row count mismatch raises error."""
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 2)  # Wrong number of rows
        with pytest.raises(ValueError):
            tria_sqrt(A, B)


class TestPinvTruncated:
    """Tests for truncated pseudo-inverse."""

    def test_full_rank(self):
        """Test with full rank matrix."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        A_pinv = pinv_truncated(A)
        # Check pseudo-inverse property
        np.testing.assert_allclose(A @ A_pinv @ A, A, rtol=1e-10)

    def test_rank_truncation(self):
        """Test with explicit rank truncation."""
        # Rank 2 matrix
        A = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])  # Rank 1
        A_pinv = pinv_truncated(A, rank=1)
        assert A_pinv.shape == (3, 3)

    def test_tolerance_truncation(self):
        """Test with tolerance-based truncation."""
        A = np.diag([1, 0.1, 0.001])
        A_pinv = pinv_truncated(A, tol=0.01)
        # Should effectively invert only the largest singular values
        assert A_pinv.shape == (3, 3)


class TestMatrixSqrt:
    """Tests for principal matrix square root."""

    def test_diagonal_matrix(self):
        """Test with diagonal matrix."""
        A = np.diag([4, 9, 16])
        S = matrix_sqrt(A, method="schur")
        np.testing.assert_allclose(S @ S, A, rtol=1e-10)
        np.testing.assert_allclose(np.diag(S), [2, 3, 4], rtol=1e-10)

    def test_eigenvalue_method(self):
        """Test eigenvalue-based method."""
        A = np.array([[4, 0], [0, 9]])
        S = matrix_sqrt(A, method="eigenvalue")
        np.testing.assert_allclose(S @ S, A, rtol=1e-10)

    def test_denman_beavers_method(self):
        """Test Denman-Beavers iterative method."""
        A = np.array([[4, 0], [0, 9]])
        S = matrix_sqrt(A, method="denman_beavers")
        # Denman-Beavers is iterative, use looser tolerance
        np.testing.assert_allclose(S @ S, A, rtol=1e-6)

    def test_invalid_method_raises(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            matrix_sqrt(np.eye(2), method="invalid")

    def test_non_square_raises(self):
        """Test non-square matrix raises error."""
        with pytest.raises(ValueError):
            matrix_sqrt(np.array([[1, 2, 3], [4, 5, 6]]))


class TestRankRevealingQR:
    """Tests for rank-revealing QR decomposition."""

    def test_full_rank(self):
        """Test with full rank matrix."""
        A = np.random.randn(4, 3)
        Q, R, P, rank = rank_revealing_qr(A)
        assert rank == 3
        # Check decomposition: A[:, P] = Q @ R
        np.testing.assert_allclose(A[:, P], Q @ R, rtol=1e-10)

    def test_rank_deficient(self):
        """Test with rank-deficient matrix."""
        # Rank 2 matrix
        A = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1], [2, 2, 2]])
        Q, R, P, rank = rank_revealing_qr(A)
        assert rank == 2


class TestNullSpace:
    """Tests for null space computation."""

    def test_null_space_basic(self):
        """Test basic null space."""
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3, rank 2
        N = null_space(A)
        # Null space should have dimension 1
        assert N.shape[1] == 1
        # A @ N should be zero
        np.testing.assert_allclose(A @ N, 0, atol=1e-10)

    def test_full_rank_empty_nullspace(self):
        """Test full rank matrix has empty null space."""
        A = np.eye(3)
        N = null_space(A)
        assert N.shape[1] == 0


class TestRangeSpace:
    """Tests for range space computation."""

    def test_range_space_basic(self):
        """Test basic range space."""
        # Rank 1 matrix
        A = np.array([[1, 2], [2, 4], [3, 6]])
        R = range_space(A)
        assert R.shape == (3, 1)
        # Should be orthonormal
        np.testing.assert_allclose(R.T @ R, np.eye(1), rtol=1e-10)


# =============================================================================
# Continuous-Time Dynamics Tests
# =============================================================================


from pytcl.dynamic_models.continuous_time.dynamics import (  # noqa: E402
    continuous_to_discrete,
    diffusion_constant_acceleration,
    diffusion_constant_velocity,
    diffusion_singer,
    discretize_lti,
    drift_constant_acceleration,
    drift_constant_velocity,
    drift_coordinated_turn_2d,
    drift_singer,
    state_jacobian_ca,
    state_jacobian_cv,
    state_jacobian_singer,
)


class TestDriftFunctions:
    """Tests for continuous-time drift functions."""

    def test_drift_constant_velocity_1d(self):
        """Test 1D constant velocity drift."""
        x = np.array([0.0, 5.0])  # pos=0, vel=5
        a = drift_constant_velocity(x, num_dims=1)
        # dx/dt = vel = 5, dv/dt = 0
        np.testing.assert_allclose(a, [5.0, 0.0])

    def test_drift_constant_velocity_3d(self):
        """Test 3D constant velocity drift."""
        x = np.array([0, 1, 0, 2, 0, 3])  # positions=0, velocities=[1,2,3]
        a = drift_constant_velocity(x, num_dims=3)
        expected = np.array([1, 0, 2, 0, 3, 0])
        np.testing.assert_allclose(a, expected)

    def test_drift_constant_acceleration_1d(self):
        """Test 1D constant acceleration drift."""
        x = np.array([0.0, 5.0, 2.0])  # pos=0, vel=5, acc=2
        a = drift_constant_acceleration(x, num_dims=1)
        # dx/dt = vel, dv/dt = acc, da/dt = 0
        np.testing.assert_allclose(a, [5.0, 2.0, 0.0])

    def test_drift_singer(self):
        """Test Singer model drift."""
        x = np.array([0.0, 0.0, 10.0])  # acc=10
        tau = 5.0
        a = drift_singer(x, tau=tau, num_dims=1)
        # da/dt = -acc/tau = -2
        np.testing.assert_allclose(a, [0.0, 10.0, -2.0])

    def test_drift_coordinated_turn_2d(self):
        """Test 2D coordinated turn drift."""
        vx, vy, omega = 10.0, 5.0, 0.1
        x = np.array([0, vx, 0, vy, omega])
        a = drift_coordinated_turn_2d(x)
        # dx/dt = vx, dvx/dt = -omega*vy, dy/dt = vy, dvy/dt = omega*vx
        expected = np.array([vx, -omega * vy, vy, omega * vx, 0])
        np.testing.assert_allclose(a, expected)


class TestDiffusionFunctions:
    """Tests for diffusion matrix functions."""

    def test_diffusion_constant_velocity(self):
        """Test constant velocity diffusion matrix."""
        x = np.zeros(6)
        D = diffusion_constant_velocity(x, sigma_a=1.0, num_dims=3)
        assert D.shape == (6, 3)
        # Noise enters through velocity components
        assert D[1, 0] == 1.0
        assert D[3, 1] == 1.0
        assert D[5, 2] == 1.0

    def test_diffusion_constant_acceleration(self):
        """Test constant acceleration diffusion matrix."""
        x = np.zeros(9)
        D = diffusion_constant_acceleration(x, sigma_j=2.0, num_dims=3)
        assert D.shape == (9, 3)
        # Noise enters through acceleration components
        assert D[2, 0] == 2.0
        assert D[5, 1] == 2.0
        assert D[8, 2] == 2.0

    def test_diffusion_singer(self):
        """Test Singer model diffusion."""
        x = np.zeros(3)
        D = diffusion_singer(x, sigma_m=1.0, tau=10.0, num_dims=1)
        assert D.shape == (3, 1)
        expected_sigma = np.sqrt(2 * 1.0**2 / 10.0)
        assert np.isclose(D[2, 0], expected_sigma)


class TestStateJacobians:
    """Tests for state Jacobian matrices."""

    def test_state_jacobian_cv_1d(self):
        """Test 1D constant velocity Jacobian."""
        A = state_jacobian_cv(None, num_dims=1)
        expected = np.array([[0, 1], [0, 0]])
        np.testing.assert_allclose(A, expected)

    def test_state_jacobian_cv_3d(self):
        """Test 3D constant velocity Jacobian."""
        A = state_jacobian_cv(None, num_dims=3)
        assert A.shape == (6, 6)
        # Block diagonal structure
        for d in range(3):
            assert A[d * 2, d * 2 + 1] == 1.0

    def test_state_jacobian_ca_1d(self):
        """Test 1D constant acceleration Jacobian."""
        A = state_jacobian_ca(None, num_dims=1)
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        np.testing.assert_allclose(A, expected)

    def test_state_jacobian_singer(self):
        """Test Singer model Jacobian."""
        tau = 5.0
        A = state_jacobian_singer(None, tau=tau, num_dims=1)
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 0, -1 / tau]])
        np.testing.assert_allclose(A, expected)


class TestContinuousToDiscrete:
    """Tests for continuous to discrete conversion."""

    def test_continuous_to_discrete_cv(self):
        """Test C2D for constant velocity."""
        A = np.array([[0, 1], [0, 0]])
        G = np.array([[0], [1]])
        Q_c = np.array([[1.0]])
        T = 0.1

        F, Q_d = continuous_to_discrete(A, G, Q_c, T)

        # Check F matches expected form
        expected_F = np.array([[1, T], [0, 1]])
        np.testing.assert_allclose(F, expected_F, rtol=1e-10)

        # Q_d should be symmetric positive semi-definite
        np.testing.assert_allclose(Q_d, Q_d.T)
        assert np.all(np.linalg.eigvalsh(Q_d) >= -1e-10)


class TestDiscretizeLTI:
    """Tests for LTI discretization."""

    def test_discretize_lti_no_input(self):
        """Test discretization without input matrix."""
        A = np.array([[0, 1], [0, 0]])
        F, G = discretize_lti(A, T=0.1)
        assert G is None
        expected_F = np.array([[1, 0.1], [0, 1]])
        np.testing.assert_allclose(F, expected_F, rtol=1e-10)

    def test_discretize_lti_with_input(self):
        """Test discretization with input matrix."""
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        T = 0.1

        F, G = discretize_lti(A, B, T=T)
        assert G is not None
        assert G.shape == (2, 1)


# =============================================================================
# Process Noise Model Tests
# =============================================================================


from pytcl.dynamic_models.process_noise.coordinated_turn import (  # noqa: E402
    q_coord_turn_2d,
    q_coord_turn_3d,
    q_coord_turn_polar,
)
from pytcl.dynamic_models.process_noise.singer import (  # noqa: E402
    q_singer,
    q_singer_2d,
    q_singer_3d,
)


class TestSingerProcessNoise:
    """Tests for Singer process noise model."""

    def test_q_singer_1d(self):
        """Test Singer process noise covariance 1D."""
        T = 0.1
        tau = 10.0
        sigma_m = 5.0
        Q = q_singer(T, tau, sigma_m, num_dims=1)
        assert Q.shape == (3, 3)
        # Should be symmetric
        np.testing.assert_allclose(Q, Q.T)
        # Check diagonal elements are non-negative
        assert np.all(np.diag(Q) >= 0)

    def test_q_singer_2d(self):
        """Test Singer process noise covariance 2D."""
        Q = q_singer_2d(T=0.1, tau=10.0, sigma_m=5.0)
        assert Q.shape == (6, 6)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_singer_3d(self):
        """Test Singer process noise covariance 3D."""
        Q = q_singer_3d(T=0.1, tau=10.0, sigma_m=5.0)
        assert Q.shape == (9, 9)
        np.testing.assert_allclose(Q, Q.T)


class TestCoordinatedTurnProcessNoise:
    """Tests for coordinated turn process noise."""

    def test_q_coord_turn_2d_position_velocity(self):
        """Test 2D coordinated turn process noise (pos/vel only)."""
        T = 0.1
        sigma_a = 1.0
        Q = q_coord_turn_2d(T, sigma_a)
        assert Q.shape == (4, 4)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_coord_turn_2d_with_omega(self):
        """Test 2D coordinated turn process noise with turn rate."""
        T = 0.1
        sigma_a = 1.0
        sigma_omega = 0.01
        Q = q_coord_turn_2d(
            T, sigma_a, sigma_omega, state_type="position_velocity_omega"
        )
        assert Q.shape == (5, 5)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_coord_turn_3d(self):
        """Test 3D coordinated turn process noise."""
        Q = q_coord_turn_3d(T=0.1, sigma_a=1.0)
        assert Q.shape == (6, 6)
        np.testing.assert_allclose(Q, Q.T)

    def test_q_coord_turn_polar(self):
        """Test polar form coordinated turn process noise."""
        Q = q_coord_turn_polar(T=0.1, sigma_a=1.0, sigma_omega_dot=0.01)
        assert Q.shape == (5, 5)
        np.testing.assert_allclose(Q, Q.T)


# =============================================================================
# Interpolation Tests
# =============================================================================


from pytcl.mathematical_functions.interpolation.interpolation import (  # noqa: E402
    akima,
    barycentric,
    cubic_spline,
    interp1d,
    interp2d,
    linear_interp,
    pchip,
)


class TestInterpolation:
    """Tests for interpolation functions."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for interpolation tests."""
        x = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        y = np.array([0, 1, 4, 9, 16], dtype=np.float64)  # y = x^2
        return x, y

    def test_linear_interp(self, sample_data):
        """Test linear interpolation."""
        xp, yp = sample_data
        # linear_interp(x, xp, fp) - x is the value to interpolate at
        result = linear_interp(1.5, xp, yp)
        # At x=1.5, linear between y[1]=1 and y[2]=4 gives 2.5
        np.testing.assert_allclose(result, 2.5, rtol=1e-10)

    def test_interp1d_linear(self, sample_data):
        """Test 1D interpolation with linear method."""
        x, y = sample_data
        # interp1d returns a callable
        f = interp1d(x, y, kind="linear")
        result = f(1.5)
        np.testing.assert_allclose(result, 2.5, rtol=1e-10)

    def test_cubic_spline(self, sample_data):
        """Test cubic spline interpolation."""
        x, y = sample_data
        # cubic_spline returns a CubicSpline object
        cs = cubic_spline(x, y)
        result = cs(2.5)
        # Should be close to 2.5^2 = 6.25
        assert 5 < result < 8

    def test_pchip(self, sample_data):
        """Test PCHIP interpolation."""
        x, y = sample_data
        # pchip returns a PchipInterpolator object
        p = pchip(x, y)
        result = p(2.5)
        assert 5 < result < 8

    def test_akima(self, sample_data):
        """Test Akima interpolation."""
        x, y = sample_data
        # akima returns an Akima1DInterpolator object
        a = akima(x, y)
        result = a(2.5)
        assert 5 < result < 8

    def test_barycentric(self, sample_data):
        """Test barycentric interpolation."""
        x, y = sample_data
        # barycentric returns a BarycentricInterpolator object
        b = barycentric(x, y)
        result = b(2.0)
        np.testing.assert_allclose(result, 4.0, rtol=1e-6)


class TestInterp2D:
    """Tests for 2D interpolation."""

    @pytest.fixture
    def grid_data(self):
        """Sample grid data."""
        x = np.array([0, 1, 2], dtype=np.float64)
        y = np.array([0, 1, 2], dtype=np.float64)
        z = np.array([[0, 1, 4], [1, 2, 5], [4, 5, 8]], dtype=np.float64)
        return x, y, z

    def test_interp2d_linear(self, grid_data):
        """Test 2D linear interpolation."""
        x, y, z = grid_data
        # interp2d returns a RegularGridInterpolator, call with [[xi, yi]]
        f = interp2d(x, y, z, kind="linear")
        result = f([[0.5, 0.5]])
        assert isinstance(result, np.ndarray)


# =============================================================================
# Statistics Tests
# =============================================================================


from pytcl.mathematical_functions.statistics.distributions import (  # noqa: E402
    Beta,
    ChiSquared,
    Exponential,
    Gamma,
    Gaussian,
    MultivariateGaussian,
    Poisson,
    StudentT,
    Uniform,
    VonMises,
    Wishart,
)


class TestGaussianDistribution:
    """Tests for Gaussian distribution class."""

    def test_gaussian_pdf_standard(self):
        """Test standard normal PDF."""
        g = Gaussian(mean=0, var=1)
        # At mean, PDF should be 1/sqrt(2*pi)
        pdf_at_mean = g.pdf(0)
        expected = 1 / np.sqrt(2 * np.pi)
        np.testing.assert_allclose(pdf_at_mean, expected)

    def test_gaussian_cdf_symmetry(self):
        """Test CDF symmetry."""
        g = Gaussian(mean=0, var=1)
        cdf_pos = g.cdf(1)
        cdf_neg = g.cdf(-1)
        np.testing.assert_allclose(cdf_pos + cdf_neg, 1.0)

    def test_gaussian_ppf_inverse(self):
        """Test PPF is inverse of CDF."""
        g = Gaussian(mean=0, var=1)
        x = 1.5
        cdf_val = g.cdf(x)
        x_recovered = g.ppf(cdf_val)
        np.testing.assert_allclose(x_recovered, x)

    def test_gaussian_logpdf(self):
        """Test log PDF."""
        g = Gaussian(mean=0, var=1)
        logpdf = g.logpdf(0)
        pdf = g.pdf(0)
        np.testing.assert_allclose(logpdf, np.log(pdf))

    def test_gaussian_sample(self):
        """Test sampling."""
        g = Gaussian(mean=0, var=1)
        samples = g.sample(1000)
        assert samples.shape == (1000,)
        # Check approximate mean and variance
        np.testing.assert_allclose(np.mean(samples), 0, atol=0.1)
        np.testing.assert_allclose(np.var(samples), 1, atol=0.1)

    def test_gaussian_mean_var(self):
        """Test mean and variance accessors."""
        g = Gaussian(mean=5, var=4)
        assert g.mean() == 5
        assert g.var() == 4
        np.testing.assert_allclose(g.std(), 2)


class TestMultivariateGaussian:
    """Tests for multivariate Gaussian."""

    def test_multivariate_gaussian_pdf(self):
        """Test multivariate Gaussian PDF."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        mg = MultivariateGaussian(mean=mean, cov=cov)
        x = np.array([0, 0])
        pdf = mg.pdf(x)
        expected = 1 / (2 * np.pi)  # 2D standard normal at origin
        np.testing.assert_allclose(pdf, expected)

    def test_multivariate_gaussian_sample(self):
        """Test multivariate Gaussian sampling."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        mg = MultivariateGaussian(mean=mean, cov=cov)
        samples = mg.sample(100)
        assert samples.shape == (100, 2)

    def test_multivariate_gaussian_mahalanobis(self):
        """Test Mahalanobis distance."""
        mean = np.array([0, 0])
        cov = np.eye(2)
        mg = MultivariateGaussian(mean=mean, cov=cov)
        # Distance at origin should be 0
        dist = mg.mahalanobis(np.array([0, 0]))
        np.testing.assert_allclose(dist, 0.0)


class TestChiSquaredDistribution:
    """Tests for chi-squared distribution."""

    def test_chi2_pdf_positive(self):
        """Test chi2 PDF is positive for x > 0."""
        chi2 = ChiSquared(df=3)
        pdf = chi2.pdf(1)
        assert pdf > 0

    def test_chi2_cdf_bounds(self):
        """Test CDF bounds."""
        chi2 = ChiSquared(df=3)
        assert chi2.cdf(0) == pytest.approx(0, abs=1e-10)
        assert chi2.cdf(100) == pytest.approx(1, abs=1e-5)

    def test_chi2_mean_var(self):
        """Test mean and variance."""
        chi2 = ChiSquared(df=5)
        assert chi2.mean() == 5
        assert chi2.var() == 10


class TestStudentTDistribution:
    """Tests for Student's t distribution."""

    def test_t_pdf_symmetric(self):
        """Test t PDF symmetry."""
        t = StudentT(df=5)
        pdf_pos = t.pdf(1)
        pdf_neg = t.pdf(-1)
        np.testing.assert_allclose(pdf_pos, pdf_neg)

    def test_t_cdf_median(self):
        """Test CDF at median."""
        t = StudentT(df=10)
        cdf_at_zero = t.cdf(0)
        np.testing.assert_allclose(cdf_at_zero, 0.5)


class TestUniformDistribution:
    """Tests for uniform distribution."""

    def test_uniform_pdf(self):
        """Test uniform PDF."""
        u = Uniform(low=0, high=1)
        pdf = u.pdf(0.5)
        np.testing.assert_allclose(pdf, 1.0)
        # Outside bounds
        assert u.pdf(-0.5) == 0

    def test_uniform_cdf(self):
        """Test uniform CDF."""
        u = Uniform(low=0, high=1)
        np.testing.assert_allclose(u.cdf(0.5), 0.5)


class TestExponentialDistribution:
    """Tests for exponential distribution."""

    def test_exponential_pdf(self):
        """Test exponential PDF at zero."""
        exp = Exponential(rate=2)
        pdf = exp.pdf(0)
        np.testing.assert_allclose(pdf, 2.0)

    def test_exponential_cdf(self):
        """Test exponential CDF."""
        exp = Exponential(rate=2)
        cdf = exp.cdf(0)
        np.testing.assert_allclose(cdf, 0.0)


class TestPoissonDistribution:
    """Tests for Poisson distribution."""

    def test_poisson_pmf(self):
        """Test Poisson PMF at lambda."""
        p = Poisson(rate=5)
        pmf = p.pdf(5)  # PDF for discrete is PMF
        assert pmf > 0

    def test_poisson_cdf(self):
        """Test Poisson CDF."""
        p = Poisson(rate=5)
        cdf = p.cdf(10)
        assert 0 < cdf < 1


class TestGammaDistribution:
    """Tests for Gamma distribution."""

    def test_gamma_pdf(self):
        """Test Gamma PDF."""
        gamma = Gamma(shape=2, scale=2)
        pdf = gamma.pdf(2)
        assert pdf > 0

    def test_gamma_mean_var(self):
        """Test Gamma mean and variance."""
        gamma = Gamma(shape=2, scale=2)
        assert gamma.mean() == 4  # shape * scale
        assert gamma.var() == 8  # shape * scale^2


class TestBetaDistribution:
    """Tests for Beta distribution."""

    def test_beta_pdf(self):
        """Test Beta PDF."""
        beta = Beta(a=2, b=5)
        pdf = beta.pdf(0.3)
        assert pdf > 0

    def test_beta_bounds(self):
        """Test Beta CDF at bounds."""
        beta = Beta(a=2, b=5)
        np.testing.assert_allclose(beta.cdf(0), 0)
        np.testing.assert_allclose(beta.cdf(1), 1)


class TestVonMisesDistribution:
    """Tests for Von Mises distribution."""

    def test_vonmises_pdf(self):
        """Test Von Mises PDF at mean."""
        vm = VonMises(mu=0, kappa=2)
        pdf_at_mean = vm.pdf(0)
        assert pdf_at_mean > 0

    def test_vonmises_symmetry(self):
        """Test Von Mises PDF symmetry."""
        vm = VonMises(mu=0, kappa=2)
        pdf_pos = vm.pdf(0.5)
        pdf_neg = vm.pdf(-0.5)
        np.testing.assert_allclose(pdf_pos, pdf_neg)


class TestWishartDistribution:
    """Tests for Wishart distribution."""

    def test_wishart_pdf(self):
        """Test Wishart PDF."""
        scale = np.eye(2)
        w = Wishart(df=3, scale=scale)
        X = np.eye(2) * 2
        pdf = w.pdf(X)
        assert pdf > 0

    def test_wishart_sample(self):
        """Test Wishart sampling."""
        scale = np.eye(2)
        w = Wishart(df=3, scale=scale)
        # Wishart sample requires a size argument
        sample = w.sample(size=1)
        # Returns shape (1, 2, 2) for size=1
        assert sample.shape == (2, 2)


# =============================================================================
# CFAR Detection Tests
# =============================================================================


from pytcl.mathematical_functions.signal_processing.detection import (  # noqa: E402
    cfar_2d,
    cfar_ca,
    cfar_go,
    cfar_os,
    cfar_so,
    detection_probability,
    threshold_factor,
)


class TestCFARDetection:
    """Tests for CFAR detection algorithms."""

    @pytest.fixture
    def test_signal(self):
        """Create test signal with targets."""
        np.random.seed(42)
        n = 100
        signal = np.abs(np.random.randn(n))  # Noise floor
        # Add targets
        signal[30] = 10  # Strong target
        signal[70] = 8  # Medium target
        return signal

    def test_cfar_ca(self, test_signal):
        """Test CA-CFAR detection."""
        result = cfar_ca(test_signal, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert hasattr(result, "detections")
        assert hasattr(result, "threshold")
        assert result.detections.shape == test_signal.shape
        # Should detect the strong target
        assert result.detections[30]

    def test_cfar_go(self, test_signal):
        """Test GO-CFAR detection."""
        result = cfar_go(test_signal, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert result.detections.shape == test_signal.shape

    def test_cfar_so(self, test_signal):
        """Test SO-CFAR detection."""
        result = cfar_so(test_signal, guard_cells=2, ref_cells=5, pfa=1e-3)
        assert result.detections.shape == test_signal.shape

    def test_cfar_os(self, test_signal):
        """Test OS-CFAR detection."""
        result = cfar_os(test_signal, guard_cells=2, ref_cells=5, pfa=1e-3, k=7)
        assert result.detections.shape == test_signal.shape

    def test_cfar_2d(self):
        """Test 2D CFAR detection."""
        np.random.seed(42)
        image = np.abs(np.random.randn(50, 50))
        image[25, 25] = 10  # Target

        result = cfar_2d(image, guard_cells=(2, 2), ref_cells=(5, 5), pfa=1e-3)
        assert result.detections.shape == image.shape
        assert result.detections[25, 25]

    def test_threshold_factor(self):
        """Test threshold factor computation."""
        alpha = threshold_factor(pfa=1e-3, n_ref=10, method="ca")
        assert alpha > 0

    def test_detection_probability(self):
        """Test detection probability computation."""
        pd = detection_probability(snr=15, pfa=1e-6, n_ref=10, method="ca")
        assert 0 < pd < 1
