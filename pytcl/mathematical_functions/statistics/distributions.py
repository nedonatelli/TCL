"""
Probability distributions.

This module provides probability distribution classes with consistent APIs
for PDF, CDF, sampling, and moment calculations. These wrap scipy.stats
distributions with additional functionality useful for tracking applications.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
from numpy.typing import ArrayLike, NDArray


class Distribution(ABC):
    """
    Abstract base class for probability distributions.

    All distribution classes inherit from this and provide consistent
    methods for probability calculations.
    """

    @abstractmethod
    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function."""
        pass

    @abstractmethod
    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function."""
        pass

    @abstractmethod
    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function."""
        pass

    @abstractmethod
    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF)."""
        pass

    @abstractmethod
    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples."""
        pass

    @abstractmethod
    def mean(self) -> float:
        """Distribution mean."""
        pass

    @abstractmethod
    def var(self) -> float:
        """Distribution variance."""
        pass

    def std(self) -> float:
        """Distribution standard deviation."""
        return np.sqrt(self.var())


class Gaussian(Distribution):
    """
    Univariate Gaussian (Normal) distribution.

    Parameters
    ----------
    mean : float
        Mean of the distribution.
    var : float
        Variance of the distribution.

    Examples
    --------
    >>> g = Gaussian(mean=0, var=1)
    >>> g.pdf(0)
    0.3989422804014327
    >>> g.cdf(0)
    0.5
    """

    def __init__(self, mean: float = 0.0, var: float = 1.0):
        if var <= 0:
            raise ValueError("Variance must be positive")
        self._mean = float(mean)
        self._var = float(var)
        self._std = np.sqrt(var)
        self._dist = stats.norm(loc=mean, scale=self._std)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return self._mean

    def var(self) -> float:
        return self._var


class MultivariateGaussian(Distribution):
    """
    Multivariate Gaussian (Normal) distribution.

    Parameters
    ----------
    mean : array_like
        Mean vector of shape (n,).
    cov : array_like
        Covariance matrix of shape (n, n).

    Examples
    --------
    >>> mg = MultivariateGaussian(mean=[0, 0], cov=[[1, 0], [0, 1]])
    >>> mg.pdf([0, 0])
    0.15915494309189535
    """

    def __init__(self, mean: ArrayLike, cov: ArrayLike):
        self._mean = np.asarray(mean, dtype=np.float64)
        self._cov = np.asarray(cov, dtype=np.float64)

        if self._mean.ndim != 1:
            raise ValueError("Mean must be a 1D array")
        if self._cov.ndim != 2:
            raise ValueError("Covariance must be a 2D array")
        if self._cov.shape[0] != self._cov.shape[1]:
            raise ValueError("Covariance must be square")
        if self._mean.shape[0] != self._cov.shape[0]:
            raise ValueError("Mean and covariance dimensions must match")

        self._dist = stats.multivariate_normal(mean=self._mean, cov=self._cov)
        self._dim = len(self._mean)

    @property
    def dim(self) -> int:
        """Dimension of the distribution."""
        return self._dim

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        raise NotImplementedError("PPF not available for multivariate normal")

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> NDArray[np.floating]:
        return self._mean.copy()

    def var(self) -> NDArray[np.floating]:
        """Return diagonal of covariance (marginal variances)."""
        return np.diag(self._cov)

    def cov(self) -> NDArray[np.floating]:
        """Return full covariance matrix."""
        return self._cov.copy()

    def mahalanobis(self, x: ArrayLike) -> NDArray[np.floating]:
        """
        Compute Mahalanobis distance from the mean.

        Parameters
        ----------
        x : array_like
            Point(s) to compute distance for.

        Returns
        -------
        d : ndarray
            Mahalanobis distance(s).
        """
        x = np.asarray(x, dtype=np.float64)
        diff = x - self._mean
        cov_inv = np.linalg.inv(self._cov)

        if diff.ndim == 1:
            return np.sqrt(diff @ cov_inv @ diff)
        else:
            return np.sqrt(np.sum(diff @ cov_inv * diff, axis=-1))


class Uniform(Distribution):
    """
    Continuous uniform distribution.

    Parameters
    ----------
    low : float
        Lower bound of the distribution.
    high : float
        Upper bound of the distribution.
    """

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if high <= low:
            raise ValueError("high must be greater than low")
        self._low = float(low)
        self._high = float(high)
        self._scale = high - low
        self._dist = stats.uniform(loc=low, scale=self._scale)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return (self._low + self._high) / 2

    def var(self) -> float:
        return self._scale**2 / 12


class Exponential(Distribution):
    """
    Exponential distribution.

    Parameters
    ----------
    rate : float
        Rate parameter (λ). Mean is 1/λ.
    """

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError("Rate must be positive")
        self._rate = float(rate)
        self._dist = stats.expon(scale=1.0 / rate)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return 1.0 / self._rate

    def var(self) -> float:
        return 1.0 / self._rate**2


class Gamma(Distribution):
    """
    Gamma distribution.

    Parameters
    ----------
    shape : float
        Shape parameter (k or α).
    rate : float, optional
        Rate parameter (β = 1/θ). Default is 1.
    scale : float, optional
        Scale parameter (θ = 1/β). Alternative to rate.

    Notes
    -----
    Either rate or scale should be specified, not both.
    """

    def __init__(
        self,
        shape: float,
        rate: Optional[float] = None,
        scale: Optional[float] = None,
    ):
        if shape <= 0:
            raise ValueError("Shape must be positive")

        self._shape = float(shape)

        if rate is not None and scale is not None:
            raise ValueError("Specify either rate or scale, not both")
        if rate is not None:
            if rate <= 0:
                raise ValueError("Rate must be positive")
            self._scale = 1.0 / rate
        elif scale is not None:
            if scale <= 0:
                raise ValueError("Scale must be positive")
            self._scale = float(scale)
        else:
            self._scale = 1.0

        self._dist = stats.gamma(a=self._shape, scale=self._scale)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return self._shape * self._scale

    def var(self) -> float:
        return self._shape * self._scale**2


class ChiSquared(Distribution):
    """
    Chi-squared distribution.

    Parameters
    ----------
    df : int
        Degrees of freedom.
    """

    def __init__(self, df: int):
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self._df = int(df)
        self._dist = stats.chi2(df=self._df)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return float(self._df)

    def var(self) -> float:
        return 2.0 * self._df


class StudentT(Distribution):
    """
    Student's t-distribution.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    loc : float, optional
        Location parameter (default 0).
    scale : float, optional
        Scale parameter (default 1).
    """

    def __init__(self, df: float, loc: float = 0.0, scale: float = 1.0):
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        if scale <= 0:
            raise ValueError("Scale must be positive")

        self._df = float(df)
        self._loc = float(loc)
        self._scale = float(scale)
        self._dist = stats.t(df=self._df, loc=self._loc, scale=self._scale)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        if self._df > 1:
            return self._loc
        return np.nan

    def var(self) -> float:
        if self._df > 2:
            return self._scale**2 * self._df / (self._df - 2)
        elif self._df > 1:
            return np.inf
        return np.nan


class Beta(Distribution):
    """
    Beta distribution.

    Parameters
    ----------
    a : float
        First shape parameter (α > 0).
    b : float
        Second shape parameter (β > 0).
    """

    def __init__(self, a: float, b: float):
        if a <= 0 or b <= 0:
            raise ValueError("Shape parameters must be positive")
        self._a = float(a)
        self._b = float(b)
        self._dist = stats.beta(a=self._a, b=self._b)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return self._a / (self._a + self._b)

    def var(self) -> float:
        ab = self._a + self._b
        return (self._a * self._b) / (ab**2 * (ab + 1))


class Poisson(Distribution):
    """
    Poisson distribution (discrete).

    Parameters
    ----------
    rate : float
        Rate parameter (λ), also the mean.
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError("Rate must be positive")
        self._rate = float(rate)
        self._dist = stats.poisson(mu=self._rate)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability mass function (PMF)."""
        return np.asarray(self._dist.pmf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability mass function."""
        return np.asarray(self._dist.logpmf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return self._rate

    def var(self) -> float:
        return self._rate


class VonMises(Distribution):
    """
    Von Mises distribution (circular normal).

    Useful for angular/directional data in tracking applications.

    Parameters
    ----------
    mu : float
        Mean direction (in radians).
    kappa : float
        Concentration parameter.
    """

    def __init__(self, mu: float = 0.0, kappa: float = 1.0):
        if kappa < 0:
            raise ValueError("Concentration parameter must be non-negative")
        self._mu = float(mu)
        self._kappa = float(kappa)
        self._dist = stats.vonmises(kappa=self._kappa, loc=self._mu)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        return self._mu

    def var(self) -> float:
        # Circular variance: 1 - I_1(kappa)/I_0(kappa)
        from scipy.special import i0, i1

        return 1 - i1(self._kappa) / i0(self._kappa)


class Wishart(Distribution):
    """
    Wishart distribution (matrix-valued).

    The Wishart distribution is used for covariance matrix estimation
    in multivariate statistics.

    Parameters
    ----------
    df : float
        Degrees of freedom.
    scale : array_like
        Scale matrix (positive definite).
    """

    def __init__(self, df: float, scale: ArrayLike):
        self._scale = np.asarray(scale, dtype=np.float64)
        if self._scale.ndim != 2 or self._scale.shape[0] != self._scale.shape[1]:
            raise ValueError("Scale must be a square matrix")

        p = self._scale.shape[0]
        if df < p:
            raise ValueError(f"Degrees of freedom must be >= dimension ({p})")

        self._df = float(df)
        self._dim = p
        self._dist = stats.wishart(df=self._df, scale=self._scale)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        raise NotImplementedError("CDF not available for Wishart distribution")

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        raise NotImplementedError("PPF not available for Wishart distribution")

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> NDArray[np.floating]:
        return self._df * self._scale

    def var(self) -> float:
        raise NotImplementedError("Use mean() for matrix-valued distribution")


__all__ = [
    "Distribution",
    "Gaussian",
    "MultivariateGaussian",
    "Uniform",
    "Exponential",
    "Gamma",
    "ChiSquared",
    "StudentT",
    "Beta",
    "Poisson",
    "VonMises",
    "Wishart",
]
