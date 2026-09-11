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

    Examples
    --------
    >>> issubclass(Gaussian, Distribution)
    True
    """

    @abstractmethod
    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function.

        Examples
        --------
        >>> round(float(Gaussian(0.0, 1.0).pdf(0.0)), 6)
        0.398942
        """
        pass

    @abstractmethod
    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(Gaussian(0.0, 1.0).logpdf(0.0)), 6)
        -0.918939
        """
        pass

    @abstractmethod
    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> float(Gaussian(0.0, 1.0).cdf(0.0))
        0.5
        """
        pass

    @abstractmethod
    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> float(Gaussian(0.0, 1.0).ppf(0.5))
        0.0
        """
        pass

    @abstractmethod
    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> Gaussian(0.0, 1.0).sample(4).shape
        (4,)
        """
        pass

    @abstractmethod
    def mean(self) -> Union[float, NDArray[np.floating]]:
        """Distribution mean.

        Examples
        --------
        >>> float(Gaussian(1.5, 1.0).mean())
        1.5
        """
        pass

    @abstractmethod
    def var(self) -> Union[float, NDArray[np.floating]]:
        """Distribution variance.

        Examples
        --------
        >>> float(Gaussian(0.0, 4.0).var())
        4.0
        """
        pass

    def std(self) -> Union[float, NDArray[np.floating]]:
        """Distribution standard deviation.

        Examples
        --------
        >>> float(Gaussian(0.0, 4.0).std())
        2.0
        """
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
    >>> round(float(g.pdf(0)), 6)
    0.398942
    >>> round(float(g.cdf(0)), 6)
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
        """Probability density function.

        Examples
        --------
        >>> round(float(Gaussian(0.0, 1.0).pdf(0.0)), 6)
        0.398942
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(Gaussian(0.0, 1.0).logpdf(0.0)), 6)
        -0.918939
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> float(Gaussian(0.0, 1.0).cdf(0.0))
        0.5
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> float(Gaussian(0.0, 1.0).ppf(0.5))
        0.0
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> Gaussian(0.0, 1.0).sample(3).shape
        (3,)
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean.

        Examples
        --------
        >>> float(Gaussian(1.5, 2.0).mean())
        1.5
        """
        return self._mean

    def var(self) -> float:
        """Distribution variance.

        Examples
        --------
        >>> float(Gaussian(0.0, 2.5).var())
        2.5
        """
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
    >>> round(float(mg.pdf([0, 0])), 6)
    0.159155
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
        """Dimension of the distribution.

        Examples
        --------
        >>> MultivariateGaussian([0, 0], [[1, 0], [0, 1]]).dim
        2
        """
        return self._dim

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function.

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        >>> round(float(mg.pdf([0.0, 0.0])), 6)
        0.159155
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        >>> round(float(mg.logpdf([0.0, 0.0])), 6)
        -1.837877
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        >>> round(float(mg.cdf([0.0, 0.0])), 4)
        0.25
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (not available; raises).

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        >>> mg.ppf(0.5)
        Traceback (most recent call last):
            ...
        NotImplementedError: PPF not available for multivariate normal
        """
        raise NotImplementedError("PPF not available for multivariate normal")

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        >>> mg.sample(5).shape
        (5, 2)
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> NDArray[np.floating]:
        """Distribution mean vector.

        Examples
        --------
        >>> mg = MultivariateGaussian([1.0, 2.0], [[1, 0], [0, 1]])
        >>> mg.mean().tolist()
        [1.0, 2.0]
        """
        return self._mean.copy()

    def var(self) -> NDArray[np.floating]:
        """Return diagonal of covariance (marginal variances).

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[4.0, 0.0], [0.0, 9.0]])
        >>> mg.var().tolist()
        [4.0, 9.0]
        """
        return np.diag(self._cov)

    def cov(self) -> NDArray[np.floating]:
        """Return full covariance matrix.

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[4.0, 0.0], [0.0, 9.0]])
        >>> mg.cov().tolist()
        [[4.0, 0.0], [0.0, 9.0]]
        """
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

        Examples
        --------
        >>> mg = MultivariateGaussian([0, 0], [[1, 0], [0, 1]])
        >>> float(mg.mahalanobis([3.0, 4.0]))
        5.0
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

    Examples
    --------
    >>> u = Uniform(low=0.0, high=2.0)
    >>> float(u.mean())
    1.0
    """

    def __init__(self, low: float = 0.0, high: float = 1.0):
        if high <= low:
            raise ValueError("high must be greater than low")
        self._low = float(low)
        self._high = float(high)
        self._scale = high - low
        self._dist = stats.uniform(loc=low, scale=self._scale)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function.

        Examples
        --------
        >>> float(Uniform(0.0, 2.0).pdf(1.0))
        0.5
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(Uniform(0.0, 2.0).logpdf(1.0)), 6)
        -0.693147
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> float(Uniform(0.0, 2.0).cdf(0.5))
        0.25
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> float(Uniform(0.0, 2.0).ppf(0.25))
        0.5
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> s = Uniform(0.0, 2.0).sample(8)
        >>> bool(((s >= 0.0) & (s <= 2.0)).all())
        True
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean.

        Examples
        --------
        >>> float(Uniform(0.0, 2.0).mean())
        1.0
        """
        return (self._low + self._high) / 2

    def var(self) -> float:
        """Distribution variance.

        Examples
        --------
        >>> round(float(Uniform(0.0, 1.0).var()), 6)
        0.083333
        """
        return self._scale**2 / 12


class Exponential(Distribution):
    """
    Exponential distribution.

    Parameters
    ----------
    rate : float
        Rate parameter (λ). Mean is 1/λ.

    Examples
    --------
    >>> e = Exponential(rate=2.0)
    >>> float(e.mean())
    0.5
    """

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError("Rate must be positive")
        self._rate = float(rate)
        self._dist = stats.expon(scale=1.0 / rate)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function.

        Examples
        --------
        >>> float(Exponential(1.0).pdf(0.0))
        1.0
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> float(Exponential(1.0).logpdf(1.0))
        -1.0
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> round(float(Exponential(1.0).cdf(1.0)), 6)
        0.632121
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> round(float(Exponential(1.0).ppf(0.5)), 6)
        0.693147
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> s = Exponential(1.0).sample(8)
        >>> bool((s >= 0.0).all())
        True
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean (1/rate).

        Examples
        --------
        >>> float(Exponential(2.0).mean())
        0.5
        """
        return 1.0 / self._rate

    def var(self) -> float:
        """Distribution variance (1/rate**2).

        Examples
        --------
        >>> float(Exponential(2.0).var())
        0.25
        """
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

    Examples
    --------
    >>> g = Gamma(shape=2.0, rate=1.0)
    >>> float(g.mean())
    2.0
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
        """Probability density function.

        Examples
        --------
        >>> round(float(Gamma(shape=2.0, rate=1.0).pdf(1.0)), 6)
        0.367879
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(Gamma(shape=2.0, rate=1.0).logpdf(1.0)), 6)
        -1.0
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> round(float(Gamma(shape=1.0, rate=1.0).cdf(1.0)), 6)
        0.632121
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> round(float(Gamma(shape=1.0, rate=1.0).ppf(0.5)), 6)
        0.693147
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> Gamma(shape=2.0, rate=1.0).sample(6).shape
        (6,)
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean (shape * scale).

        Examples
        --------
        >>> float(Gamma(shape=3.0, scale=2.0).mean())
        6.0
        """
        return self._shape * self._scale

    def var(self) -> float:
        """Distribution variance (shape * scale**2).

        Examples
        --------
        >>> float(Gamma(shape=3.0, scale=2.0).var())
        12.0
        """
        return self._shape * self._scale**2


class ChiSquared(Distribution):
    """
    Chi-squared distribution.

    Parameters
    ----------
    df : int
        Degrees of freedom.

    Examples
    --------
    >>> c = ChiSquared(df=4)
    >>> float(c.mean())
    4.0
    """

    def __init__(self, df: int):
        if df <= 0:
            raise ValueError("Degrees of freedom must be positive")
        self._df = int(df)
        self._dist = stats.chi2(df=self._df)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function.

        Examples
        --------
        >>> float(ChiSquared(2).pdf(0.0))
        0.5
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(ChiSquared(2).logpdf(0.0)), 6)
        -0.693147
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> round(float(ChiSquared(2).cdf(2.0)), 6)
        0.632121
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> round(float(ChiSquared(2).ppf(0.5)), 6)
        1.386294
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> s = ChiSquared(3).sample(5)
        >>> bool((s >= 0.0).all())
        True
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean (df).

        Examples
        --------
        >>> float(ChiSquared(5).mean())
        5.0
        """
        return float(self._df)

    def var(self) -> float:
        """Distribution variance (2 * df).

        Examples
        --------
        >>> float(ChiSquared(5).var())
        10.0
        """
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

    Examples
    --------
    >>> t = StudentT(df=10)
    >>> float(t.mean())
    0.0
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
        """Probability density function.

        Examples
        --------
        >>> round(float(StudentT(df=1.0).pdf(0.0)), 6)
        0.31831
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(StudentT(df=1.0).logpdf(0.0)), 6)
        -1.14473
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> float(StudentT(df=3.0).cdf(0.0))
        0.5
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> float(StudentT(df=3.0).ppf(0.5))
        0.0
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> StudentT(df=5.0).sample(7).shape
        (7,)
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean (loc for df > 1, NaN otherwise).

        Examples
        --------
        >>> float(StudentT(df=5.0, loc=2.0).mean())
        2.0
        """
        if self._df > 1:
            return self._loc
        return np.nan

    def var(self) -> float:
        """Distribution variance (finite for df > 2).

        Examples
        --------
        >>> float(StudentT(df=4.0).var())
        2.0
        """
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

    Examples
    --------
    >>> beta = Beta(a=2.0, b=2.0)
    >>> float(beta.mean())
    0.5
    """

    def __init__(self, a: float, b: float):
        if a <= 0 or b <= 0:
            raise ValueError("Shape parameters must be positive")
        self._a = float(a)
        self._b = float(b)
        self._dist = stats.beta(a=self._a, b=self._b)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function.

        Examples
        --------
        >>> round(float(Beta(2.0, 2.0).pdf(0.5)), 6)
        1.5
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(Beta(2.0, 2.0).logpdf(0.5)), 6)
        0.405465
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> round(float(Beta(1.0, 1.0).cdf(0.3)), 6)
        0.3
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> round(float(Beta(1.0, 1.0).ppf(0.25)), 6)
        0.25
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> s = Beta(2.0, 2.0).sample(8)
        >>> bool(((s >= 0.0) & (s <= 1.0)).all())
        True
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean, a / (a + b).

        Examples
        --------
        >>> float(Beta(2.0, 6.0).mean())
        0.25
        """
        return self._a / (self._a + self._b)

    def var(self) -> float:
        """Distribution variance.

        Examples
        --------
        >>> round(float(Beta(2.0, 2.0).var()), 6)
        0.05
        """
        ab = self._a + self._b
        return (self._a * self._b) / (ab**2 * (ab + 1))


class Poisson(Distribution):
    """
    Poisson distribution (discrete).

    Parameters
    ----------
    rate : float
        Rate parameter (λ), also the mean.

    Examples
    --------
    >>> p = Poisson(rate=3.0)
    >>> float(p.mean())
    3.0
    """

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError("Rate must be positive")
        self._rate = float(rate)
        self._dist = stats.poisson(mu=self._rate)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability mass function (PMF).

        Examples
        --------
        >>> round(float(Poisson(2.0).pdf(0)), 6)
        0.135335
        """
        return np.asarray(self._dist.pmf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability mass function.

        Examples
        --------
        >>> float(Poisson(2.0).logpdf(0))
        -2.0
        """
        return np.asarray(self._dist.logpmf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> round(float(Poisson(1.0).cdf(0)), 6)
        0.367879
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> float(Poisson(2.0).ppf(0.5))
        2.0
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> s = Poisson(2.0).sample(6)
        >>> bool((s >= 0.0).all())
        True
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Distribution mean (rate).

        Examples
        --------
        >>> float(Poisson(3.0).mean())
        3.0
        """
        return self._rate

    def var(self) -> float:
        """Distribution variance (rate).

        Examples
        --------
        >>> float(Poisson(3.0).var())
        3.0
        """
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

    Examples
    --------
    >>> vm = VonMises(mu=0.0, kappa=2.0)
    >>> float(vm.mean())
    0.0
    """

    def __init__(self, mu: float = 0.0, kappa: float = 1.0):
        if kappa < 0:
            raise ValueError("Concentration parameter must be non-negative")
        self._mu = float(mu)
        self._kappa = float(kappa)
        self._dist = stats.vonmises(kappa=self._kappa, loc=self._mu)

    def pdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Probability density function.

        Examples
        --------
        >>> round(float(VonMises(0.0, 1.0).pdf(0.0)), 6)
        0.34171
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> round(float(VonMises(0.0, 1.0).logpdf(0.0)), 6)
        -1.073791
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function.

        Examples
        --------
        >>> float(VonMises(0.0, 1.0).cdf(0.0))
        0.5
        """
        return np.asarray(self._dist.cdf(x), dtype=np.float64)

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (inverse of CDF).

        Examples
        --------
        >>> float(VonMises(0.0, 1.0).ppf(0.5))
        0.0
        """
        return np.asarray(self._dist.ppf(q), dtype=np.float64)

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> VonMises(0.0, 1.0).sample(5).shape
        (5,)
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> float:
        """Mean direction (mu).

        Examples
        --------
        >>> float(VonMises(mu=1.0, kappa=2.0).mean())
        1.0
        """
        return self._mu

    def var(self) -> float:
        """Circular variance, 1 - I_1(kappa) / I_0(kappa).

        Examples
        --------
        >>> round(float(VonMises(0.0, 1.0).var()), 6)
        0.55361
        """
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

    Examples
    --------
    >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
    >>> w.mean().tolist()
    [[3.0, 0.0], [0.0, 3.0]]
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
        """Probability density function.

        Examples
        --------
        >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
        >>> round(float(w.pdf([[1.0, 0.0], [0.0, 1.0]])), 6)
        0.029275
        """
        return np.asarray(self._dist.pdf(x), dtype=np.float64)

    def logpdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Log of probability density function.

        Examples
        --------
        >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
        >>> round(float(w.logpdf([[1.0, 0.0], [0.0, 1.0]])), 6)
        -3.531024
        """
        return np.asarray(self._dist.logpdf(x), dtype=np.float64)

    def cdf(self, x: ArrayLike) -> NDArray[np.floating]:
        """Cumulative distribution function (not available; raises).

        Examples
        --------
        >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
        >>> w.cdf([[1.0, 0.0], [0.0, 1.0]])
        Traceback (most recent call last):
            ...
        NotImplementedError: CDF not available for Wishart distribution
        """
        raise NotImplementedError("CDF not available for Wishart distribution")

    def ppf(self, q: ArrayLike) -> NDArray[np.floating]:
        """Percent point function (not available; raises).

        Examples
        --------
        >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
        >>> w.ppf(0.5)
        Traceback (most recent call last):
            ...
        NotImplementedError: PPF not available for Wishart distribution
        """
        raise NotImplementedError("PPF not available for Wishart distribution")

    def sample(
        self, size: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> NDArray[np.floating]:
        """Generate random samples.

        Examples
        --------
        >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
        >>> w.sample(1).shape
        (2, 2)
        """
        return np.asarray(self._dist.rvs(size=size), dtype=np.float64)

    def mean(self) -> NDArray[np.floating]:
        """Distribution mean matrix (df * scale).

        Examples
        --------
        >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
        >>> w.mean().tolist()
        [[3.0, 0.0], [0.0, 3.0]]
        """
        return self._df * self._scale

    def var(self) -> float:
        """Variance is not defined here (raises; use mean()).

        Examples
        --------
        >>> w = Wishart(df=3, scale=[[1.0, 0.0], [0.0, 1.0]])
        >>> w.var()
        Traceback (most recent call last):
            ...
        NotImplementedError: Use mean() for matrix-valued distribution
        """
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
