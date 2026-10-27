"""Generate the Marcum Q reference values in tests/fixtures/marcum_q/oracle.csv.

Usage::

    uv run python scripts/marcum_q_reference.py [--jobs N]

Each row holds ``a, b, m`` (``b`` as the float64 the tests pass) and
``log Q_m(a, b)`` to 25 significant digits.

Two independent mpmath evaluations are used.

Quadrature: composite fixed-order 48-node Gauss-Legendre of the Marcum
integrand ``x (x/a)^(m-1) exp(-(x^2+a^2)/2) I_{m-1}(a x)``, written as
``x (x/a)^(m-1) ive(m-1, a x) exp(-(x-a)^2/2)``, at ``40 + log10(a)``
digits. Panels are ``1/max(|c|, 1)`` wide, where ``c = b - sqrt(a^2+2m)``
is the distance from the centre of the distribution, so every panel spans
at most about one e-fold of the integrand. For ``c >= 0`` the upper tail
``int_b^inf`` is integrated; for ``c < 0`` the lower tail ``int_0^b`` is
integrated and ``log Q = log1p(-lower)``. Each value is recomputed with
panels half as wide and must agree to 1e-30 relative. ``ive`` is the
Hankel expansion where ``nu^2 < 0.05 z`` and ``z > 200`` (its smallest
term is then below ``e^-400``), otherwise ``besseli(nu, z) e^-z``.
``mp.quad``'s default rule is not used: it gave 6.477e-90 against
6.4734786432915806534e-90 at ``(a=10, b=sqrt(102)+20, m=1)``.

Series: ``Q_m(a, b) = sum_k Pois(k; a^2/2) Q(m+k, b^2/2)``, where for
integer ``n`` the regularized upper incomplete gamma ``Q(n, x)`` is the
Poisson CDF ``P(Pois(x) <= n-1)``. Both ``Q`` and ``1 - Q`` are built from
sums of positive terms (``Q(n, x)`` accumulated upward in ``n``, its
complement downward), so the tail that sets ``log Q`` keeps its relative
accuracy. The ``k`` window starts at ``k_peak +- (60 sqrt(k_peak) + 100)``
and doubles until its edge terms are below 1e-50 of the sum. It runs where
the window has at most 2e5 terms.

Where both run, the smaller of ``Q`` and ``1 - Q`` must agree to 1e-20
relative, or the point is dropped and listed in the header. Where only the
quadrature runs, its upper and lower tails must sum to 1 within 1e-30. Where
``(m-1)^2 > 4ab`` (the series corner) the series is the oracle and the
quadrature, when mpmath's ``besseli`` converges, is the cross-check.
"""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mpmath
from mpmath import mp
from mpmath.calculus.quadrature import GaussLegendre

OUT = Path(__file__).resolve().parent.parent / "tests/fixtures/marcum_q/oracle.csv"

A_GRID = [10.0, 1e3, 1e5, 1e7, 1e8]
M_GRID = [1, 2, 100, 10_000, 10_000_000]
C_GRID = [-40.0, -10.0, -3.0, 0.0, 3.0, 10.0, 20.0]
CORNER_A = 1e3
CORNER_M = [3000, 10_000]
CORNER_C = [-3.0, 0.0, 7.0, 20.0]
SERIES_MAX_TERMS = 200_000


def _is_corner(a: float, b: float, m: int) -> bool:
    return (m - 1.0) ** 2 > 4.0 * a * b


def _grid() -> list[tuple[float, float, int]]:
    points = []
    for a in A_GRID:
        for m in M_GRID:
            for c in C_GRID:
                b = math.sqrt(a * a + 2.0 * m) + c
                if b < 0.0:
                    continue
                if a >= 1e3 and _is_corner(a, b, m):
                    continue
                points.append((a, b, m))
    for m in CORNER_M:
        for c in CORNER_C:
            points.append((CORNER_A, math.sqrt(CORNER_A**2 + 2.0 * m) + c, m))
    return points


def _log_ive(nu, z):
    if nu * nu < 0.05 * z and z > 200:
        mu = 4 * nu * nu
        total = term = mpmath.mpf(1)
        tol = mpmath.mpf(10) ** (-mp.dps - 5)
        j = 0
        while abs(term) > tol * abs(total):
            j += 1
            term = -term * (mu - (2 * j - 1) ** 2) / (8 * j * z)
            total += term
        return mpmath.log(total) - mpmath.log(2 * mpmath.pi * z) / 2
    return mpmath.log(mpmath.besseli(nu, z)) - z


def _gl_integral(log_f, lo, hi, width, nodes):
    """Composite Gauss-Legendre over [lo, hi]; returns (total, first, last panel)."""
    n = max(1, int(mpmath.ceil((hi - lo) / width)))
    h = (hi - lo) / n
    panels = []
    for i in range(n):
        mid = lo + (i + mpmath.mpf(0.5)) * h
        panel = mpmath.fsum(w * mpmath.exp(log_f(mid + h / 2 * x)) for x, w in nodes)
        panels.append(panel * h / 2)
    return mpmath.fsum(panels), panels[0], panels[-1]


def _quad_tail(a, b, m, refine, upper=None):
    """Upper tail (c >= 0) or lower tail (c < 0) of Q by Gauss-Legendre.

    ``upper`` forces one tail, for the normalization check.
    """
    mp.dps = 40 + max(0, int(math.log10(a)))
    nodes = GaussLegendre(mp).calc_nodes(5, mp.prec)
    assert len(nodes) == 48
    a_, b_ = mpmath.mpf(a), mpmath.mpf(b)
    nu = mpmath.mpf(m - 1)
    mu = mpmath.sqrt(a_ * a_ + 2 * m)
    c = b_ - mu
    width = 1 / max(abs(c), 1) / refine
    big_k = (mp.dps + 10) * mpmath.log(10)

    def log_f(x):
        return (
            mpmath.log(x)
            + nu * (mpmath.log(x) - mpmath.log(a_))
            + _log_ive(nu, a_ * x)
            - (x - a_) ** 2 / 2
        )

    edge_tol = mpmath.mpf(10) ** (-mp.dps + 5)
    if upper is None:
        upper = c >= 0
    if upper:
        span = (
            max(-c, 0) + mpmath.sqrt(c * c + 2 * big_k) + mpmath.sqrt(c * c + 2 * big_k)
        )
        total, _, edge = _gl_integral(log_f, b_, b_ + span, width, nodes)
        assert edge < edge_tol * total, (a, b, m)
        return total, True
    lo = max(mpmath.mpf(0), mu - mpmath.sqrt(max(c, 0) ** 2 + c * c + 2 * big_k))
    total, edge, _ = _gl_integral(log_f, lo, b_, width, nodes)
    assert lo == 0 or edge < edge_tol * total, (a, b, m)
    return total, False


def _series(a, b, m):
    """(Q, 1 - Q) from the Poisson mixture, or None past SERIES_MAX_TERMS."""
    mp.dps = 40
    lam = mpmath.mpf(a) ** 2 / 2
    x = mpmath.mpf(b) ** 2 / 2
    k_peak = max(0, (mpmath.sqrt((m - 1) ** 2 + 4 * lam * x) - (m + 1)) / 2)
    half = 60 * mpmath.sqrt(k_peak) + 100
    while True:
        k0 = int(max(0, mpmath.floor(k_peak - half)))
        k1 = int(mpmath.ceil(k_peak + half))
        if k1 - k0 > SERIES_MAX_TERMS:
            return None
        result = _series_window(lam, x, m, k0, k1)
        if result is not None:
            return result
        half *= 2


def _series_window(lam, x, m, k0, k1):
    """Sum the k-window [k0, k1]; None if an edge term is not negligible."""
    n0, n1 = m + k0, m + k1
    eps = mpmath.mpf(10) ** (-mp.dps - 10)
    log_x = mpmath.log(x)

    def pmf_x(j):
        return mpmath.exp(j * log_x - x - mpmath.loggamma(j + 1))

    # Q(n0, x) = sum_{j < n0} pmf_x(j): summed downward from n0 - 1.
    q_n0 = mpmath.mpf(0)
    j = n0 - 1
    term = pmf_x(j) if j >= 0 else mpmath.mpf(0)
    while j >= 0:
        q_n0 += term
        if j < x and term < eps * q_n0:
            break
        term = term * j / x
        j -= 1
    # P(n1 + 1, x) = sum_{j > n1} pmf_x(j): summed upward from n1 + 1.
    p_top = mpmath.mpf(0)
    j = n1 + 1
    term = pmf_x(j)
    while True:
        p_top += term
        if j > x and term < eps * p_top:
            break
        j += 1
        term = term * x / j
    pmfs = [pmf_x(n0)]
    for n in range(n0 + 1, n1 + 1):
        pmfs.append(pmfs[-1] * x / n)
    upper_q = [q_n0]
    for p in pmfs[:-1]:
        upper_q.append(upper_q[-1] + p)
    lower_p = [mpmath.mpf(0)] * len(pmfs)
    acc = p_top
    for i in range(len(pmfs) - 1, -1, -1):
        acc += pmfs[i]
        lower_p[i] = acc
    w = mpmath.exp(k0 * mpmath.log(lam) - lam - mpmath.loggamma(k0 + 1)) if lam else 1
    q_terms, p_terms = [], []
    for i in range(len(pmfs)):
        q_terms.append(w * upper_q[i])
        p_terms.append(w * lower_p[i])
        w = w * lam / (k0 + i + 1)
    q_sum, p_sum = mpmath.fsum(q_terms), mpmath.fsum(p_terms)
    tol = mpmath.mpf(10) ** -50
    for terms, total in ((q_terms, q_sum), (p_terms, p_sum)):
        if terms[-1] > tol * total or (k0 > 0 and terms[0] > tol * total):
            return None
    return q_sum, p_sum


def _evaluate(point):
    a, b, m = point
    series = _series(a, b, m)
    try:
        quad, upper = _quad_tail(a, b, m, 1)
        fine, _ = _quad_tail(a, b, m, 2)
        if abs(fine - quad) > mpmath.mpf(10) ** -30 * abs(fine):
            return point, None, "quadrature refinement disagrees"
    except mpmath.libmp.NoConvergence:
        quad, upper = None, b >= math.sqrt(a * a + 2.0 * m)
    mp.dps = 40
    if series is not None:
        series = series[0] if upper else series[1]
    if _is_corner(a, b, m):
        primary, check, missing = series, quad, "quadrature did not converge"
    else:
        primary, check, missing = quad, series, "series window too large"
    if primary is None:
        return point, None, "oracle unavailable"
    log_q = mpmath.log(primary) if upper else mpmath.log1p(-primary)
    if check is None:
        if quad is not None:
            other, _ = _quad_tail(a, b, m, 1, upper=not upper)
            mp.dps = 40
            if abs(quad + other - 1) > mpmath.mpf(10) ** -30:
                return point, None, "quadrature tails do not sum to 1"
        return point, log_q, missing
    rel = abs(check - primary) / abs(primary)
    if rel > mpmath.mpf(10) ** -20:
        return point, None, f"series and quadrature disagree by {mpmath.nstr(rel, 3)}"
    return point, log_q, "both"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=None)
    args = parser.parse_args()
    grid = _grid()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, result in enumerate(pool.map(_evaluate, grid), 1):
            print(f"{i}/{len(grid)} {result[0]} {result[2]}", flush=True)
            results.append(result)
    rows, dropped, single = [], [], {}
    for (a, b, m), log_q, how in results:
        if log_q is None:
            dropped.append(f"#   a={a!r} b={b!r} m={m}: {how}")
            continue
        if how != "both":
            single.setdefault(how, set()).add(f"({a:g}, {m:g})")
        rows.append(
            f"{a!r},{b!r},{m},{mpmath.nstr(log_q, 25, min_fixed=1, max_fixed=0)}"
        )
    header = [
        "# Marcum Q reference values: log Q_m(a, b), 25 significant digits.",
        "# Generated by: uv run python scripts/marcum_q_reference.py",
        f"# mpmath {mpmath.__version__}; methods in the script's docstring.",
        "# Grid: b = sqrt(a^2 + 2m) + c; a in {10, 1e3, 1e5, 1e7, 1e8},",
        "#   m in {1, 2, 100, 1e4, 1e7}, c in {-40, -10, -3, 0, 3, 10, 20};",
        "#   b < 0 skipped; (m-1)^2 > 4ab skipped for a >= 1e3;",
        "# Series corner ((m-1)^2 > 4ab): a = 1e3, m in {3000, 1e4},",
        "#   c in {-3, 0, 7, 20}; also a = 10, m = 1e7 from the grid above;",
        f"# Rows: {len(rows)}. Dropped: {len(dropped) or 'none'}",
        *dropped,
        "# Checked by one method only, as (a, m) over every c:",
        *(f"#   {how}: {', '.join(sorted(am))}" for how, am in single.items()),
        "a,b,m,log_q",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(header + rows) + "\n")
    print(f"wrote {len(rows)} rows to {OUT}; dropped {len(dropped)}")


if __name__ == "__main__":
    main()
