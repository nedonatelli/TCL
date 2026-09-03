"""
Simultaneous multivariate polynomial root finding.

Port of the MATLAB TCL ``polyRootsMultiDim.m`` and the helpers it draws
from (``multiDimPolyMat2Terms``, ``rankTComposition``,
``unrankTComposition``, ``nullspace``), implementing the affine
null-space Macaulay-matrix method of [1]_ (Algorithm 3): the root
finding problem becomes a generalized eigenvalue problem on the null
space of a degree-augmented Macaulay matrix.

References
----------
.. [1] P. Dreesen, "Back to the roots: Polynomial system solving using
   linear algebra," Ph.D. dissertation, Katholieke Universiteit Leuven,
   Leuven, Flanders, Belgium, Sep. 2013.
"""

from math import comb
from typing import NamedTuple, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _rank_colex_combination(combo: NDArray[np.int_]) -> int:
    """Colexicographic rank of a combination (``rankColexCombination``,
    ``firstElMostSig=false``, ``startVal=0``)."""
    c = combo[::-1]
    m = len(c)
    return sum(comb(int(c[i]), m - i) for i in range(m))


def _unrank_colex_combination(
    the_rank: int, n: int, m: int
) -> Optional[NDArray[np.int_]]:
    """Colexicographic unranking (``unrankColexCombo``,
    ``firstElMostSig=false``, ``startVal=0``)."""
    if the_rank >= comb(n, m):
        return None
    combo = np.zeros(m, dtype=np.int64)
    cap = n - 1
    cur_floor = the_rank
    for i in range(m):
        if cur_floor == 0:
            for k in range(i, m):
                combo[k] = m - k - 1
            break
        j = 0
        cur_binom = comb(cap - j, m - i)
        while cur_binom > cur_floor:
            j += 1
            cur_binom = comb(cap - j, m - i)
        combo[i] = cap - j
        cap = cap - j - 1
        cur_floor -= cur_binom
    return combo[::-1]


def _rank_t_composition(p: NDArray[np.int_]) -> int:
    """Rank of a composition (``rankTComposition``,
    ``firstElMostSig=true``)."""
    if len(p) == 1:
        return 0
    p = p[::-1]
    m_combo = len(p) - 1
    c = np.zeros(m_combo, dtype=np.int64)
    c[0] = p[0] - 1
    for k in range(1, m_combo):
        c[k] = p[k] + c[k - 1]
    return _rank_colex_combination(c)


def _unrank_t_composition(the_rank: int, t: int, n: int) -> NDArray[np.int_]:
    """Unrank a composition of ``n`` into ``t`` parts >= 1
    (``unrankTComposition``, ``firstElMostSig=true``)."""
    n = n - 1
    t_combo = t - 1
    if t_combo > 0:
        c = _unrank_colex_combination(the_rank, n, t_combo)
        p = np.zeros(t, dtype=np.int64)
        p[0] = c[0] + 1
        for cur in range(1, t_combo):
            p[cur] = c[cur] - c[cur - 1]
        p[t - 1] = n - c[t_combo - 1]
    else:
        p = np.array([n + 1], dtype=np.int64)
    return p[::-1]


def _poly_mat_to_terms(
    coeffs: NDArray[np.floating], n_vars: int
) -> NDArray[np.floating]:
    """Nonzero terms of a coefficient hypermatrix as a (n_vars+1, k)
    matrix of [coefficient; exponents], in MATLAB's column-major find
    order (``multiDimPolyMat2Terms`` with ``ordering=2``)."""
    flat = coeffs.ravel(order="F")
    idx = np.flatnonzero(flat)
    num_dims = coeffs.ndim
    term_mat = np.zeros((n_vars + 1, len(idx)))
    exps = np.array(np.unravel_index(idx, coeffs.shape, order="F"))
    term_mat[0, :] = flat[idx]
    term_mat[1 : num_dims + 1, :] = exps
    return term_mat


def _nullspace(A: NDArray[np.floating]) -> NDArray[np.floating]:
    """Right null space via SVD.

    Deviation from ``nullspace.m``: the original uses ``matrixRank``
    algorithm 0, whose ``eps(norm(A, 1))`` tolerance (~4e-16) sits
    inside LAPACK's roundoff noise band for the near-zero singular
    values of augmented Macaulay blocks, so the computed nullity
    depends on which SVD driver produced the values. This uses the
    original's ``matrixRank`` algorithm 1 (also MATLAB ``rank()``'s
    default), ``max(size(A)) * eps(max(s))``, which sits orders of
    magnitude above the noise and orders below the true rank gap.
    """
    _, s, vh = np.linalg.svd(A)
    if len(s) == 0:
        return np.eye(A.shape[1])
    tol = max(A.shape) * np.spacing(s[0])
    rank_val = int(np.sum(s > tol))
    return vh[rank_val:, :].conj().T


def _motzkin_row(b: NDArray[np.floating], eps_val: float) -> NDArray[np.floating]:
    """Row subroutine of Motzkin null-space computation (Ch. 3.2.1 of
    [1]_)."""
    n = len(b)
    b = b.copy()
    b[np.abs(b) < eps_val] = 0.0

    nonzero = np.flatnonzero(b)
    if len(nonzero) == 0:
        return np.eye(n)
    ip = nonzero[-1]

    b = b / b[ip]
    W = np.zeros((n, n - 1))
    for i in range(ip + 1, n):
        W[i, i - 1] = 1.0
    for i in range(ip - 1, -1, -1):
        W[i, i] = 1.0
        W[ip, i] = -b[i]
    return W


def _motzkin_matrix(A: NDArray[np.floating]) -> NDArray[np.floating]:
    """Motzkin (canonical) null space of a matrix (Ch. 3.2.2 of [1]_)."""
    m = A.shape[0]
    eps_val = max(A.shape) * np.spacing(np.linalg.norm(A))
    H = _motzkin_row(A[0, :], eps_val)
    for i in range(1, m):
        b = A[i, :] @ H
        H = H @ _motzkin_row(b, eps_val)
    return H


def _get_num_els_before_deg(n: int, d_max: int) -> list:
    """Cumulative monomial counts below each total degree."""
    num_before = [0] * (d_max + 2)
    num_before[1] = 1
    for degree in range(1, d_max + 1):
        num_before[degree + 1] = num_before[degree] + comb(degree + n - 1, n - 1)
    return num_before


def _macaulay_matrix_size(d0: int, d: NDArray[np.int_]) -> tuple:
    """Rows and columns of a degree-``d0`` Macaulay matrix (Lemma 5.8 of
    [1]_)."""
    n = len(d)
    q = comb(n + d0, d0)
    p = sum(comb(n + d0 - int(di), d0 - int(di)) for di in d)
    return p, q


def _build_initial_macaulay(
    term_mats: Sequence[NDArray[np.floating]],
    d: NDArray[np.int_],
    d0: int,
    num_before_deg: list,
    shape: Optional[tuple] = None,
) -> NDArray[np.floating]:
    """Minimum-size Macaulay matrix of degree ``d0`` (Section 5.1.2 of
    [1]_); also used to build the shift matrix Sg.

    ``shape`` overrides the Lemma-5.8 allocation: the MATLAB original
    under-allocates when building Sg (its size formula reads the number
    of variables off ``length(d)``) and silently relies on MATLAB's
    implicit array growth, which numpy does not do.
    """
    s = len(term_mats)
    n = term_mats[0].shape[0] - 1
    if shape is None:
        shape = _macaulay_matrix_size(d0, np.asarray(d))
    M = np.zeros(shape)

    cur_row = 0
    for i in range(s):
        term_mat = term_mats[i]
        num_terms = term_mat.shape[1]
        for cur_term in range(num_terms):
            exps = term_mat[1:, cur_term].astype(np.int64)
            deg = int(np.sum(exps))
            offset = num_before_deg[deg]
            idx = _rank_t_composition(exps + 1) + offset
            M[cur_row, idx] = term_mat[0, cur_term]
        cur_row += 1

        for degree in range(1, d0 - int(d[i]) + 1):
            num_monomials = comb(degree + n - 1, n - 1)
            for j in range(num_monomials):
                cur_monomial = _unrank_t_composition(j, n, degree + n) - 1
                for cur_term in range(num_terms):
                    exps = term_mat[1:, cur_term].astype(np.int64)
                    monomial = cur_monomial + exps
                    deg = int(np.sum(monomial))
                    offset = num_before_deg[deg]
                    idx = _rank_t_composition(monomial + 1) + offset
                    M[cur_row, idx] = term_mat[0, cur_term]
                cur_row += 1
    return M


def _new_rows_for_macaulay(
    p: int,
    num_before_deg: list,
    term_mats: Sequence[NDArray[np.floating]],
    d: NDArray[np.int_],
    d_cur: int,
) -> tuple:
    """Rows appended when the Macaulay matrix grows one degree."""
    s = len(term_mats)
    n = term_mats[0].shape[0] - 1
    d0 = d_cur + 1

    num_before_deg = num_before_deg + [
        num_before_deg[-1] + comb(d_cur + 1 + n - 1, n - 1)
    ]

    p_new, q_new = _macaulay_matrix_size(d0, np.asarray(d))
    m_rows = np.zeros((p_new - p, q_new))

    cur_row = 0
    for i in range(s):
        term_mat = term_mats[i]
        num_terms = term_mat.shape[1]
        degree = d0 - int(d[i])
        num_monomials = comb(degree + n - 1, n - 1)
        for j in range(num_monomials):
            cur_monomial = _unrank_t_composition(j, n, degree + n) - 1
            for cur_term in range(num_terms):
                exps = term_mat[1:, cur_term].astype(np.int64)
                monomial = cur_monomial + exps
                deg = int(np.sum(monomial))
                offset = num_before_deg[deg]
                idx = _rank_t_composition(monomial + 1) + offset
                m_rows[cur_row, idx] = term_mat[0, cur_term]
            cur_row += 1
    return m_rows, num_before_deg


def _check_for_dg(Z: NDArray[np.floating], num_before_deg: list, max_deg: int) -> tuple:
    """Detect the affine basis set via the rank gap (Corollary 6.12 of
    [1]_).

    Deviation from ``checkFordG``: the loop stops before
    ``cur_deg == max_deg``. The degree-``max_deg`` block spans every
    row of ``Z``, so its rank trivially equals the nullity, and
    comparing it with the previous block can declare a spurious gap
    whenever a borderline singular value pushes that block's rank up
    to the nullity (this happened on the five-variable
    frequency-ratio localization system, yielding 32 corrupted roots
    where the true gap — found one degree later — has 28). The MATLAB
    original includes the trivial block and is saved only by roundoff
    landing on the other side of the rank tolerance. A genuine gap at
    ``max_deg`` is simply found at the next degree increase.
    """
    cur_rank = 1
    for cur_deg in range(1, max_deg):
        sel = num_before_deg[cur_deg + 1]
        new_rank = int(np.linalg.matrix_rank(Z[:sel, :]))
        if new_rank == cur_rank:
            return True, cur_rank, cur_deg
        cur_rank = new_rank
    return False, None, None


def _construct_sg(n: int, max_deg: int, num_before_deg: list) -> NDArray[np.floating]:
    """Shift-function matrix for g(x) = x1 + 2*x2 + ... + n*xn
    (Proposition 6.3 of [1]_; the choice of g is arbitrary)."""
    term_mat = np.vstack([np.arange(1, n + 1, dtype=np.float64), np.eye(n)])
    # Degree-1 g times every monomial of degree < max_deg lands in the
    # monomials of degree <= max_deg.
    shape = (num_before_deg[max_deg], num_before_deg[max_deg + 1])
    return _build_initial_macaulay(
        [term_mat], np.array([1]), max_deg, num_before_deg, shape
    )


class PolyRootsResult(NamedTuple):
    """Result of :func:`poly_roots_multi_dim`.

    Attributes
    ----------
    roots : ndarray
        (n, num_sol) matrix of the affine roots found (complex in
        general), or an empty (n, 0) array when ``exit_code`` is
        nonzero.
    exit_code : int
        0 on success; 1 if the maximum number of degree increases
        elapsed; 2 if a finite-precision error made the Macaulay
        nullity change after stabilizing or decrease with degree.
    """

    roots: NDArray[np.complexfloating]
    exit_code: int


def poly_roots_multi_dim(
    poly_coeff_mats: Sequence[ArrayLike],
    max_deg_increases: Optional[int] = None,
    use_motzkin_null: bool = False,
) -> PolyRootsResult:
    """
    Roots of a system of simultaneous multivariate polynomials.

    Only the affine roots are found (generally the only ones desired),
    not the roots at infinity. Due to finite-precision effects and the
    combinatorial growth of the Macaulay matrix, the method is best
    suited to systems of at most 3 variables and degree at most 3;
    sparse systems fare much better than dense ones.

    Parameters
    ----------
    poly_coeff_mats : sequence of array_like
        n coefficient hypermatrices, one per polynomial in n variables.
        ``coeffs[a1, a2, ..., an]`` is the coefficient of
        ``x1**a1 * x2**a2 * ... * xn**an`` (note: zero-based exponents,
        the reverse of MATLAB's 1-based indices with the same layout).
    max_deg_increases : int, optional
        Maximum number of degree increases of the Macaulay matrix.
        Too small a value makes the solve fail with exit code 1.
        Default: ``10 * n``.
    use_motzkin_null : bool, optional
        Use the Motzkin null-space algorithm of [1]_ instead of the
        SVD. Generally less numerically stable; provided to allow
        stepping through the reference values in [1]_. Default False.

    Returns
    -------
    result : PolyRootsResult
        The roots as an (n, num_sol) complex matrix and the exit code.

    Examples
    --------
    The two-variable system from Section 2.1 of [1]_, whose four roots
    are all real: (4, -5), (1, 0), (3, -2) and (0, -1).

    >>> import numpy as np
    >>> p = np.zeros((3, 3))
    >>> p[0, 0], p[2, 0], p[1, 1], p[0, 2] = -4.0, -1.0, 2.0, 1.0
    >>> p[1, 0], p[0, 1] = 5.0, -3.0
    >>> q = np.zeros((3, 3))
    >>> q[0, 0], q[2, 0], q[1, 1], q[0, 2] = -1.0, 1.0, 2.0, 1.0
    >>> roots, exit_code = poly_roots_multi_dim([p, q])
    >>> exit_code
    0
    >>> sorted(np.round(roots.real.T, 6).tolist())
    [[0.0, -1.0], [1.0, 0.0], [3.0, -2.0], [4.0, -5.0]]

    Notes
    -----
    Port of ``polyRootsMultiDim.m``, implementing Algorithm 3 of [1]_.
    The shift function g(x) is the arbitrary choice
    ``sum_i i * x_i`` made by the original. Monomials are tracked with
    composition ranking/unranking; the Macaulay matrix's sparsity is
    not exploited, as in the original.
    """
    n = len(poly_coeff_mats)
    if max_deg_increases is None:
        max_deg_increases = 10 * n

    term_mats = []
    d = np.zeros(n, dtype=np.int64)
    for cur_poly in range(n):
        coeffs = np.asarray(poly_coeff_mats[cur_poly], dtype=np.float64)
        term_mat = _poly_mat_to_terms(coeffs, n)
        # Normalize so the largest coefficient has magnitude one, which
        # reduces finite-precision problems.
        max_val = np.max(np.abs(term_mat[0, :]))
        term_mat[0, :] = term_mat[0, :] / max_val
        term_mats.append(term_mat)
        d[cur_poly] = int(np.max(np.sum(term_mat[1:, :], axis=0)))

    d0 = int(np.max(d))
    num_before_deg = _get_num_els_before_deg(n, d0)

    N = _build_initial_macaulay(term_mats, d, d0, num_before_deg)
    q = N.shape[1]

    if use_motzkin_null:
        Z = _motzkin_matrix(N)
    else:
        Z = _nullspace(N)

    nullity = Z.shape[1]
    p = N.shape[0]

    d_g = None
    d_cur = d0
    ma = None
    deg_of_gap = None

    nullity_stabilized = False
    empty = np.zeros((n, 0), dtype=np.complex128)
    for _ in range(max_deg_increases):
        n_rows, num_before_deg = _new_rows_for_macaulay(
            p, num_before_deg, term_mats, d, d_cur
        )
        q_new = n_rows.shape[1]

        # Expand the null space with the block method of Section 6.2.5.
        N1 = n_rows[:, :q]
        N2 = n_rows[:, q:q_new]
        if use_motzkin_null:
            XY = _motzkin_matrix(np.hstack([N1 @ Z, N2]))
        else:
            XY = _nullspace(np.hstack([N1 @ Z, N2]))
        num_z_prev = Z.shape[1]
        X = XY[:num_z_prev, :]
        Y = XY[num_z_prev:, :]
        Z = np.vstack([Z @ X, Y])

        nullity_new = Z.shape[1]

        d_cur += 1
        q = q_new

        if nullity_new == nullity:
            nullity_stabilized = True
            is_at_dg, ma, deg_of_gap = _check_for_dg(Z, num_before_deg, d_cur)
            if is_at_dg:
                d_g = d_cur
                break
        elif nullity_stabilized or nullity_new < nullity:
            # The nullity restabilized or decreased; both indicate
            # finite-precision failure.
            return PolyRootsResult(empty, 2)
        else:
            nullity = nullity_new

    if d_g is None:
        return PolyRootsResult(empty, 1)

    # Use all monomials up to the degree of the gap; column-compress Z
    # to get W11 (Theorem 6.9 of [1]).
    k = num_before_deg[deg_of_gap + 1]
    S1 = np.eye(num_before_deg[deg_of_gap], k)

    _, _, vh = np.linalg.svd(Z[:k, :])
    W = Z @ vh.conj().T
    W11 = W[:k, :ma]

    Sg = _construct_sg(n, deg_of_gap, num_before_deg)

    # The rectangular generalized eigenvalue problem
    # S1*W11*V11*D = Sg*W11*V11 becomes square via Section 6.2.2.
    A = np.linalg.lstsq(S1 @ W11, Sg @ W11, rcond=None)[0]
    _, V11 = np.linalg.eig(A)

    # Corollary 6.11 extracts the solutions.
    ka1 = W11 @ V11
    ka1 = ka1 / ka1[0, :]
    roots = ka1[1 : n + 1, :]

    return PolyRootsResult(roots, 0)


__all__ = [
    "PolyRootsResult",
    "poly_roots_multi_dim",
]
