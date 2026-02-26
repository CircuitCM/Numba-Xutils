from __future__ import annotations

import itertools
import math as mt
import random as rand
from collections.abc import Sequence

import numpy as np

import nbux.op._misc as opi
import nbux.op.vector as opv
import nbux.utils as nbu

fb_ = nbu.fb_
aligned_buffer = nbu.aligned_buffer


@nbu.jtc
def gershgorin_l1_norms(A: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Gershgorin L1 norms for a symmetric matrix (C-order).

    Two-pass:

    1. Extract diag and abs(diag).
    2. Tight nested loop to accumulate column L1 sums.

    :param A: Symmetric matrix (C-order assumed).
    :param t1: Work buffer (used for column sums; must be zeroed by caller).
    :param t2: Work buffer (used for diagonal values).
    :returns: ``(anorm, diag, rad)``.
    """
    n = A.shape[0]
    colsum = t1  # NOTE t1 needs to zeroed, assume user handles it
    diag = t2
    for j in range(n): diag[j] = A[j, j]

    # I don't parallelize at the outerloop as the gershgorin bounds are typically used for initial brackets
    # of some larger n^3 routine which will lead to calling into lapack, so less threading traffic will be better.
    # copy paste for parallel version.
    for i in range(n):
        # colsum for symmetric is same as row sum, and i indexing colsum sb faster.
        for j in range(n): colsum[i] += abs(A[i, j])
    # anorm
    an = opv.vmax(colsum)  # shortcut
    # for j in range(1, n):
    #     s = colsum[j]
    #     if s > an:an = s

    rad = colsum
    for j in range(n): rad[j] -= abs(diag[j])

    return an, diag, rad


@nbu.jtnc  # this cant take a parallel decorator for some reason but that's not a problem I think.
def lars1_constraintsolve(
    A: np.ndarray,
    y: np.ndarray,
    out: np.ndarray,
    At: np.ndarray,
    T1: np.ndarray,
    T2: np.ndarray,
    T3: np.ndarray,
    C: np.ndarray,
    idx_buf: np.ndarray,
    Ib: np.ndarray,  # required memory.
    eps: float = 1e-10,
    l2cond: float = -1.0,
) -> np.ndarray:
    """
    For more efficient memory usage we can assume n<=m always. and n s/b >= 2.
    List-free 1-add LARS / homotopy algorithm (basis pursuit).

    for gradient solution n is # unique samples, m is gradient dimensions.

    :param A: Measurement matrix (shape ``(n, m)``, ``n <= m``).
    :param y: Observed measurements (shape ``(n,)``).
    :param out: Output solution vector (shape ``(m,)``).
    :param At: Work buffer.
    :param T1: Work buffer.
    :param T2: Work buffer.
    :param T3: Work buffer.
    :param C: Work buffer.
    :param idx_buf: Work buffer.
    :param Ib: Work buffer.
    :param eps: Residual tolerance.
    :param l2cond: Conditioning parameter (``-1.`` uses a heuristic default).
    :returns: ``out`` (solution vector).
    """

    # At, T1, T2, T3, C, idx_buf, Ib = lars1_memspec()

    # make sure type casting is happening as well.
    ctp = nbu.type_ref(T1)  # calc type
    n, m = A.shape

    def maxabs_c():
        lam = ctp(0.0)
        for i in range(m):
            bc = abs(C[i])
            if lam < bc: lam = bc
        return lam

    _mf = ctp(nbu.prim_info(ctp, 1))
    tol = nbu.prim_info(ctp, 2) * 2.0
    if l2cond == -1.0: l2cond = tol * 64.0
    x = out  # size m
    np.dot(A.T, y, out=C)

    lam = maxabs_c()
    Ib[:] = False
    dtl = lam - tol
    atdx = 0
    # initial placement, seems like this can often start off > 1
    for i in range(m):
        if abs(C[i]) >= dtl and atdx <= n:
            idx_buf[atdx] = i
            Ib[i] = True
            atdx += 1
    # if atdx==0:
    #     return x
    if abs(A.strides[0]) >= abs(A.strides[-1]):  # its C ordered
        for j in range(n):
            for i in range(atdx):  # atdx ~ dimension reference m
                At[i, j] = A[j, idx_buf[i]]
    else:
        for i in range(atdx):
            for j in range(n):  # atdx ~ dimension reference m
                At[i, j] = A[j, idx_buf[i]]
    while True:
        # --- direction on active set
        Gt = T1[: atdx * atdx].reshape((atdx, atdx))
        S2 = T2[:atdx]
        for i in range(atdx): S2[i] = mt.copysign(1.0, C[idx_buf[i]])
        opi.mmul_cself(At[:atdx], Gt, sym=False, outer=True)  # At perm mem
        # if l2cond!=0.:
        # conditioner so cholesky shouldn't ever blow up.
        # Also improves results a bit. If it's significantly larger than a
        # roundoff buffer it turns it into a lasso-like solver.
        opv.dadd(Gt, l2cond * opv.dtrace(Gt) / atdx)
        opi.cholesky_fsolve_inplace(Gt, S2)
        # A[:atdx].T is a view and can be non-contiguous. We copy into T1 to
        # avoid dot heap temps in this path.
        Ast = T1[: atdx * n].reshape((n, atdx))
        for i in range(atdx):
            for j in range(n): Ast[j, i] = At[i, j]
        # --- Solution instance relations
        a = np.dot(Ast, S2, out=T3)
        denr = np.dot(A.T, a, out=T1[-m:])  # this could be copying.. but shouldn't be as a is a vector.

        # --- update greedy magnitude, find next candidate.
        # Simplified Homotophy index decision.
        y_star = _mf
        nidx = -1
        for i in range(m):
            if not Ib[i]:
                denom1 = 1 - denr[i]
                denom2 = 1 + denr[i]
                if abs(denom1) > tol:
                    num1 = lam - C[i]
                    y1 = num1 / denom1
                    if y1 > tol and y1 < y_star:
                        y_star = y1
                        nidx = i
                if abs(denom2) > tol:
                    num2 = lam + C[i]
                    y2 = num2 / denom2
                    if y2 > tol and y2 < y_star:
                        y_star = y2
                        nidx = i

        # if nidx == -1: #theoretically it's not possible
        #     break
        # --- update x along S2
        for i in range(atdx): x[idx_buf[i]] += y_star * S2[i]

        # --- refresh residual, correlations, λ
        r = np.dot(A, x, out=T3)
        rn = 0.0
        for j in range(n):
            rv = y[j] - r[j]
            rn += rv * rv
            r[j] = rv
        rn = mt.sqrt(rn)  # ** .5
        np.dot(A.T, r, out=C)
        lam = maxabs_c()

        # if verbose:
        #     print('|I|=', atdx,
        #           'γ=', y_star,
        #           'idx=', nidx,
        #           '||r||_2=', rn,
        #           'λ=', lam)

        if atdx >= n or rn < eps or lam < tol: break

        At[atdx] = A.T[nidx]
        idx_buf[atdx] = nidx
        Ib[nidx] = True
        atdx += 1

    return x


@nbu.rgc
def lars1_memspec(
    sample_size: int,
    sample_dims: int,
    type_flt: type[np.float64] = np.float64,
    alignb: int = 64,
    buffer: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    # alignb is used to init arrays along instruction aligned memory blocks, avx512=64.
    def cd_(x, dv): return (x + dv - 1) // dv

    flt = nbu.prim_info(type_flt, 3)

    t0, t1, t2, t3 = 8 * sample_size, flt * sample_dims, flt * sample_size, sample_size * sample_size
    t4, t5, t6, t7, t8 = flt * t3, max(sample_dims, t3), cd_(t0, alignb), cd_(t1, alignb), cd_(t2, alignb)
    t9, t10, t11 = flt * t5, alignb * t8, cd_(t4, alignb)
    t12 = cd_(t9, alignb)

    if buffer is None: buffer = aligned_buffer(alignb * (t11 + t12 + t6 + t7 + 2 * t8 + cd_(sample_dims, alignb)), 4096)  # page align

    At = fb_(buffer[:t4], type_flt).reshape((sample_size, sample_size))
    buffer = buffer[alignb * t11 :]
    T1 = fb_(buffer[:t9], type_flt).reshape((t5,))
    buffer = buffer[alignb * t12 :]
    T2 = fb_(buffer[:t2], type_flt).reshape((sample_size,))
    buffer = buffer[t10:]
    T3 = fb_(buffer[:t2], type_flt).reshape((sample_size,))
    buffer = buffer[t10:]
    C = fb_(buffer[:t1], type_flt).reshape((sample_dims,))
    buffer = buffer[alignb * t7 :]
    idx_buf = fb_(buffer[:t0], np.int64).reshape((sample_size,))
    buffer = buffer[alignb * t6 :]
    Ib = fb_(buffer[:sample_dims], np.bool_).reshape((sample_dims,))

    return At, T1, T2, T3, C, idx_buf, Ib


_I64 = nbu.prim_info(np.int64, 1)


@nbu.jtc
def latin_hypercube_sample(
    n_samples: int, bds: Sequence[tuple[float, float]] | np.ndarray | tuple[tuple[float, float], ...]
) -> np.ndarray:
    # Initialize the sample array.
    lb = len(bds)
    sample = np.empty((n_samples, lb), dtype=np.float64)
    # For each dimension...
    i = 0
    for bd in nbu.unroll(bds):
        # Create N equally spaced intervals [0, 1).
        # for compatibility with tuples [i][_s]
        # s[i]
        for s in range(0, n_samples): sample[s, i] = s * (bd[1] - bd[0]) / n_samples + bd[0]
        durstenfeld_p_shuffle(sample[:, i])
        i += 1

    return sample


def edge_sample(bounds: list[tuple[float, float]] | tuple[tuple[float, float], ...], num: int) -> np.ndarray:
    """
    Sample points along the edges of a hyperrectangle.

    :param bounds: Bounds per dimension as ``(lower, upper)`` pairs.
    :param num: Number of samples per edge.

    :returns: An array of shape ``(total_points, dim)`` where
        ``total_points = num * (dim * 2^(dim-1))`` for ``dim > 1``. For 1-D,
        returns ``num`` points.
    """
    dim = len(bounds)
    points: list[list[float]] = []

    # If only one dimension, sample directly along the interval.
    if dim == 1:
        for val in np.linspace(bounds[0][0], bounds[0][1], num=num): points.append([val])
    else:
        # For each dimension, sample along the free edge while fixing the others.
        for free_dim in range(dim):
            # Iterate over all combinations of fixed lower/upper bounds for the other dimensions.
            for fixed_combo in itertools.product([0, 1], repeat=dim - 1):
                pt = [0.0] * dim
                fixed_idx = 0
                for d in range(dim):
                    if d == free_dim: continue
                    # Set fixed value based on combination (0 for lower bound, 1 for upper bound)
                    pt[d] = bounds[d][fixed_combo[fixed_idx]]
                    fixed_idx += 1
                # Sample the free dimension using `num` evenly spaced points.
                for val in np.linspace(bounds[free_dim][0], bounds[free_dim][1], num=num):
                    pt[free_dim] = val
                    points.append(pt.copy())

    return np.array(points)
