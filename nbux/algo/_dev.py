from __future__ import annotations

import math as mt

import numpy as np

import nbux.op._misc as opi
import nbux.op.vector as opv
import nbux.utils as nbu


def lars1_constraintsolve_dev(
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
    verbose: bool = False,
    mxitrs: int = -1,
    l2cond: float = -1.0,
) -> np.ndarray:
    """
    For more efficient memory usage we can assume m<=n always. and m s/b >= 2.
    List-free 1-add LARS / homotopy algorithm (basis pursuit).

    for gradient solution m is # unique samples, n is gradient dimensions.

    :param A: Measurement matrix (shape ``(m, n)``, ``m <= n``).
    :param y: Observed measurements (shape ``(m,)``).
    :param out: Output solution vector (shape ``(n,)``).
    :param At: Work buffer.
    :param T1: Work buffer.
    :param T2: Work buffer.
    :param T3: Work buffer.
    :param C: Work buffer.
    :param idx_buf: Work buffer.
    :param Ib: Work buffer.
    :param eps: Residual tolerance.
    :param verbose: Verbosity flag.
    :param mxitrs: Maximum iterations (``-1`` uses a heuristic default).
    :param l2cond: Conditioning parameter (``-1.`` uses a heuristic default).
    :returns: ``out`` (solution vector).
    """

    # older notes on memory dependency.
    # x perm dim n, accumulates the returned vector. can be individual.
    # gp 1: r/v buffer dim m
    # c buffer dim n
    # active_mask dim n but dtype = bool (or maybe int if quicker for indexing mask... or loop it)

    # at_i at most (m,n), but n will be shrunk so reshape first
    # G, sqr I reshaped subindexed buffer, at most n x n.

    # gp 2: d_full/s_I dim n reusable, separate from G as s_i is needed for it's use.

    # gp 1 and 2 can overlap..

    # d_I at most dim n, separate from gp 1/2 but maybe can be used in c.. about to see
    # can use G[0]

    # aiv dim n can be gp 1 2

    # dim_max is actually <= m always.
    # seems I cant avoid making c, which is another size n array.
    ctp = nbu.type_ref(T1)  # calc type
    # dim_max is how many dimensions remain significant in allocated memory.
    # The solution can still have various coefficients.
    dim_max = At.shape[0]
    m, n = A.shape
    if mxitrs == -1:
        if dim_max < m:
            mxitrs = (
                m * 0.8
            )  # (.5**.5) #absolutely zero idea why this is v good stop for memory defficient solvers. even if m>n
            # if it goes >n-1 it explodes so...
        else:
            mxitrs = max(m, dim_max)
    tol = nbu.prim_info(ctp, 2) * 2.0
    if l2cond == -1.0:
        l2cond = tol * 64.0
    x = out  # size n
    # x[:]=0. ASSUME USER SETS TO ZEROS
    np.dot(A.T, y, out=C)

    def maxabs_c():
        lam = ctp(0.0)
        for i in range(n):
            bc = abs(C[i])
            if lam < bc:
                lam = bc
        return lam

    lam = maxabs_c()
    # print('lam',lam)
    Ib[:] = False
    dtl = lam - tol
    atdx = 0
    # initial placement, seems like this can often start of > 1
    for j in range(n):
        if abs(C[j]) >= dtl and atdx <= dim_max:
            # print('kk',j,C[j])
            idx_buf[atdx] = j
            Ib[j] = True
            # T2[atdx]=math.copysign(C[j],1.)
            atdx += 1

    if atdx == 0:  # shouldn't be possible.
        return x

    for j in range(m):
        for i in range(atdx):  # atdx ~ dimension reference n
            At[i, j] = A[j, idx_buf[i]]

    ctrs = 0  # atdx-1
    while True:
        # --- direction on active set
        Gt = T1[: atdx * atdx].reshape((atdx, atdx))
        S2 = T2[:atdx]
        for i in range(atdx):
            S2[i] = mt.copysign(1.0, C[idx_buf[i]])
        # T2 occupied.
        # print(idx_buf)
        # print('As',At[:atdx])
        opi.mmul_cself(At[:atdx], Gt, sym=False, outer=True)  # At perm mem
        # T1 occupied.
        # print('b4',Gt)
        # print(epsr*sqr_trace(Gt)/atdx)
        opv.dadd(
            Gt, l2cond * opv.dtrace(Gt) / atdx
        )  # conditioner so cholesky shouldn't ever blow up. Though we might need more than tll**.9...
        # print('G',Gt[:10,:10])
        # print('signs',S2)
        opi.cholesky_fsolve_inplace(Gt, S2)
        # print('I',idx_buf[:atdx])
        # print('s2',S2)
        # S2 contains our solution. T2 still occupied.
        # T1 free.
        # A[:atdx].T is a view and can be non-contiguous.
        # Copying into T1 avoids heap temp allocation in this dot path.
        Ast = T1[: atdx * m].reshape((m, atdx))
        for i in range(atdx):
            for j in range(m):
                Ast[j, i] = At[i, j]
        # T1 occupied.
        v = np.dot(Ast, S2, out=T3)  # v : T3 size m
        # T3 occupied.
        # T1 free.
        denr = np.dot(A.T, v, out=T1[-n:])  # size n im kinda sure.
        # T1 occupied.
        # T3 free.

        # --- update greedy magnitude, find next candidate.
        y_star = ctp(nbu.prim_info(ctp, 1))
        nidx = -1
        for i in range(n):
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
        # C free.
        # T1 free.
        # --- update x along S2
        if nidx == -1:
            y_star = tol
        for i in range(atdx):
            x[idx_buf[i]] += y_star * S2[i]
        # T2 free.

        # if nidx!=-1:
        #     # Update I
        #     idx_buf[atdx]=nidx
        #     atdx+=1
        #     Ib[nidx]=True
        # --- refresh residual, correlations, λ
        r = np.dot(A, x, out=T3)  # size m
        # T3 occupied.
        rn = 0.0
        for j in range(m):
            rv = y[j] - r[j]
            rn += rv * rv
            r[j] = rv
        rn = rn**0.5
        np.dot(A.T, r, out=C)  # size n
        # T3 freed.
        # C occupied.
        lam = maxabs_c()
        ctrs += 1

        if verbose:
            print(ctrs, "|I|=", atdx, "γ=", y_star, "idx=", nidx, "||r||_2=", rn, "λ=", lam)

        if ctrs >= mxitrs or rn < eps or lam < tol:
            break
        elif atdx >= dim_max and nidx != -1:
            # sort of failed experiment, if you need significantly more samples than necessary for expected sparsity
            # this can reduce mem requirements by at most 50%.
            # but the much more obvious solution is to reduce the # of samples to match dimension # expectation.
            mnxii = 0
            mnxv = abs(x[idx_buf[0]])
            for iv in range(1, atdx):
                tv = abs(x[idx_buf[iv]])
                if tv < mnxv:
                    mnxv = tv
                    mnxii = iv
            mnxidx = idx_buf[mnxii]
            x[mnxidx] = ctp(0.0)
            ii = mnxii
            At[ii] = A.T[nidx]
            odx = idx_buf[ii]
            idx_buf[ii] = nidx
            Ib[odx] = False
            Ib[nidx] = True
        elif nidx != -1:
            # if continue now we add the new candidate.
            At[atdx] = A.T[nidx]
            idx_buf[atdx] = nidx
            atdx += 1
            Ib[nidx] = True
            # print(A.T[nidx],nidx)
        else:
            print("NIDX negative")  # shouldnt be able to get here under normal conditions.

        # if atdx>2:
        #     break

    return x
