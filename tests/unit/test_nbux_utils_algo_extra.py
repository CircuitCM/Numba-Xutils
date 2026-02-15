from __future__ import annotations

import numba as nb
import numba.core.errors as nb_error
import numpy as np
import pytest

import nbux
import nbux.algo._dev as algo_dev
import nbux.algo._linesearch as lsi
import nbux.algo._misc as algo_misc
import nbux.op._misc as op_misc
import nbux.utils as nbu


def test_utils_runtime_helpers_cover_remaining_python_paths() -> None:
    called = {"ok": False}

    def _rrg(func):
        called["ok"] = True
        return func

    def _f(x: int) -> int:
        return x + 1

    wrapped = nbu.rg_no_parallelperf_warn(_rrg)(_f)
    assert called["ok"] is True
    assert wrapped(2) == 3

    assert nbu.display_round(1.23456789, m=1, s=3) == 1.235
    assert nbu.prim_info(np.bool_, 0) is False
    assert nbu.prim_info(np.bool_, 1) is True
    assert nbu.prim_info(np.complex128, 2) == np.finfo(np.complex128).eps
    assert nbu.prim_info(np.float64, 99) is None

    class _Dummy:
        @staticmethod
        def py_func(v: int) -> int:
            return v + 10

        @staticmethod
        def __call__(v: int) -> int:
            raise nb_error.TypingError("dispatch failed")

    assert nbu.run_py(nbux.op.vector.dot, np.array([1.0]), np.array([2.0])) == 2.0
    assert nbu.run_numba(_Dummy(), 7) == 17
    assert nbu.op_call("noop", None) == "noop"
    assert nbu.op_call_args(None, (), None) is None
    assert nbu.op_args(None, (), None) is None
    assert nbu.op_args(lambda x: x + 3, 4) == 7
    assert nbu.op_args((lambda a, b: a - b, 7), 2) == 5

    se = nbu.stack_empty(8, (2, 2), np.float64)
    assert se.shape == (2, 2)

    assert nbu.l_1_0(5) == 5
    assert nbu.l_1_1((1, 2, 3), 1) == (1, 2, 3)
    assert nbu.l_1_2(((1, 2),), 0) == ((1, 2),)
    assert nbu.l_12_0(np.array([9, 8, 7]), 1, 2) == 8
    assert nbu.l_12_0((42,), 0, 0) == 42
    assert nbu.l_21_0(np.array([5, 6, 7]), 0, 2) == 7
    assert nbu.l_21_0((99,), 0, 0) == 99
    assert nbu.l_12_d(np.array([1, 2, 3]), 1, 0, 0) == 2
    assert nbu.l_12_d((11,), 0, 0, 0) == 11


@nb.njit
def _jit_force_helpers(x: np.ndarray) -> tuple[int, int, int]:
    # Trigger overload/intrinsic paths in nbu under compilation.
    t0 = nbu.l_1_0(x, 0)
    t1 = nbu.l_12_0(x, 0, 0)
    t2 = nbu.l_12_d(x, 0, 0, 0)
    return int(t0), int(t1), int(t2)


@nb.njit
def _jit_buffer_nelems(a: np.ndarray) -> int:
    n, _ = nbu.buffer_nelems_andp(a)
    return int(n)


def test_utils_jit_compilation_paths_cover_intrinsic_and_overloads() -> None:
    x1 = np.array([3, 4, 5], dtype=np.int64)
    assert _jit_force_helpers(x1) == (3, 3, 3)
    assert _jit_buffer_nelems(np.arange(7, dtype=np.float64)) == 7
    arr_f = np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    assert _jit_buffer_nelems(arr_f) >= arr_f.size


def test_algo_misc_linesearch_extra_branches() -> None:
    # edge_sample dim=1 branch
    edge1 = algo_misc.edge_sample(((2.0, 5.0),), num=4)
    np.testing.assert_allclose(edge1.ravel(), np.linspace(2.0, 5.0, 4))
    lhs = algo_misc.latin_hypercube_sample.py_func(8, ((-1.0, 1.0), (2.0, 4.0)))
    assert lhs.shape == (8, 2)
    assert np.isfinite(lhs).all()

    # Fortran branch in lars solver setup path.
    A = np.asfortranarray(np.eye(3, dtype=np.float64))
    y = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    At, T1, T2, T3, C, idx_buf, Ib = nbux.algo.lars1_memspec(3, 3)
    mmul_orig = algo_misc.opi.mmul_cself
    chol_orig = algo_misc.opi.cholesky_fsolve_inplace
    algo_misc.opi.mmul_cself = mmul_orig.py_func
    algo_misc.opi.cholesky_fsolve_inplace = chol_orig.py_func
    try:
        res = nbux.algo.lars1_constraintsolve.py_func(A, y, out, At, T1, T2, T3, C, idx_buf, Ib, 1e-10, -1.0)
    finally:
        algo_misc.opi.mmul_cself = mmul_orig
        algo_misc.opi.cholesky_fsolve_inplace = chol_orig
    assert np.isfinite(res).all()

    # Cover sign=-1 branches and non-exported helper wrappers.
    def f(x: float) -> float:
        return 1.0 - x + 0.1 * (x - 1.0) * (x - 1.0)

    def f_lin(x: float) -> float:
        return 1.0 - x

    def g(x: float) -> float:
        return -1.0 + 0.2 * (x - 1.0)

    def c(_: float) -> float:
        return 0.2
    lo, hi = lsi.root_bisect(f, 0.0, 2.0, max_iters=30)
    assert lo <= 1.0 <= hi
    lam, _, _, status = lsi.signedroot_secant.py_func(f, 0.0, 2.0, sign=-1, eager=True, max_iters=40)
    assert status == 0 and abs(float(lam) - 1.0) < 1e-8
    with pytest.raises(ZeroDivisionError):
        lsi.signedroot_quadinterp.py_func(f_lin, 0.0, 2.0, sign=-1, eager=True, max_iters=40)
    lam, _, _, status = lsi.signedroot_newton.py_func(f, g, 0.0, 2.0, sign=-1, eager=True, max_iters=40)
    assert status == 0 and abs(float(lam) - 1.0) < 1e-8
    lam, _, _, status = lsi.signseeking_halley.py_func(f, g, c, 0.0, 2.0, sign=-1, eager=True, max_iters=40)
    assert status == 0 and abs(float(lam) - 1.0) < 1e-8
    with pytest.raises(nb_error.TypingError):
        lsi.posroot_nofallb_secant.py_func(f, 0.0, 2.0, max_iters=20)

    # Trigger helper algorithms with fd-op tuple path.
    fd = (lambda x: (x - 1.0, 1.0),)
    brn = lsi._bracketed_newton(fd, 0.0, 2.0, sign=1, max_iters=20)
    assert abs(float(brn) - 1.0) < 1e-5
    brs = lsi._bracketed_secant.py_func((lambda x: x - 1.0,), 0.0, 2.0, sign=-1, max_iters=20)
    assert np.isfinite(brs)

    # op._misc uncovered branches: complex conjugation and rem_mult accumulation.
    ac = np.array([[1.0 + 2.0j, 0.5 - 1.0j], [0.25 + 0.0j, -2.0 + 1.0j]], dtype=np.complex128)
    outc = np.zeros((2, 2), dtype=np.complex128)
    op_misc.mmul_cself.py_func(ac, outc, a_mult=0.5, rem_mult=0.0, outer=True)
    assert np.isfinite(outc.real).all() and np.isfinite(outc.imag).all()
    out2 = np.ones((2, 2), dtype=np.float64)
    ar = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    op_misc.mmul_cself.py_func(ar, out2, a_mult=1.0, rem_mult=0.25, outer=False)
    assert np.isfinite(out2).all()


@pytest.mark.skip(reason="dev solver is excluded from default pytest runs")
def test_algo_dev_smoke_run_excluded_by_default() -> None:
    A = np.asfortranarray(np.eye(3, dtype=np.float64))
    y = np.array([1.0, -0.5, 0.25], dtype=np.float64)
    out_dev = np.zeros(3, dtype=np.float64)
    At, T1, T2, T3, C, idx_buf, Ib = nbux.algo.lars1_memspec(3, 3)

    mmul_orig = algo_dev.opi.mmul_cself
    chol_orig = algo_dev.opi.cholesky_fsolve_inplace
    algo_dev.opi.mmul_cself = mmul_orig.py_func
    algo_dev.opi.cholesky_fsolve_inplace = chol_orig.py_func
    try:
        dev = algo_dev.lars1_constraintsolve_dev(A, y, out_dev, At, T1, T2, T3, C, idx_buf, Ib, mxitrs=3, verbose=False)
    finally:
        algo_dev.opi.mmul_cself = mmul_orig
        algo_dev.opi.cholesky_fsolve_inplace = chol_orig
    assert np.isfinite(dev).all()
