from __future__ import annotations

import math as mt

import numba as nb
import numpy as np

import nbux
from tests.numba_layers import iter_function_layers


def _eval_quad_py(x: np.ndarray, cx: float, cy: float) -> float:
    return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 + 0.03 * mt.sin(5.0 * x[0] - 2.0 * x[1])


@nb.njit
def _eval_quad_jit(x: np.ndarray, cx: float, cy: float) -> float:
    return (x[0] - cx) ** 2 + (x[1] - cy) ** 2 + 0.03 * mt.sin(5.0 * x[0] - 2.0 * x[1])


def test_op_public_exports_and_polynomial_helpers() -> None:
    """Cover public op exports and coefficient/evaluation helper routines."""
    for name in nbux.op.__all__:
        assert hasattr(nbux.op, name)

    a2, b2, c2 = nbux.op.quadratic_newton_coef(0.0, 1.0, 2.0, 2.0, 6.0, 12.0)
    np.testing.assert_allclose([a2, b2, c2], [1.0, 3.0, 2.0], atol=1e-12)
    assert abs(nbux.op.horner_eval(3.0, (a2, b2, c2)) - 20.0) < 1e-12

    def cubic(x: float) -> float:
        return 2.0 * x**3 - x**2 + 0.5 * x - 3.0

    xs = (-2.0, -1.0, 1.0, 2.0)
    fs = tuple(cubic(x) for x in xs)
    c3 = nbux.op.cubic_newton_coef(xs[0], xs[1], xs[2], xs[3], fs[0], fs[1], fs[2], fs[3])
    c3_l = nbux.op.cubic_lagrange_coef(xs[0], xs[1], xs[2], xs[3], fs[0], fs[1], fs[2], fs[3])
    np.testing.assert_allclose(c3, c3_l, atol=1e-10)
    assert abs(nbux.op.horner_eval(1.3, c3) - nbux.op.horner_eval(1.3, c3_l)) < 1e-10


def test_op_public_matrix_helpers_cover_symmetry_and_linear_solve_paths() -> None:
    """Exercise matrix helper exports including square fills, multiply-self, and solve routines."""
    sym = np.array([[2.0, 3.0, -1.0], [3.0, 4.0, 5.0], [-1.0, 5.0, 9.0]], dtype=np.float64)
    lower = sym.copy()
    upper = sym.copy()
    for _, layer_fn in iter_function_layers(nbux.op.sqr_lh):
        lower = sym.copy()
        layer_fn(lower)
        np.testing.assert_allclose(lower, lower.T)
    for _, layer_fn in iter_function_layers(nbux.op.sqr_uh):
        upper = sym.copy()
        layer_fn(upper)
        np.testing.assert_allclose(upper, upper.T)

    a = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=np.float64)
    out_outer = np.empty((a.shape[0], a.shape[0]), dtype=np.float64)
    out_inner = np.empty((a.shape[1], a.shape[1]), dtype=np.float64)
    np.testing.assert_allclose(
        nbux.op.mmul_cself.py_func(a, out_outer, 1.0, 0.0, False, True),
        a @ a.T,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        nbux.op.mmul_cself.py_func(a, out_inner, 1.0, 0.0, False, False),
        a.T @ a,
        atol=1e-12,
    )

    spd = np.array([[4.0, 1.5], [1.5, 3.5]], dtype=np.float64)
    rhs = np.array([1.0, -2.0], dtype=np.float64)
    L = np.linalg.cholesky(spd)
    sol = nbux.op.potrs.py_func(L, rhs.copy())
    np.testing.assert_allclose(spd @ sol, rhs, atol=1e-10)

    sol2 = nbux.op.cholesky_fsolve_inplace.py_func(spd, rhs.copy())
    np.testing.assert_allclose(spd @ sol2, rhs, atol=1e-10)


def test_op_public_grid_eval_paths_cover_exec_and_high_level_wrappers() -> None:
    """Cover grid_eval_exec and grid_eval public entry points for py/jit callable operators."""
    bounds_tuple = (-1.0, 1.0, -2.0, 2.0)
    fitness = np.empty((81,), dtype=np.float32)
    for layer, layer_fn in iter_function_layers(nbux.op.grid_eval_exec):
        eval_op = (_eval_quad_py, 0.25, -0.5) if layer == "py" else (_eval_quad_jit, 0.25, -0.5)
        fit_out, mn, mx = layer_fn(bounds_tuple, fitness.copy(), eval_op)
        assert fit_out.shape == fitness.shape
        assert mn <= mx
        assert float(np.min(fit_out)) >= mn - 1e-6
        assert float(np.max(fit_out)) <= mx + 1e-6

    fit, mn, mx = nbux.op.grid_eval((_eval_quad_jit, 0.25, -0.5), ((-1.0, 1.0), (-2.0, 2.0)), 1024, True)
    assert fit.ndim == 2
    assert mn <= mx
    assert float(np.min(fit)) >= mn - 1e-6


def test_public_vector_module_alias_and_kernel_ops() -> None:
    """Exercise vector module aliases and representative kernel operations across active layers."""
    vec = nbux.op.vector
    assert nbux.vector is vec

    x = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    y = np.array([0.5, 4.0, -1.0], dtype=np.float64)
    z = np.array([-3.0, 0.25, 2.0], dtype=np.float64)

    for _, layer_fn in iter_function_layers(vec.dot):
        assert abs(float(layer_fn(x, y)) - float(np.dot(x, y))) < 1e-12
    for _, layer_fn in iter_function_layers(vec.ndot):
        assert abs(float(layer_fn(x, y)) + float(np.dot(x, y))) < 1e-12
    for _, layer_fn in iter_function_layers(vec.doti):
        assert abs(float(layer_fn(x)) - float(np.dot(x, x))) < 1e-12
    for _, layer_fn in iter_function_layers(vec.l2nm):
        assert abs(float(layer_fn(x)) - float(np.linalg.norm(x))) < 1e-12

    for _, layer_fn in iter_function_layers(vec.cxy):
        out = np.empty_like(x)
        np.testing.assert_allclose(layer_fn(out, y), y)
    for _, layer_fn in iter_function_layers(vec.nxy):
        out = np.empty_like(x)
        np.testing.assert_allclose(layer_fn(out, y), -y)
    for _, layer_fn in iter_function_layers(vec.axpy):
        out = x.copy()
        np.testing.assert_allclose(layer_fn(out, y, 2.0), x + 2.0 * y)
    for _, layer_fn in iter_function_layers(vec.axpyapz):
        out = x.copy()
        np.testing.assert_allclose(layer_fn(out, y, z, 2.0, -0.5), x + 2.0 * y - 0.5 * z)

    for _, layer_fn in iter_function_layers(vec.cxypz):
        o1 = np.empty_like(x)
        o2 = np.empty_like(x)
        r1, r2 = layer_fn(o1, o2, z, 2.0, -3.0)
        np.testing.assert_allclose(r1, 2.0 * z)
        np.testing.assert_allclose(r2, -3.0 * z)

    for _, layer_fn in iter_function_layers(vec.argminmax):
        mn, mx, i, j = layer_fn(y)
        assert mn == np.min(y)
        assert mx == np.max(y)
        assert y[i] == mn
        assert y[j] == mx

    mat = np.array([[3.0, 1.0], [2.0, 4.0]], dtype=np.float64)
    for _, layer_fn in iter_function_layers(vec.dtrace):
        assert abs(float(layer_fn(mat.copy())) - 7.0) < 1e-12
    for _, layer_fn in iter_function_layers(vec.dadd):
        m = mat.copy()
        layer_fn(m, 2.0)
        np.testing.assert_allclose(np.diag(m), np.array([5.0, 6.0]))
    for _, layer_fn in iter_function_layers(vec.dvmult):
        m = mat.copy()
        layer_fn(m, np.array([2.0, -1.0], dtype=np.float64))
        np.testing.assert_allclose(np.diag(m), np.array([6.0, -4.0]))
