from __future__ import annotations

import math as mt

import numba as nb
import numpy as np

import nbux
from tests.numba_layers import iter_function_layers


def _f_poly_wave(x: float) -> float:
    return (x - 0.15) * (x - 1.35) * (x + 0.4) + 0.04 * mt.sin(9.0 * x)


@nb.njit
def _f_poly_wave_jit(x: float) -> float:
    return (x - 0.15) * (x - 1.35) * (x + 0.4) + 0.04 * mt.sin(9.0 * x)


def _f_mix(x: float) -> float:
    t = 3.0 * (x - 0.9)
    return (x - 0.9) + 0.2 * mt.tanh(t) + 0.05 * mt.sin(7.0 * x)


def _g_mix(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return 1.0 + 0.6 * sech2 + 0.35 * mt.cos(7.0 * x)


def _c_mix(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return -3.6 * mt.tanh(t) * sech2 - 2.45 * mt.sin(7.0 * x)


@nb.njit
def _f_mix_jit(x: float) -> float:
    t = 3.0 * (x - 0.9)
    return (x - 0.9) + 0.2 * mt.tanh(t) + 0.05 * mt.sin(7.0 * x)


@nb.njit
def _g_mix_jit(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return 1.0 + 0.6 * sech2 + 0.35 * mt.cos(7.0 * x)


@nb.njit
def _c_mix_jit(x: float) -> float:
    t = 3.0 * (x - 0.9)
    c = mt.cosh(t)
    sech2 = 1.0 / (c * c)
    return -3.6 * mt.tanh(t) * sech2 - 2.45 * mt.sin(7.0 * x)


def test_algo_exports_and_bisection_brent_public_paths() -> None:
    """Cover exported bisection/Brent interfaces and module surface imports."""
    for name in nbux.algo.__all__:
        assert hasattr(nbux.algo, name)

    def step(x: float) -> float:
        return 0.0 if x >= 0.75 else -1.0

    edge = float(nbux.algo.not0_bisect(step, 0.0, 1.0, max_iters=40, side=1))
    assert 0.0 <= edge <= 0.75

    lo, hi = nbux.algo.root_bisect(lambda x: 0.0 if x >= 1.25 else -1.0, 0.0, 2.0, max_iters=80)
    assert lo <= 1.25 <= hi
    assert hi - lo < 5e-6

    root = float(nbux.algo.brents_method(lambda x: (x - 0.77) * (x + 0.25), 0.0, 1.5, max_iters=60))
    assert abs(root - 0.77) < 1e-8


def test_signedroot_public_algorithms_cover_active_numba_layers() -> None:
    """Exercise all signedroot exported algorithms across active py/jit layers."""
    for layer, layer_fn in iter_function_layers(nbux.algo.signedroot_secant):
        f = _f_poly_wave if layer == "py" else _f_poly_wave_jit
        root, lo, hi, status = layer_fn(f, 0.2, 1.8, br_rate=0.45, max_iters=60, sign=1, eager=True, fallb=True)
        assert status == 0
        assert lo <= root <= hi
        assert abs(_f_poly_wave(float(root))) < 5e-9

    for layer, layer_fn in iter_function_layers(nbux.algo.signedroot_quadinterp):
        f = _f_poly_wave if layer == "py" else _f_poly_wave_jit
        root, lo, hi, status = layer_fn(f, 0.2, 1.8, br_rate=0.45, max_iters=60, sign=1, eager=True)
        assert status == 0
        assert lo <= root <= hi
        assert abs(_f_poly_wave(float(root))) < 5e-9

    for layer, layer_fn in iter_function_layers(nbux.algo.signedroot_newton):
        f = _f_mix if layer == "py" else _f_mix_jit
        g = _g_mix if layer == "py" else _g_mix_jit
        root, lo, hi, status = layer_fn(f, g, 0.0, 1.8, br_rate=0.45, max_iters=60, sign=1, eager=True)
        assert status == 0
        assert lo <= root <= hi
        assert abs(_f_mix(float(root))) < 1e-8

    for layer, layer_fn in iter_function_layers(nbux.algo.signseeking_halley):
        f = _f_mix if layer == "py" else _f_mix_jit
        g = _g_mix if layer == "py" else _g_mix_jit
        c = _c_mix if layer == "py" else _c_mix_jit
        root, lo, hi, status = layer_fn(f, g, c, 0.0, 1.8, br_rate=0.45, max_iters=60, sign=1, eager=True)
        assert status == 0
        assert lo <= root <= hi
        assert abs(_f_mix(float(root))) < 5e-8


def test_algo_misc_public_exports_cover_sampling_norms_and_lars_paths() -> None:
    """Validate exported misc algorithm paths including lars memory/solve behavior."""
    for _, layer_fn in iter_function_layers(nbux.algo.durstenfeld_p_shuffle):
        arr = np.arange(20, dtype=np.int64)
        layer_fn(arr, 7)
        assert np.array_equal(np.sort(arr), np.arange(20))

    sample = nbux.algo.latin_hypercube_sample(16, ((-2.0, -1.0), (0.0, 3.0)))
    assert sample.shape == (16, 2)
    assert np.all(sample[:, 0] >= -2.0)
    assert np.all(sample[:, 0] <= -1.0)
    assert np.all(sample[:, 1] >= 0.0)
    assert np.all(sample[:, 1] <= 3.0)

    edges = nbux.algo.edge_sample(((-1.0, 1.0), (0.0, 2.0), (3.0, 4.0)), num=5)
    assert edges.shape == (60, 3)
    assert np.all(edges[:, 0] >= -1.0)
    assert np.all(edges[:, 2] <= 4.0)

    A = np.array([[4.0, -1.0, 2.0], [-1.0, 5.0, 0.0], [2.0, 0.0, 6.0]], dtype=np.float64)
    t1 = np.zeros(A.shape[0], dtype=np.float64)
    t2 = np.zeros(A.shape[0], dtype=np.float64)
    anorm, diag, rad = nbux.algo.gershgorin_l1_norms.py_func(A, t1, t2)
    assert anorm > 0.0
    np.testing.assert_allclose(diag, np.diag(A))
    assert np.all(rad >= 0.0)

    n, m = 4, 7
    At, T1, T2, T3, C, idx_buf, Ib = nbux.algo.lars1_memspec(n, m)
    assert At.shape == (n, n)
    assert T1.ndim == 1 and T2.shape == (n,) and C.shape == (m,)
    assert idx_buf.dtype == np.int64 and Ib.dtype == np.bool_

    A = np.array(
        [
            [1.0, 0.5, 0.2, 0.0, 1.1, -0.4, 0.3],
            [0.0, 1.2, -0.2, 0.3, 0.2, 0.5, -0.1],
            [1.0, -0.3, 0.4, 0.7, 0.0, -0.2, 1.1],
            [0.6, 0.1, 0.0, -0.5, 1.0, 0.3, 0.4],
        ],
        dtype=np.float64,
    )
    x_true = np.zeros(m, dtype=np.float64)
    x_true[[1, 4]] = np.array([1.25, -0.7], dtype=np.float64)
    y = A @ x_true
    out = np.zeros(m, dtype=np.float64)
    import nbux.algo._misc as _algo_misc

    mmul_orig = _algo_misc.opi.mmul_cself
    chol_orig = _algo_misc.opi.cholesky_fsolve_inplace
    _algo_misc.opi.mmul_cself = mmul_orig.py_func
    _algo_misc.opi.cholesky_fsolve_inplace = chol_orig.py_func
    try:
        solver = nbux.algo.lars1_constraintsolve.py_func
        result = solver(A, y, out, At, T1, T2, T3, C, idx_buf, Ib, 1e-10, -1.0)
    finally:
        _algo_misc.opi.mmul_cself = mmul_orig
        _algo_misc.opi.cholesky_fsolve_inplace = chol_orig

    np.testing.assert_allclose(A @ result, y, atol=1e-6, rtol=1e-6)
