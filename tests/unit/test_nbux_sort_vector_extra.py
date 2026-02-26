from __future__ import annotations

import numpy as np

import nbux.op.vector as vec
import nbux.sort as sorti


def test_sort_internal_paths_cover_impl_and_alt_merge_variants() -> None:
    arr = np.array([4.0, -1.0, 7.0, 2.0, 2.0], dtype=np.float64)
    sorti.impl_insert_sort.py_func(arr, lambda a, b: a < b)
    np.testing.assert_allclose(arr, np.array([-1.0, 2.0, 2.0, 4.0, 7.0], dtype=np.float64))

    vals = np.array([1.1, -2.0, 3.3, -2.0], dtype=np.float64)
    idx = np.arange(vals.size, dtype=np.int64)
    sorti.impl_arg_insert_sort.py_func(vals, idx, lambda a, b: a < b)
    np.testing.assert_allclose(vals[idx], np.sort(vals))

    # Force recursive branch with tiny insertion cutoff.
    msort_td = sorti.make_merge_sort(argsort=False, top_down=True, ins_sep=2)
    a_td = np.array([8.0, 3.0, -1.0, 9.0, 1.5, 2.5], dtype=np.float64)
    ws_td = np.empty(max(1, a_td.size // 2), dtype=np.float64)
    msort_td.py_func(a_td, None, ws_td)
    np.testing.assert_allclose(a_td, np.sort(a_td))
    # right side exhausted first -> left leftovers copy path
    a_td2 = np.array([6.0, 7.0, 1.0, 2.0], dtype=np.float64)
    ws_td2 = np.empty(max(1, a_td2.size // 2), dtype=np.float64)
    msort_td.py_func(a_td2, None, ws_td2)
    np.testing.assert_allclose(a_td2, np.sort(a_td2))

    # Cover alternate non-recursive implementation branch.
    msort_bu = sorti.make_merge_sort(argsort=False, top_down=False, ins_sep=2)
    a_bu = np.array([6.0, -3.0, 0.0, 10.0, -1.5, 2.0, 1.0, 8.0], dtype=np.float64)
    ws_bu = np.empty(max(1, a_bu.size // 2), dtype=np.float64)
    msort_bu.py_func(a_bu, None, ws_bu)
    np.testing.assert_allclose(a_bu, np.sort(a_bu))
    a_bu2 = np.array([9.0, 10.0, 0.0, 1.0, -2.0, -1.0, 4.0, 5.0], dtype=np.float64)
    ws_bu2 = np.empty(max(1, a_bu2.size // 2), dtype=np.float64)
    msort_bu.py_func(a_bu2, None, ws_bu2)
    np.testing.assert_allclose(a_bu2, np.sort(a_bu2))


def test_sort_search_paths_cover_large_and_unsafe_modes() -> None:
    large = np.arange(256, dtype=np.float64)
    assert sorti.binary_argsearch.py_func(large, 7.25) == int(np.searchsorted(large, 7.25))

    small = np.array([-4.0, -1.0, 0.0, 3.0], dtype=np.float64)
    assert sorti.binary_argsearch.py_func(small, 0.5) == 3

    base = np.array([-3.0, -1.0, 2.0, 5.0], dtype=np.float64)
    assert sorti._sqleq_arg(base, 2.0) == 2
    assert sorti._sqleq_arg(base, 2.0, unsafe=1) == 2


def test_vector_extra_kernels_cover_remaining_public_ops() -> None:
    x0 = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    y = np.array([0.5, 4.0, -1.0], dtype=np.float64)
    z = np.array([-3.0, 0.25, 2.0], dtype=np.float64)

    x = x0.copy()
    np.testing.assert_allclose(vec.nx.py_func(x), -x0)

    out = np.empty_like(x0)
    np.testing.assert_allclose(vec.cxpy.py_func(out, y, 2.0), 2.0 * y)
    np.testing.assert_allclose(vec.cxay.py_func(out, y, 1.5), 1.5 + y)
    np.testing.assert_allclose(vec.cxapy.py_func(out, y, -1.0, 0.5), -1.0 + 0.5 * y)

    x = x0.copy()
    np.testing.assert_allclose(vec.axay.py_func(x, y, -0.5), x0 - 0.5 + y)
    x = x0.copy()
    np.testing.assert_allclose(vec.axapy.py_func(x, y, 1.0, 2.0), x0 + 1.0 + 2.0 * y)
    x = x0.copy()
    np.testing.assert_allclose(vec.pxaxpy.py_func(x, y, 0.25, -2.0), 0.25 * x0 - 2.0 * y)
    x = x0.copy()
    np.testing.assert_allclose(vec.pxaxy.py_func(x, y, 0.5), 0.5 * x0 + y)

    np.testing.assert_allclose(vec.cxpyapz.py_func(out, y, z, 2.0, -1.0), 2.0 * y - z)
    np.testing.assert_allclose(vec.cxpyaz.py_func(out, y, z, -0.25), -0.25 * y + z)
    np.testing.assert_allclose(vec.cxyaz.py_func(out, y, z), y + z)

    x1, y1 = np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 2.0])
    rx, ry = vec.axypz.py_func(x1.copy(), y1.copy(), z, 2.0, -3.0)
    np.testing.assert_allclose(rx, x1 + 2.0 * z)
    np.testing.assert_allclose(ry, y1 - 3.0 * z)

    vv = np.array([4.0, -2.0, 8.0, 1.0], dtype=np.float64)
    assert float(vec.vmax.py_func(vv)) == 8.0
    assert float(vec.vmin.py_func(vv)) == -2.0
    mn, mx = vec.vminmax.py_func(vv)
    assert float(mn) == -2.0 and float(mx) == 8.0
    mn, mx, i, j = vec.__argminmax_2k.py_func(vv)
    assert float(mn) == -2.0 and float(mx) == 8.0
    assert int(i) >= 0 and int(j) >= 0
    mn2, mx2, i2, j2 = vec.__argminmax_2k.py_func(np.array([-2.0, 4.0, 1.0, 3.0], dtype=np.float64))
    assert float(mn2) == -2.0 and float(mx2) == 4.0
    assert int(i2) == 0 and int(j2) == 1
    mn3, mx3, _, j3 = vec.__argminmax_2k.py_func(np.array([1.0, 0.0, 3.0], dtype=np.float64))
    assert float(mn3) == 0.0 and float(mx3) == 3.0
    assert int(j3) == 2

    m = np.array([[2.0, 1.0], [3.0, -4.0]], dtype=np.float64)
    vec.dvadd.py_func(m, np.array([1.5, -2.5], dtype=np.float64))
    np.testing.assert_allclose(np.diag(m), np.array([3.5, -6.5], dtype=np.float64))
    vec.dmult.py_func(m, 2.0)
    np.testing.assert_allclose(np.diag(m), np.array([7.0, -13.0], dtype=np.float64))

    t1, t2 = vec.tridot.py_func(x0, y, z)
    assert abs(float(t1) - float(np.dot(x0, y))) < 1e-12
    assert abs(float(t2) - float(np.dot(x0, z))) < 1e-12
