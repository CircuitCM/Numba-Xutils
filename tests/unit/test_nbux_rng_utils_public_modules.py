from __future__ import annotations

import numpy as np

import nbux
from tests.numba_layers import iter_function_layers


def test_public_rng_module_paths_cover_distribution_fill_modes() -> None:
    """Cover public rng entry points and both sync/parallel wrapper branches."""
    nbux.rng.set_seed(12345)

    for _, layer_fn in iter_function_layers(nbux.rng.durstenfeld_p_shuffle):
        arr = np.arange(12, dtype=np.int64)
        layer_fn(arr, 8)
        assert np.array_equal(np.sort(arr), np.arange(12))

    for _, layer_fn in iter_function_layers(nbux.rng.jt_uniform_rng):
        sample = float(layer_fn(-2.5, 3.5))
        assert -2.5 <= sample <= 3.5

    for parallel in (False,):
        buf = np.empty(512, dtype=np.float64)
        nbux.rng.place_uniform(buf, -3.0, 3.0, parallel)
        assert np.all(buf >= -3.0)
        assert np.all(buf <= 3.0)

    for parallel in (False,):
        buf = np.empty(256, dtype=np.float64)
        nbux.rng.place_gauss(buf, mu=0.5, sigma=0.7, parallel=parallel)
        assert np.isfinite(buf).all()
        assert abs(float(np.mean(buf)) - 0.5) < 0.35

    for parallel in (False,):
        buf = np.empty(256, dtype=np.float64)
        nbux.rng.place_rademacher(buf, low=-2.0, high=5.0, parallel=parallel)
        assert set(np.unique(buf)).issubset({-2.0, 5.0})

    protected = nbux.rng.normal_rng_protect(0.0, 1.0, pr=0.05)
    assert abs(protected) >= 0.05


def test_public_utils_module_helpers_cover_runtime_and_index_lowering_paths() -> None:
    """Validate utility helpers exposed via the root public utils module alias."""
    arr = np.zeros(6, dtype=np.int64)
    for _, layer_fn in iter_function_layers(nbux.utils.placerange):
        arr.fill(0)
        layer_fn(arr, 2, 3)
        np.testing.assert_array_equal(arr, np.array([2, 5, 8, 11, 14, 17]))

    for _, layer_fn in iter_function_layers(nbux.utils.swap):
        vals = np.array([1.0, 4.0, -3.0], dtype=np.float64)
        layer_fn(vals, 0, 2)
        np.testing.assert_allclose(vals, np.array([-3.0, 4.0, 1.0]))

    assert nbux.utils.type_ref(np.array([1.0], dtype=np.float32)) is np.float32
    assert nbux.utils.type_ref(7) is int
    assert nbux.utils.prim_info(np.float64, 2) > 0.0
    assert nbux.utils.if_val_cast(float, 7) == 7.0
    np.testing.assert_array_equal(nbux.utils.if_val_cast(float, np.array([1, 2])), np.array([1, 2]))

    assert nbux.utils.op_call(lambda: 4) == 4
    assert nbux.utils.op_call((lambda x, y: x + y, 2, 3)) == 5
    assert nbux.utils.op_call_args(lambda x, y: x * y, (3, 5)) == 15
    assert nbux.utils.op_call_args((lambda x, y, z: x + y + z, 4), (1, 2)) == 7
    assert nbux.utils.op_args((lambda x, y, z: x + y + z, 4), (1, 2)) == 7

    aligned = nbux.utils.aligned_buffer(256, 64)
    assert aligned.dtype == np.uint8
    assert aligned.shape[0] >= 256

    assert nbux.utils.l_1_0((9, 8, 7), 1) == 8
    assert nbux.utils.l_1_1(((1, 2), (3, 4)), 0) == (1, 2)
    assert nbux.utils.l_1_2((((5,),),), 0) == ((5,),)
    assert nbux.utils.l_12_0(((1, 2), (3, 4)), 1, 0) == 3
    assert nbux.utils.l_21_0(((1, 2), (3, 4)), 0, 1) == 2
    assert nbux.utils.l_12_d(((1, 2), (3, 4)), 1, 1, 0) == 4
