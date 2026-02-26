from __future__ import annotations

import numpy as np

import nbux
import nbux.sort as nbux_sort
from tests.numba_layers import iter_function_layers


def test_nbux_root_exports_and_aliases() -> None:
    """Validate root-level exports and module aliases in the public API."""
    for name in nbux.__all__:
        assert hasattr(nbux, name)
    assert nbux.vector is nbux.op.vector
    assert nbux.algo is not None
    assert nbux.rng is not None
    assert nbux.utils is not None


def test_public_sort_dispatchers_cover_insert_and_merge_paths() -> None:
    """Exercise insert/argsort/merge public sort call paths across active numba layers."""
    values = np.array([3.5, -1.2, 7.0, 3.5, 0.0, -4.0], dtype=np.float64)
    asc = np.sort(values)
    desc = asc[::-1]

    for _, layer_fn in iter_function_layers(nbux.insert_sort):
        arr = values.copy()
        layer_fn(arr, True)
        np.testing.assert_allclose(arr, asc)
        arr = values.copy()
        layer_fn(arr, False)
        np.testing.assert_allclose(arr, desc)

    for _, layer_fn in iter_function_layers(nbux.arg_insert_sort):
        idx = np.arange(values.size, dtype=np.int64)
        layer_fn(values, idx, True)
        np.testing.assert_allclose(values[idx], asc)
        idx = np.arange(values.size, dtype=np.int64)
        layer_fn(values, idx, False)
        np.testing.assert_allclose(values[idx], desc)

    ws = np.empty(max(1, values.size // 2), dtype=values.dtype)
    for _, layer_fn in iter_function_layers(nbux.merge_sort):
        arr = values.copy()
        layer_fn(arr, None, ws.copy())
        np.testing.assert_allclose(arr, asc)

    ws_idx = np.empty(max(1, values.size // 2), dtype=np.int64)
    for _, layer_fn in iter_function_layers(nbux.arg_merge_sort):
        idx = np.arange(values.size, dtype=np.int64)
        layer_fn(idx, values, ws_idx.copy())
        np.testing.assert_allclose(values[idx], asc)

    # Reach sorted search routine through the underlying sort module.
    assert int(nbux_sort.binary_argsearch(np.array([-2.0, -1.0, 0.0, 4.0]), 1.5)) == 3
