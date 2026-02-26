from __future__ import annotations

import math as mt
from collections.abc import Callable
from typing import Any

import numba as nb
import numpy as np

import nbux.utils as nbu

INS_SEP = 56
FI = int | float
MArray = np.ndarray | None


@nbu.jti
def _lessthan(a, b, vals=None):
    return vals[a] < vals[b] if vals is not None else a < b


@nbu.jti
def _greaterthan(a, b, vals=None):
    return vals[a] > vals[b] if vals is not None else a > b


@nbu.jt
def impl_insert_sort(sr: np.ndarray, idxr: MArray, comp_call: Callable[[FI, FI, MArray], bool]) -> None:
    cr = idxr if idxr is not None else sr
    vals = sr if idxr is not None else None
    # because idxr is hardcoded as None or an array, this becomes compile-time evaluated and removed.
    for i in range(1, cr.shape[0]):
        k = cr[i]
        j = i
        while j > 0 and comp_call(k, cr[j - 1], vals):
            # Make place for moving A[i] downwards
            cr[j] = cr[j - 1]
            j -= 1
        cr[j] = k


@nbu.jt
def insert_sort(sr: np.ndarray, small_first: bool = True) -> None:
    """
    Insertion sort.

    (Tested several slightly different implementations; this had the best
    performance for Numba ``njit`` compilation.)

    :param sr: Array to sort in-place.
    :returns: None.
    """
    if small_first: impl_insert_sort(sr, None, _lessthan)
    else: impl_insert_sort(sr, None, _greaterthan)


@nbu.jt
def arg_insert_sort(sr: np.ndarray, idxr: np.ndarray, small_first: bool = True) -> None:
    """
    Insertion based index sort.

    :param sr: Array for sort comparison.
    :param idxr: Array of sr indexes to sort.
    :returns: None.
    """
    if small_first: impl_insert_sort(sr, idxr, _lessthan)
    else: impl_insert_sort(sr, idxr, _greaterthan)


# SMALL_MERGESORT = 20 #original size, ~50 seems to provide better performance profile over large range

#in the future make this like insert sort... way too ugly.
def _make_merge_sort(argsort: bool = False, top_down: bool = True) -> Callable[..., None]:
    """
    Create a merge sort implementation for Numba.

    Taken from Numba's merge sort implementation, letting the user sort already
    initialized arrays. From local testing, this merge sort appears to be
    consistently quicker than Numba's quick sort.

    The returned ``merge_sort`` callable has the signature
    ``merge_sort(idxr, vals, ws, ...)`` where:

    - ``idxr`` is the array sorted in-place (or indices for argsort),
    - ``vals`` is ``None`` for normal sort (or the values array for argsort),
    - ``ws`` is a workspace array of size ``idxr.size // 2``.

    :param argsort: If True, build an argsort variant.
    :param top_down: If True, build the top-down recursive variant.
    :returns: A Numba-jittable ``merge_sort`` function.
    """
    if argsort:

        @nbu.jt
        def lessthan(a, b, vals): return vals[a] < vals[b]

    else:

        @nbu.jt
        def lessthan(a, b, vals): return a < b

    if top_down:

        @nbu.jt
        def _merge_sort(arr, vals, _ws) -> None:  # assume ws is made by user
            ws = _ws
            if arr.size > INS_SEP:
                # Merge sort
                mid = arr.size // 2

                _merge_sort(arr[:mid], vals, ws)
                _merge_sort(arr[mid:], vals, ws)

                # Copy left half into workspace so we don't overwrite it
                for i in range(mid): ws[i] = arr[i]

                # Merge
                left = ws[:mid]
                right = arr[mid:]
                out = arr

                i = j = k = 0
                ls = left.size
                rs = right.size
                while i < ls and j < rs:
                    if not lessthan(right[j], left[i], vals):
                        out[k] = left[i]
                        i += 1
                    else:
                        out[k] = right[j]
                        j += 1
                    k += 1

                # Leftovers
                while i < left.size:
                    out[k] = left[i]
                    i += 1
                    k += 1

                # unecessary because if we get here, out[k] is literally the same memory address as right[j]
                # while j < right.size:
                #     out[k] = right[j]
                #     j += 1
                #     k += 1
            else:
                # fastest insert sort style I found in numba after testing several slightly different styles.
                for i in range(1, arr.shape[0]):
                    k = arr[i]
                    j = i
                    while j > 0 and lessthan(k, arr[j - 1], vals):
                        # Make place for moving A[i] downwards
                        arr[j] = arr[j - 1]
                        j -= 1
                    arr[j] = k
    else:
        # WARNING TO USERS. In theory this should be faster because no recursion and the parallelized insert sort, while
        # the insert sort pl does improve performance the merge sort bottom_up loops run significantly slower.
        # Idk why, maybe it has to do with rounding to the nearest index, perhaps initializing them outside of the loops
        # or another method would let this method surpass the recursive stack method.
        @nbu.jtp
        def _merge_sort(arr, vals, _ws, *__ext):
            # implement bottom up merge sort that first identifies the smallest # of partitions that equally divides the
            # array such that
            # each array is smaller than or equal to SMALL_MERGESORT and the # of partitions is v power multiple of 2.
            # then it performs these insert sorts for each group.
            # next begins v bottom up merging like how it's done in top down only we use two for loops, the outer one
            # specifying the merge layer and the inner loop placing back into the array to be sorted.
            tsz = arr.size
            nlayers = mt.ceil(mt.log2(tsz / INS_SEP))
            parts = 2**nlayers
            gs = tsz / parts
            #parallelizable, the merging part technically could be as well but less efficient.
            for v in nb.prange(1, parts + 1):
                st = nbu.ri64((v - 1) * gs)
                ed = nbu.ri64(v * gs)
                for i in range(st, ed):
                    k = arr[i]
                    j = i
                    while j > 0 and lessthan(k, arr[j - 1], vals):
                        # Make place for moving A[i] downwards
                        arr[j] = arr[j - 1]
                        j -= 1
                    arr[j] = k

            ws = _ws  # workspace array provided by user

            # Merge layers: in each iteration, merge adjacent pairs of segments of size current_size.
            # won't run quicker for some reason.
            for _ in range(nlayers):
                # Iterate over the array in steps of 2*current_size
                parts //= 2
                gs *= 2
                # should be pretty close to parallelizing this too, but id need to double check ws v bit more first.
                for v in range(1, parts + 1):
                    start = nbu.ri64((v - 1) * gs)
                    end = nbu.ri64(v * gs)
                    # if end>tsz:
                    #     print('shouldnt be possible',v)
                    #     end=max(end,tsz)
                    mid = nbu.ri64((v - 0.5) * gs)
                    # if mid >= tsz: #shouldn't be possible. as well as the other checks
                    #     break  # no pair to merge

                    # Copy the left segment [start:mid] into the workspace.
                    left_size = mid - start
                    for j in range(left_size):  # for complete parallel ws would need to be offset.
                        ws[j] = arr[start + j]

                    # Merge ws (left) and idxr[mid:end] (right) back into idxr[start:end]
                    i = 0  # index for ws (left segment)
                    j = mid  # index for right segment in idxr
                    k = start  # output index in idxr
                    while i < left_size and j < end:
                        if not lessthan(arr[j], ws[i], vals):
                            arr[k] = ws[i]
                            i += 1
                        else:
                            arr[k] = arr[j]
                            j += 1
                        k += 1

                    # Copy any remaining elements from the left segment.
                    while i < left_size:
                        arr[k] = ws[i]
                        i += 1
                        k += 1

    return _merge_sort


_arg_merge_sort = _make_merge_sort(True, True)
_merge_sort = _make_merge_sort(False, True)


@nbu.jt
def impl_merge_sort(sr: np.ndarray, idxr: MArray, ws: np.ndarray, comp_call: Callable[[FI, FI, MArray], bool]) -> None:
    cr = idxr if idxr is not None else sr
    vals = sr if idxr is not None else None
    if cr.size > INS_SEP:
        # Merge sort
        mid = cr.size // 2

        # Keep sr fixed for argsort recursion; recurse on value views for direct sort.
        if idxr is None:
            impl_merge_sort(sr[:mid], None, ws, comp_call)
            impl_merge_sort(sr[mid:], None, ws, comp_call)
        else:
            impl_merge_sort(sr, idxr[:mid], ws, comp_call)
            impl_merge_sort(sr, idxr[mid:], ws, comp_call)

        # Copy left half into workspace so we don't overwrite it
        for i in range(mid): ws[i] = cr[i]

        # Merge
        left = ws[:mid]
        right = cr[mid:]
        out = cr

        i = j = k = 0
        ls = left.size
        rs = right.size
        while i < ls and j < rs:
            if not comp_call(right[j], left[i], vals):
                out[k] = left[i]
                i += 1
            else:
                out[k] = right[j]
                j += 1
            k += 1

        # Leftovers
        while i < left.size:
            out[k] = left[i]
            i += 1
            k += 1

        # unnecessary because if we get here, out[k] is literally the same memory address as right[j]
        # while j < right.size:
        #     out[k] = right[j]
        #     j += 1
        #     k += 1
    else:
        # fastest insert sort style I found in numba after testing several slightly different styles.
        for i in range(1, cr.shape[0]):
            k = cr[i]
            j = i
            while j > 0 and comp_call(k, cr[j - 1], vals):
                # Make place for moving A[i] downwards
                cr[j] = cr[j - 1]
                j -= 1
            cr[j] = k


# WARNING TO USERS. In theory this should be faster because no recursion and the parallelized insert sort, while
# the insert sort pl does improve performance the merge sort bottom_up loops run significantly slower.
# Idk why, maybe it has to do with rounding to the nearest index, perhaps initializing them outside of the loops
# or another method would let this method surpass the recursive stack method.
@nbu.jtp
def impl_bu_merge_sort(sr: np.ndarray, idxr: MArray, ws: np.ndarray, comp_call: Callable[[FI, FI, MArray], bool], *__ext) -> None:
    cr = idxr if idxr is not None else sr
    vals = sr if idxr is not None else None
    # implement bottom up merge sort that first identifies the smallest # of partitions that equally divides the
    # array such that
    # each array is smaller than or equal to SMALL_MERGESORT and the # of partitions is v power multiple of 2.
    # then it performs these insert sorts for each group.
    # next begins v bottom up merging like how it's done in top down only we use two for loops, the outer one
    # specifying the merge layer and the inner loop placing back into the array to be sorted.
    tsz = cr.size
    nlayers = mt.ceil(mt.log2(tsz / INS_SEP))
    parts = 2**nlayers
    gs = tsz / parts
    #parallelizable, the merging part technically could be as well but less efficient.
    for v in nb.prange(1, parts + 1):
        st = nbu.ri64((v - 1) * gs)
        ed = nbu.ri64(v * gs)
        for i in range(st, ed):
            k = cr[i]
            j = i
            while j > 0 and comp_call(k, cr[j - 1], vals):
                # Make place for moving A[i] downwards
                cr[j] = cr[j - 1]
                j -= 1
            cr[j] = k

    # Merge layers: in each iteration, merge adjacent pairs of segments of size current_size.
    # won't run quicker for some reason.
    for _ in range(nlayers):
        # Iterate over the array in steps of 2*current_size
        parts //= 2
        gs *= 2
        # should be pretty close to parallelizing this too, but id need to double check ws v bit more first.
        for v in range(1, parts + 1):
            start = nbu.ri64((v - 1) * gs)
            end = nbu.ri64(v * gs)
            # if end>tsz:
            #     print('shouldnt be possible',v)
            #     end=max(end,tsz)
            mid = nbu.ri64((v - 0.5) * gs)
            # if mid >= tsz: #shouldn't be possible. as well as the other checks
            #     break  # no pair to merge

            # Copy the left segment [start:mid] into the workspace.
            left_size = mid - start
            for j in range(left_size):  # for complete parallel ws would need to be offset.
                ws[j] = cr[start + j]

            # Merge ws (left) and idxr[mid:end] (right) back into idxr[start:end]
            i = 0  # index for ws (left segment)
            j = mid  # index for right segment in idxr
            k = start  # output index in idxr
            while i < left_size and j < end:
                if not comp_call(cr[j], ws[i], vals):
                    cr[k] = ws[i]
                    i += 1
                else:
                    cr[k] = cr[j]
                    j += 1
                k += 1

            # Copy any remaining elements from the left segment.
            while i < left_size:
                cr[k] = ws[i]
                i += 1
                k += 1


@nbu.jt
def merge_sort(sr: np.ndarray, small_first: bool = True, ws:MArray=None) -> None:
    """
    Top-down merge sort.

    Sorts ``sr`` in-place. If no workspace is provided, one is allocated
    with size ``sr.size // 2``.

    :param sr: Array to sort in-place.
    :param small_first: If True, sort ascending; otherwise descending.
    :param ws: Optional workspace array.
    :returns: None.
    """
    if ws is None: ws = np.empty(sr.size // 2, dtype=sr.dtype)
    if small_first: impl_merge_sort(sr, None, ws, _lessthan)
    else: impl_merge_sort(sr, None, ws, _greaterthan)


@nbu.jt
def arg_merge_sort(sr: np.ndarray, idxr: np.ndarray, small_first: bool = True, ws:MArray=None) -> None:
    """
    Top-down merge argsort.

    Sorts ``idxr`` in-place based on values in ``sr``. If no workspace is
    provided, one is allocated with size ``idxr.size // 2``.

    :param sr: Array used for sort comparison.
    :param idxr: Index array to sort in-place.
    :param small_first: If True, sort ascending by ``sr`` values; otherwise descending.
    :param ws: Optional workspace array.
    :returns: None.
    """
    if ws is None: ws = np.empty(idxr.size // 2, dtype=idxr.dtype)
    if small_first: impl_merge_sort(sr, idxr, ws, _lessthan)
    else: impl_merge_sort(sr, idxr, ws, _greaterthan)


@nbu.jt
def _bu_merge_sort(sr: np.ndarray, small_first: bool = True, ws:MArray=None) -> None:
    """
    Bottom-up merge sort.

    Sorts ``sr`` in-place using the iterative bottom-up implementation.
    If no workspace is provided, one is allocated with size ``sr.size // 2``.

    :param sr: Array to sort in-place.
    :param small_first: If True, sort ascending; otherwise descending.
    :param ws: Optional workspace array.
    :returns: None.
    """
    if ws is None: ws = np.empty(sr.size // 2, dtype=sr.dtype)
    if small_first: impl_bu_merge_sort(sr, None, ws, _lessthan)
    else: impl_bu_merge_sort(sr, None, ws, _greaterthan)


@nbu.jt
def _bu_arg_merge_sort(sr: np.ndarray, idxr: np.ndarray, small_first: bool = True, ws:MArray=None) -> None:
    """
    Bottom-up merge argsort.

    Sorts ``idxr`` in-place based on values in ``sr`` using the iterative
    bottom-up implementation. If no workspace is provided, one is allocated
    with size ``idxr.size // 2``.

    :param sr: Array used for sort comparison.
    :param idxr: Index array to sort in-place.
    :param small_first: If True, sort ascending by ``sr`` values; otherwise descending.
    :param ws: Optional workspace array.
    :returns: None.
    """
    if ws is None: ws = np.empty(idxr.size // 2, dtype=idxr.dtype)
    if small_first: impl_bu_merge_sort(sr, idxr, ws, _lessthan)
    else: impl_bu_merge_sort(sr, idxr, ws, _greaterthan)

"""
Specialized partial insert sorts below, that insert into already sorted array blocks.
For 1*, cost <=O(m).
For n*, cost <=n*O(m).
"""

@nbu.jt
def insert_sort_1l(sr: np.ndarray, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_1l(sr, None, _lessthan)
    else: impl_insert_sort_1l(sr, None, _greaterthan)

@nbu.jt
def arg_insert_sort_1l(sr: np.ndarray, idxr: np.ndarray, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_1l(sr, idxr, _lessthan)
    else: impl_insert_sort_1l(sr, idxr, _greaterthan)

@nbu.jt
def insert_sort_nl(sr: np.ndarray, n: int, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_nl(sr, None, n, _lessthan)
    else: impl_insert_sort_nl(sr, None, n, _greaterthan)

@nbu.jt
def arg_insert_sort_nl(sr: np.ndarray, idxr: np.ndarray, n: int, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_nl(sr, idxr, n, _lessthan)
    else: impl_insert_sort_nl(sr, idxr, n, _greaterthan)

@nbu.jt
def insert_sort_1r(sr: np.ndarray, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_1r(sr, None, _lessthan)
    else: impl_insert_sort_1r(sr, None, _greaterthan)

@nbu.jt
def arg_insert_sort_1r(sr: np.ndarray, idxr: np.ndarray, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_1r(sr, idxr, _lessthan)
    else: impl_insert_sort_1r(sr, idxr, _greaterthan)

@nbu.jt
def insert_sort_nr(sr: np.ndarray, n: int, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_nr(sr, None, n, _lessthan)
    else: impl_insert_sort_nr(sr, None, n, _greaterthan)

@nbu.jt
def arg_insert_sort_nr(sr: np.ndarray, idxr: np.ndarray, n: int, small_first: bool = True) -> None:
    if small_first: impl_insert_sort_nr(sr, idxr, n, _lessthan)
    else: impl_insert_sort_nr(sr, idxr, n, _greaterthan)


@nbu.jt
def impl_insert_sort_1l(sr: np.ndarray, idxr: MArray, comp_call: Callable[[FI, FI, MArray], bool]) -> None:
    cr = idxr if idxr is not None else sr
    vals = sr if idxr is not None else None
    n_elements = cr.shape[0] - 1
    k = cr[0]
    j = 0
    while j < n_elements and comp_call(cr[j + 1], k, vals):
        cr[j] = cr[j + 1]
        j += 1
    cr[j] = k


@nbu.jt
def impl_insert_sort_nl(sr: np.ndarray, idxr: MArray, n: int, comp_call: Callable[[FI, FI, MArray], bool]) -> None:
    cr = idxr if idxr is not None else sr
    vals = sr if idxr is not None else None
    n_elements = cr.shape[0] - 1
    for i in range(n - 1, -1, -1):
        k = cr[i]
        j = i
        while j < n_elements and comp_call(cr[j + 1], k, vals):
            cr[j] = cr[j + 1]
            j += 1
        cr[j] = k



@nbu.jt
def impl_insert_sort_1r(sr: np.ndarray, idxr: MArray, comp_call: Callable[[FI, FI, MArray], bool]) -> None:
    cr = idxr if idxr is not None else sr
    vals = sr if idxr is not None else None
    j = cr.shape[0] - 1
    k = cr[j]
    while j > 0 and comp_call(k, cr[j - 1], vals):
        cr[j] = cr[j - 1]
        j -= 1
    cr[j] = k


@nbu.jt
def impl_insert_sort_nr(sr: np.ndarray, idxr: MArray, n: int, comp_call: Callable[[FI, FI, MArray], bool]) -> None:
    cr = idxr if idxr is not None else sr
    vals = sr if idxr is not None else None
    n_elements = cr.shape[0]
    start_idx = n_elements - n
    for i in range(start_idx, n_elements):
        k = cr[i]
        j = i
        while j > 0 and comp_call(k, cr[j - 1], vals):
            cr[j] = cr[j - 1]
            j -= 1
        cr[j] = k



# I'm adding search sorted here as unlike smooth/non-smooth lines, this is for finite sets.
@nbu.jt
def binary_argsearch(x: np.ndarray, v: Any, unsafe: Any | None = None) -> int:
    """
    Search sorted with tradeoff for a little perf benefit on small arrays, tradeoff calculated for f64.

    For a few more microseconds set unsafe not None, but there MUST be a value in x smaller than v or it
    will not terminate.

    May add Argsearch sorted later.

    :param x: Sorted array.
    :param v: Search value.
    :param unsafe: If not None, enables an unsafe fast-path (see notes above).
    :returns: The insertion index.
    """
    if x.size > 165: return _sqleq_arg(x, v, unsafe)
    else: return np.searchsorted(x, v)


def _sqleq_arg(x: np.ndarray, v: Any, unsafe: Any | None = None) -> int:
    """
    Sequential less than or equal right step. Unsafe not None will give you a small perf boost, but v MUST be in x.
    Note: NASA would hate this.

    :param x: Sorted array.
    :param v: Search value (must be present in ``x`` when unsafe mode is used).
    :param unsafe: If not None, enables an unsafe fast-path.
    :returns: The insertion index.
    """
    return int(np.searchsorted(x, v))


_N = nbu.types.none


@nbu.ovsi(_sqleq_arg)
def _sqleq_arg_(x, v, unsafe=None):  # pragma: no cover
    if unsafe is None or unsafe is _N:

        def impl(x, v, unsafe=None) -> int:
            i, n = 0, x.shape[0]
            while i < n and x[i] < v: i += 1

            return i
    else:

        def impl(x, v, unsafe=None) -> int:
            i = 0
            while x[i] < v: i += 1
            return i

    return impl
