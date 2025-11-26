import numba as nb
import numpy as np
import nbux._utils as nbu
import math as mt




@nbu.jt
def impl_insert_sort(sr, comp_call):
    """
    Insertion sort.
    (Tested several slightly different implementations, this had the best performance for numba njit compilation).
    """
    for i in range(1, sr.shape[0]):
        k = sr[i]
        j = i
        while j > 0 and comp_call(k, sr[j - 1]):
            # Make place for moving A[i] downwards
            sr[j] = sr[j - 1]
            j -= 1
        sr[j] = k

@nbu.jt
def insert_sort(sr, small_first=True):
    if small_first:impl_insert_sort(sr, lambda k1, k2: k1 < k2)
    else:impl_insert_sort(sr, lambda k1, k2: k1 > k2)

@nbu.jt
def impl_arg_insert_sort(sr, idxr, comp_call):
    for i in range(1, idxr.shape[0]):
        k = idxr[i]
        j = i
        while j > 0 and comp_call(sr[k], sr[idxr[j - 1]]):
            # Make place for moving A[i] downwards
            idxr[j] = idxr[j - 1]
            j -= 1
        idxr[j] = k

@nbu.jt
def arg_insert_sort(sr, idxr, small_first=True):
    if small_first:impl_arg_insert_sort(sr,idxr,lambda k1,k2:k1<k2)
    else:impl_insert_sort(sr,idxr,lambda k1,k2:k1>k2)
    


#SMALL_MERGESORT = 20 #original size, ~50 seems to provide better performance profile over large range

def make_merge_sort(argsort=False,top_down=True,ins_sep=56):
    """Taken from numba merge sort implementation, letting the user sort already initialized arrays.
    Also from my testing, this merge sort implementation seems to be always quicker than numba's quick sort.
    Maybe because of too many func pointers and being generic idk.

    Parameters
    ----------
    idxr : array [read+write]
        The values being sorted inplace.  For argsort, this is the
        indices.
    vals : array [readonly]
        ``None`` for normal sort.  In argsort, this is the actual array values.
    ws : array [write]
        The workspace.  Must be of size ``idxr.size // 2``
    """
    SMALL_MERGESORT=ins_sep
    if argsort:
        @nbu.jt
        def lessthan(a, b, vals):
            return vals[a]<vals[b]

    else:
        @nbu.jt
        def lessthan(a, b, vals):
            return a<b

    if top_down:
        @nbu.jt
        def merge_sort(arr, vals, _ws): #assume ws is made by user
            ws=_ws
            if arr.size > SMALL_MERGESORT:
                # Merge sort
                mid = arr.size // 2

                merge_sort(arr[:mid], vals, ws)
                merge_sort(arr[mid:], vals, ws)

                # Copy left half into workspace so we don't overwrite it
                for i in range(mid):
                    ws[i] = arr[i]

                # Merge
                left = ws[:mid]
                right = arr[mid:]
                out = arr

                i = j = k = 0
                ls=left.size
                rs=right.size
                while i < ls and j < rs:
                    if not lessthan(right[j], left[i], vals):
                        out[k] = left[i]
                        i += 1
                    else:
                        out[k] = right[j]
                        j += 1
                    k += 1

                #Leftovers
                while i < left.size:
                    out[k] = left[i]
                    i += 1
                    k += 1

                #unecessary because if we get here, out[k] is literally the same memory address as right[j]
                # while j < right.size:
                #     out[k] = right[j]
                #     j += 1
                #     k += 1
            else:
                #fastest insert sort style I found in numba after testing several slightly different styles.
                for i in range(1, arr.shape[0]):
                    k = arr[i]
                    j = i
                    while j > 0 and lessthan(k, arr[j - 1],vals):
                        # Make place for moving A[i] downwards
                        arr[j] = arr[j - 1]
                        j -= 1
                    arr[j] = k
    else:
        #WARNING TO USERS. In theory this should be faster because no recursion and the parallelized insert sort, while the insert sort pl does improve performance the merge sort bottom_up loops run significantly slower.
        #Idk why, maybe it has to do with rounding to the nearest index, perhaps initializing them outside of the loops or another method would let this method surpass the recursive stack method.
        @nbu.jtp
        def merge_sort(arr, vals, _ws,*__ext):
            #implement bottom up merge sort that first identifies the smallest # of partitions that equally divides the array such that
            #each array is smaller than or equal to SMALL_MERGESORT and the # of partitions is a power multiple of 2.
            #then it performs these insert sorts for each group.
            #next begins a bottom up merging like how it's done in top down only we use two for loops, the outer one specifying the merge layer and the inner loop placing back into the array to be sorted.
            tsz=arr.size
            nlayers=mt.ceil(mt.log2(tsz/SMALL_MERGESORT))
            parts=2**nlayers
            gs=tsz/parts
            for v in nb.prange(1,parts+1): #parallelizable the merging part technically could be as well but less efficient.
                st=nbu.ri64((v-1)*gs)
                ed=nbu.ri64(v*gs)
                for i in range(st, ed):
                    k = arr[i]
                    j = i
                    while j > 0 and lessthan(k, arr[j - 1],vals):
                        # Make place for moving A[i] downwards
                        arr[j] = arr[j - 1]
                        j -= 1
                    arr[j] = k

            ws = _ws  # workspace array provided by user

            # Merge layers: in each iteration, merge adjacent pairs of segments of size current_size.
            #won't run quicker for some reason.
            for _ in range(nlayers):
                # Iterate over the array in steps of 2*current_size
                parts//=2
                gs*=2
                for v in range(1, parts+1): #should be pretty close to parallelizing this too, but id need to double check ws a bit more first.
                    start = nbu.ri64((v - 1) * gs)
                    end = nbu.ri64(v * gs)
                    # if end>tsz:
                    #     print('shouldnt be possible',v)
                    #     end=max(end,tsz)
                    mid = nbu.ri64((v - .5) * gs)
                    # if mid >= tsz: #shouldn't be possible. as well as the other checks
                    #     break  # no pair to merge

                    # Copy the left segment [start:mid] into the workspace.
                    left_size = mid - start
                    for j in range(left_size): #for complete parallel ws would need to be offset.
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


    return merge_sort

arg_merge_sort=make_merge_sort(True,True,56)
#merge_sort20=make_merge_sort(False,20)
merge_sort=make_merge_sort(False,True,56)
#merge_sort1=make_merge_sort(False,True,56)
#merge_sort2=make_merge_sort(False, False,56)
