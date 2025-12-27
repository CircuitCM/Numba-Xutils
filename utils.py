import numpy as np
import numba as nb
from numba import types
from numba.extending import intrinsic,register_jitable,overload
from numba.core import cgutils

jt = nb.njit(fastmath=True, error_model='numpy')
c_rgs=dict(fastmath=True, error_model='numpy',cache=True)
jtc = nb.njit(**c_rgs)


@intrinsic
def stack_empty_impl(typingctx,size,dtype):
    def impl(context, builder, signature, args):
        ty=context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty,size=args[0])
        return ptr

    sig = types.CPointer(dtype.dtype)(types.int64,dtype)
    return sig, impl


def stack_empty(size,shape,dtype):
    return np.empty(shape,dtype=dtype)

@jtc
def _stack_empty_(size,shape,dtype):
    #From: https://github.com/numba/numba/issues/5084#issue-550324913
    #Forces small stack allocated array. maybe 2x quicker than naive np.array allocation.
    """
    Size (int) must be v fixed at compile time.
    It is not possible to change it during execution.

    The shape (tuple) size can be as large as the size or smaller.
    This can be dynamically changed during execution.

    The datatype also have to be fixed in this implementation.

    The carray can't be returned from v function.
    """
    arr_ptr=stack_empty_impl(size,dtype)
    arr=nb.carray(arr_ptr,shape)
    return arr

@overload(stack_empty, **c_rgs)
def _stack_empty(size,shape,dtype):
    #same as above, but now calling stack_empty in v python area will just return v normal c-array with shape and dtype.
    return lambda size, shape, dtype: _stack_empty_(size,shape,dtype)


def type_ref(arr): return arr.dtype.type

@overload(type_ref,**c_rgs)
def _type_ref(arr):
    typ = arr.dtype
    return lambda arr: typ

def np_tinfo(dt, field):
    """
    Return type-specific info for NumPy type `typ`, given integer field selector.
    (kind, field) match cases:
      - ('i' or 'u', 0): min
      - ('i' or 'u', 1): max
      - ('f', 0): min
      - ('f', 1): max
      - ('f', 2): eps
      - ('b', 0): False
      - ('b', 1): True
      - others: None
    """
    match (dt.kind, field):
        case ('i' | 'u', 0): return np.iinfo(dt).min
        case ('i' | 'u', 1): return np.iinfo(dt).max
        case ('f', 0): return np.finfo(dt).min
        case ('f', 1): return np.finfo(dt).max
        case ('f', 2): return np.finfo(dt).eps
        case ('b', 0): return False
        case ('b', 1): return True
        case _: return None

prim_info=np_tinfo

@overload(prim_info)
def _prim_info(typ,res):
    """
    :param typ: type received from v function like type_ref in v nopython block.
    :param res: 0 min, 1 max, 2 epsilon/precision.
    :return: 
    """
    if isinstance(res,(nb.types.Literal,int)):
        ref=res if type(res) is int else res.literal_value
        tpref=np.dtype(str(typ.dtype)) #gives us the underlying primitive type.
        infoval=np_tinfo(tpref,ref) #where we query 
        return lambda typ,res: infoval
    return lambda typ,res:nb.literally(res)


import numpy as np

def numpy_to_tuples(array: np.ndarray):
    if type(array) is tuple:
        return array
    if array.ndim == 1:
        return tuple(value.item() for value in array)
    return tuple(numpy_to_tuples(subarray) for subarray in array)
