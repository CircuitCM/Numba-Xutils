from __future__ import annotations

import inspect
import os
import textwrap
import warnings
from types import NoneType
from typing import Any, Callable, Sequence

import numba as nb
import numba.core.errors as nb_error
import numpy as np
from numba import types
from numba.core import cgutils
from numba.core.errors import NumbaPerformanceWarning
from numba.extending import intrinsic, overload, register_jitable
from numba.np.numpy_support import as_dtype

_N = types.none
# Some shorthands
unroll = nb.literal_unroll
CSeq = tuple[Any, ...] | list[Any]
CSeqRuntime = (tuple, list)


# implements
def njit_no_parallelperf_warn(**njit_kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    """
    Drop-in replacement for ``nb.njit(parallel=True)`` that suppresses the
    "parallel=True but no parallel transform" warning for this function only.

    :param njit_kwargs: Keyword arguments forwarded to ``numba.njit``.
    :returns: A decorator.
    """
    jt = nb.njit(**njit_kwargs)

    def decorator(func: Callable[..., Any]) -> Any:
        name = func.__name__  # might have to become __qualname__
        pattern = f"(?s)(?=.*parallel=True)(?=.*{name}).*"
        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning, message=pattern)
        return jt(func)

    return decorator


def rg_no_parallelperf_warn(rrg: Callable[[Callable[..., Any]], Any]) -> Callable[[Callable[..., Any]], Any]:
    def decorator(func: Callable[..., Any]) -> Any:
        name = func.__name__
        pattern = f"(?s)(?=.*parallel=True)(?=.*{name}).*"
        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning, message=pattern)
        return rrg(func)

    return decorator


# This only changes once at import time.
# --- Numba Global Fastmath : you can frequently get a 2x performance boost, for virtually no loss in error range (only
# 4x larger than 2**-{# your bit precision range})
_fm = os.environ.get("NB_GLOB_FM", "true")
_fm = False if not _fm or _fm.lower() in "false" else eval(_fm) if any(i in _fm for i in ("[", "{", "(")) else True
# --- Numba Global Error Model : 'numpy'|'python' will do less checks but frequently has much faster jitted code,
# because it isn't forced to fall back to larger data types, [see this article]().
_erm = os.environ.get("NB_GLOB_EM", "numpy")


"""
Before continuation some notes on this section:

## Configurations
s|p : Sync or Parallel, the threading strategy.
c : Cache the compilation for new signatures.
n : nogil, does numba bypass the global interpreter lock of python (array writes are not thread safe!).
i : Manual/forced Numba-IR level inline. Convenient if types are causing compilations to fail, sometimes to force
inlining of low level operations if LLVM won't do it.

## Decorators
jt - Numba jit using the base defaults and extension characters seen above.
rg - Register Jittable, these functions will compile into the Numba IR but run as python when called from the
interpreter. This can actually improve performance significantly for the python function as Numba functions have a high
overhead cost to call. you can call a jitted functions py func by `jitfunc.py_func(*args, **kwargs)`.
ov - Override decorators. See the numba docs for coverage on this.

From my past experience, these behaviors might be true, but need to be verified from source:
- Cache:
    - Great for reducing initial startup times by not needing to recompile the entire library. This even significantly
    benefits compilation speeds of functions that can't be cached but utilize cached functions.
    - If your procedure is compiling with reference to an already cached function, it will **always** treat it as a
    function pointer and not benefit from LLVM's unified compilations. Therefore you shouldn't cache small external
    functions, caching should be at the scope of larger procedures.
- Inline:
    - Inline decorated functions will make the first compile slower. If you want to be very certain the procedure is
    being compiled as one scope, by sacrificing a little compilation speed, this inline option will help. Usually Inline
    doesn't do anything.
- Nogil:
    - GIL bypass seems to occur before execution of the compiled source. Meaning that any nogil decorations on inner
    linked functions will have no effect. For Numba library developers it means that you only need to wrap the outermost
    scope that you expect users to call in to.
- Parallel:
    - Will **not** cache if you modify the block size of dispatch. Normally leaving it on automatic (0) works well for
    large vectorized array operations, however if you use it more like a dispatch or work queue you may want
    significantly smaller blocks (optimal for asymmetric or long running tasks). You could change block size outside of
    the parallel block though. It also won't cache if you request thread scope IDs.
    - Will run parallel in scope. So if you call a @jtp function from within a @jt function you will get parallel
    benefits, but expecting parallel dispatch from a @jt function within @jtp block is not applicable.
    - Using a parallel decorator but a branch to decide if the block should be run parallel or not, will **always**
    compile in the parallel and sync branch, even if the branch comes from a constant. If you want the low overhead
    benefits of the sync branch alone, then you need to define them as different functions, they could then be tied in
    with a custom overloads. To make this situation generic or at least less boilerplate to implement, I developed a
    hacky approach below.
    - If you have multiple layers of `prange` only the outer one will be parallelized, **unless** all inner ranges are
    the same, (see here)[]. Example:
    ```python
    #Will parallelize over both
    for n in nb.prange(100_000):
        for m in nb.prange(100_000):
            pass
    
    #Will not parallelize...
    for n in nb.prange(100_000):
        for m in nb.prange(90_000):
            pass
    ```    
"""

_dft = dict(fastmath=_fm, error_model=_erm)  # base python arguments.
jit_s = _dft
jit_sn = jit_s | dict(nogil=True)
jit_sc = jit_s | dict(cache=True)
jit_scn = jit_sc | dict(nogil=True)
jit_si = jit_s | dict(inline="always")
jit_sci = jit_si | dict(cache=True)
jit_scin = jit_sci | dict(nogil=True)

jit_p = _dft | dict(parallel=True)
jit_pn = jit_p | dict(nogil=True)
jit_pc = jit_p | dict(cache=True)
jit_pcn = jit_pc | dict(nogil=True)
jit_pi = jit_p | dict(inline="always")
jit_pci = jit_pi | dict(cache=True)

# --- JIT DECORATORS
jt = nb.njit(**jit_s)  # plain jit
jtc = nb.njit(**jit_sc)  # cache
jtn = nb.njit(**jit_sn)  # nogil
jtnc = nb.njit(**jit_scn)  # nogil and cache
jti = nb.njit(**jit_si)  # inline
jtic = nb.njit(**jit_sci)  # inline and cache - might have no effect as inline forces code injection.
# there is barely a concievable need for both inline and nogil in the same decorator.
# jtinc=nb.njit(**jit_scin)

jtp = nb.njit(**jit_p)  # jit parallel.
jtp_s = njit_no_parallelperf_warn(**jit_p)  # jit parallel - silence no parallel warning.
jtpc = nb.njit(**jit_pc)
jtpc_s = njit_no_parallelperf_warn(**jit_pc)
# we might want parallel nogil if we eg run less python threads than number of cores and we limit thread dispatch in the
# parallel block
jtpn = nb.njit(**jit_pn)
jtpn_s = njit_no_parallelperf_warn(**jit_pn)
jtpnc = nb.njit(**jit_pcn)
jtpnc_s = njit_no_parallelperf_warn(**jit_pcn)
# Parallel + inline are not relevant together - inline will push the code to the next function scope, this scope will
# need a non-inline parallel to run the prange
# jtpi=nb.njit(**jit_pi)
# jtpi_s=njit_no_parallelperf_warn(**jit_pi)
# jtpic=nb.njit(**jit_pci)
# jtpic_s=njit_no_parallelperf_warn(**jit_pci)

# --- REGISTER JITTABLE DECORATORS
_rg = register_jitable
rg = _rg(**jit_s)  # base sync
rgc = _rg(**jit_sc)  # cache
# in theory nogil is never utilized with register jittable, if you call rg from python scope it will run the py func.
# rgn=_rg(**jit_sn) #nogil
# rgnc=_rg(**jit_scn) #nogil cache
rgi = _rg(**jit_si)  # Inline
rgic = _rg(**jit_sci)  # inline cache

rgp = _rg(**jit_p)
rgp_s = rg_no_parallelperf_warn(rgp)
rgpc = _rg(**jit_pc)
rgpc_s = rg_no_parallelperf_warn(rgpc)


# --- OVERLOADS DECORATORS
# I'm pretty sure caching is redundant for overloads, but assuming not and including.
def ovs(impl: Callable[..., Any]) -> Callable[..., Any]: return overload(impl, jit_options=jit_s)


def ovsi(impl: Callable[..., Any]) -> Callable[..., Any]: return overload(impl, jit_options=jit_s, inline="always")


def ovsc(impl: Callable[..., Any]) -> Callable[..., Any]: return overload(impl, jit_options=jit_sc)


def ovsic(impl: Callable[..., Any]) -> Callable[..., Any]: return overload(impl, jit_options=jit_sc, inline="always")


# It's also possible parallel is never directly utilized by overloads.
def ovp(impl: Callable[..., Any]) -> Callable[..., Any]: return overload(impl, jit_options=jit_p)


def ovpc(impl: Callable[..., Any]) -> Callable[..., Any]: return overload(impl, jit_options=jit_pc)


# shorthand
fb_ = np.frombuffer


def compiletime_parallelswitch() -> None: pass  # pragma: no cover


@intrinsic
def stack_empty_impl(typingctx, size, dtype):  # pragma: no cover
    """
    Low level llvm call for stack_empty.

    :param size: Number of elements to allocate.
    :param dtype: Target dtype.
    :returns: A pointer to the stack-allocated buffer.
    """

    def impl(context, builder, signature, args):
        ty = context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty, size=args[0])
        return ptr

    sig = types.CPointer(dtype.dtype)(types.int64, dtype)  # int64 is the os level pointer dtype, may need to change
    return sig, impl


def stack_empty(size: int, shape: int | tuple[int, ...], dtype: Any) -> np.ndarray:
    """
    Create a small stack-allocated array (Numba).

    ``size`` must be fixed at compile time and cannot change during execution.
    ``shape`` may be dynamic, but must be no larger than ``size``. ``dtype`` is
    fixed for the compiled implementation.

    Notes
    -----
    The carray cannot be returned from a function.

    Reference: https://github.com/numba/numba/issues/5084#issue-550324913

    :param size: Number of elements to allocate.
    :param shape: Target shape.
    :param dtype: Target dtype.
    :returns: An empty array/carray with the requested shape and dtype.
    """
    return np.empty(shape, dtype=dtype)


@jtc
def stack_empty_(size: int, shape: int | tuple[int, ...], dtype: Any) -> np.ndarray:  # pragma: no cover
    """
    The Numba implementation of ``stack_empty``.

    :param size: Number of elements to allocate.
    :param shape: Target shape.
    :param dtype: Target dtype.
    :returns: A stack-allocated carray with the requested shape and dtype.
    """
    arr_ptr = stack_empty_impl(size, dtype)  # type: ignore[bad-argument-count]
    arr = nb.carray(arr_ptr, shape)
    return arr


@ovs  # c
def _stack_empty(size, shape, dtype):  # pragma: no cover
    """
    Overload for ``stack_empty``.

    Calling ``stack_empty`` from Python returns a normal NumPy array with the
    requested shape and dtype.

    :param size: Number of elements to allocate.
    :param shape: Target shape.
    :param dtype: Target dtype.
    :returns: An overload implementation for ``stack_empty``.
    """
    return lambda size, shape, dtype: stack_empty_(size, shape, dtype)


# can probably use this to replace the self archive tracking for de-bpesta mutation.
@intrinsic
def nb_val_ptr(typingctx, data):  # pragma: no cover
    """Get the address of any numba primitive value.

    Use for setting or getting values in some non-local setting within numba, e.g. a call to certain LAPACK or BLAS
    routines with a value return.

    :param data: Primitive value to take the address of.
    :returns: A pointer to ``data``.

    Example
    -------
    Within a Numba ``njit`` block:

    .. code-block:: python

        a = np.int64(5)

        ap = nb_val_ptr(ap)
        ap += 1

        print(nb_pointer_val(ap))

    Output:

    .. code-block:: text

        6
    """

    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder, args[0])
        return ptr

    sig = types.CPointer(data)(data)
    return sig, impl


@intrinsic
def nb_ptr_val(typingctx, data):  # pragma: no cover
    """Get the value from a pointer/address.

    See ``nb_val_ptr``.

    :param data: Pointer/address to load.
    :returns: The loaded value.
    """

    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val

    sig = data.dtype(data)
    return sig, impl


@intrinsic
def nb_array_ptr(typingctx, arr_typ):
    """
    Get the base pointer offset of an extant array.

    :param arr_typ: Array type.
    :returns: A pointer to the base of the underlying array buffer.
    """
    elptr = types.CPointer(arr_typ.dtype)

    def codegen(ctx, builder, sig, args): return ctx.make_helper(builder, sig.args[0], args[0]).data

    return elptr(arr_typ), codegen


@jtic
def buffer_nelems_andp(arr: np.ndarray) -> tuple[int, Sequence]:  # pragma: no cover
    """
    Return the total element count and a pointer-like view for an array buffer.

    This is primarily a helper for iterating over potentially non-contiguous
    arrays by operating on the contiguous backing buffer (along the leading
    axis; supports Fortran and C ordering).

    - ``ptr`` is the dtype-correct offset to the start of the underlying buffer.
    - ``nelems`` is the total number of values in the backing buffer, including
      any extra fill needed for a discontinuous leading dimension. If only the
      leading dimension is discontinuous, the final element is accessible as
      ``ptr[nelems - 1]``.

    :param arr: Input array.
    :returns: ``(nelems, ptr)`` where ``ptr`` can be indexed like a 1D sequence.
    """
    # We will tell typing that this is a Sequence, because a datatype pointer will have a __getitem__.
    ptr: Sequence = nb_array_ptr(arr)  # type: ignore[bad-assignment]
    size = arr.size
    if arr.ndim < 2: return size, ptr  # noqa: E701
    # fast C vs F test via stride magnitudes
    s1, s2 = (
        (-1, -2) if abs(arr.strides[0]) >= abs(arr.strides[-1]) else (0, 1)
    )  # we use this to detect if C or F ordered
    tail_prod = size // arr.shape[s1]
    first_len = abs(arr.strides[s2]) // abs(arr.strides[s1])
    nelems: int = tail_prod * first_len
    # There is so much less implementation code than as seen in np.frombuffer from numba, because we are only concerned
    # with the contiguous 1D buffer.
    return nelems, ptr


# WARNING fastmath can produce different results.
# The only one that is a little faster than np.round is the int32
def ri64(rd: float) -> int: return int(rd + 0.5)


ri32 = ri64
ovsic(ri64)(lambda rd: (lambda rd: nb.int64(rd + 0.5)))
ovsic(ri32)(lambda rd: (lambda rd: nb.int32(rd + nb.float32(0.5))))

# def ri64s(rd:float,s): return int((rd/s) + .5)*s
# ri32s=ri64s
# overload(ri64s, **cfg.jit_s,cache=True)(lambda rd,s: (lambda rd,s: nb.int64((rd/s) + .5)*s))
# overload(ri32s, **cfg.jit_s,cache=True)(lambda rd,s: (lambda rd,s: nb.int32((rd/s) + nb.float32(.5))*s))


@rgic
def display_round(f: float, m: int = 1, s: int = 10) -> float:
    """
    A method to round floats so that when printed from numba's stdout it doesn't show roundoff tail error.

    Note: this still doesn't work, looking for something that does.

    :param f: Input value.
    :param m: Base-10 order offset.
    :param s: Significant digits.
    :returns: Rounded value.
    """
    # so that we don't get roundoff tails and mess up the string, we  instead float -> int -> float it.
    # mv and s are integers that represent base 10 factors, mv is b10 order offset and s is significant digits, s=0 no
    # digits, mv=1 no offset.
    m10 = 10.0 ** (m - 1)
    s10 = 10.0**s
    return np.float64(np.int64(f * s10 * m10 + 0.5)) / s10


"""
## Supported Numpy types in Numba:

| Type name(s)    | Shorthand | Comments                               |
|-----------------|-----------|----------------------------------------|
| boolean         | b1        | represented as a byte                  |
| uint8, byte     | u1        | 8-bit unsigned byte                    |
| uint16          | u2        | 16-bit unsigned integer                |
| uint32          | u4        | 32-bit unsigned integer                |
| uint64          | u8        | 64-bit unsigned integer                |
| int8, char      | i1        | 8-bit signed byte                      |
| int16           | i2        | 16-bit signed integer                  |
| int32           | i4        | 32-bit signed integer                  |
| int64           | i8        | 64-bit signed integer                  |
| intc            | –         | C int-sized integer                    |
| uintc           | –         | C int-sized unsigned integer           |
| intp            | –         | pointer-sized integer                  |
| uintp           | –         | pointer-sized unsigned integer         |
| float32         | f4        | single-precision floating-point number |
| float64, double | f8        | double-precision floating-point number |
| complex64       | c8        | single-precision complex number        |
| complex128      | c16       | double-precision complex number        |

We can get the type using type_ref and check it against np.dtype's within the jit scope.
But it has to be single boolean statements eg typ is np.float64 or typ is np.float32. Likely because this is compile 
time and not run time.

"""


def type_ref(arg: Any) -> type[Any]:
    """Get the data type of an array, otherwise get the type of a value.

    Works in python and numba blocks.

    Useful for gathering type specific information at signature compile time. E.g. dtype epsilon, or min-max values.

    :param arg: Value or array.
    :returns: A dtype/type reference for ``arg``.
    """
    if isinstance(arg, np.ndarray): return arg.dtype.type
    else: return type(arg)


@ovsic(type_ref)
def _type_ref(arg):  # pragma: no cover
    """
    Numba overloads for type_ref.

    :param arg: Value or array.
    :returns: A dtype/type reference for ``arg``.
    """
    # now supports primitive types, literals and array memory type.
    if isinstance(arg, types.Literal):  # we should never even get this result
        typ = arg._literal_type_cache
        return lambda arg: typ
    elif isinstance(arg, types.Array):
        typ = arg.dtype
        return lambda arg: typ
    else:
        typ = arg  # It's already the correct type. it only sees type in this scope not value
        return lambda arg: typ


def if_val_cast(typ: type[Any], val: Any) -> Any:
    if isinstance(val, (np.ndarray, Sequence)): return val
    else: return typ(val)


@ovsic(if_val_cast)
def _if_val_cast(typ, val):  # pragma: no cover
    """
    Overloads impl.

    :param typ: Target type.
    :param val: Input value.
    :returns: An overload implementation for ``if_val_cast``.
    """
    if isinstance(val, types.IterableType): return lambda typ, val: val
    else: return lambda typ, val: typ(val)


Op = Callable[..., Any] | NoneType | CSeq


def op_call(call_op: Op, defr: Any = True) -> Any:
    """
    An evaluator (caller) for call operators.

    Specifically targets constraints in numba first class functions, and also works in no-python mode.

    It has the form:

    .. code-block:: python

        op = (callable, *args)
        #or maybe:
        op = (callable, arg1, arg2, arg3)

    Why is this useful? Numba implements first class functions. Depending on the definition, it supports
    full LLVM optimization and cached signatures, e.g.:

    .. code-block:: python

        @nbux.jt
        def ctest(arg1, arg2):
            return arg1-arg2

        @nbux.jt
        def fclass_test(func_op, arg3):
            return func_op[0](*func_op[1:]) + arg3

        @nbux.jtc
        def static_compile(a,b,c):
            #supports caching and the byte code will produce an ``a - b + c`` hot path.
            return fclass_test((ctest,a,b),c)

    Or it can be called externally, in which case ``ctest`` and ``fclass_test`` will be compiled separately
    and linked by function pointers (in this example it will produce a much slower subtract add operation).

    .. code-block:: python

        fclass_test((ctest,a,b),c)

    We get the same result but now ctest will not have a compilation overhead if called again from
    the python interpreter.

    ``op_call`` and it's offspring ``op_args`` simplify this process:

    .. code-block:: python

        @nbux.jt
        def fclass_test(func_op, arg3):
            return op_call(func_op) + arg3

    The purpose of ``op_call`` is to provide additional flexibility by allowing ``call_op`` to be None, a function,
    or a ``Sequence[Callable, *Any]``. See ``op_call_args`` and ``op_args`` for more examples.

    :param call_op: ``None``, a callable, or an operator sequence.
    :param defr: Default return value when ``call_op`` is ``None`` (or invalid).
    :returns: The evaluated result, or ``defr``.
    """

    if callable(call_op): return call_op()
    elif isinstance(call_op, (tuple, list)):
        if isinstance(call_op[0], Callable): return call_op[0](*call_op[1:])
    # In case we want a default value to return when is optional. eg a premature stopping criterial for an algorithm
    # when not used should return True always for a 'should continue?'
    if defr is not None: return defr
    return call_op


@ovsic(op_call)
def _op_call(call_op: Op, defr: bool = True):  # pragma: no cover
    if isinstance(call_op, types.Callable): return lambda call_op, defr=True: call_op()
    # ruff: disable[F821]
    elif isinstance(call_op, types.BaseTuple | types.LiteralList):
        if isinstance(call_op[0], types.Callable): return lambda call_op, defr=True: call_op[0](*call_op[1:])
    if defr is not _N: return lambda call_op, defr=True: defr
    return lambda call_op, defr=True: call_op


def op_call_args(call_op: Op, args: CSeq | Any = (), defr: Any = None) -> Any:
    """
    A callable with arguments supplied either directly or via an operator tuple.

    ``op_call_args`` accepts either:

    - a callable ``call_op``, or
    - a sequence whose first element is a callable and whose remaining elements
      are pre-bound arguments,

    and applies ``args`` to it. If ``args`` is a tuple or list it is expanded;
    otherwise it is treated as a single argument.

    Example:

    .. code-block:: python

        @nbux.jt
        def callop(a, b, c):
            return a + b + c

        # plain callable
        nbux.op_call_args(callop, (1, 2, 3))
        nbux.op_call_args(callop, 1)

        # operator with attached arguments
        op = (callop, 10)

        nbux.op_call_args(op, (2, 3))
        # -> callop(2, 3, 10)

        nbux.op_call_args(op, 5)
        # -> callop(5, 10)

    :param call_op: Callable or tuple/list whose first element is callable, remaining elements are fixed arguments.
    :param args: Arguments to apply, either as a tuple/list or a single value.
    :param defr: Default return value when ``call_op`` is ``None``.
    :returns: Function output.
    """

    if isinstance(call_op, NoneType):  # so ruff doesn't complain
        if defr is None: return call_op
        else: return defr

    ct = callable(call_op)  # otherwise CSeq
    rt = isinstance(args, CSeqRuntime)  # otherwise single element.
    if ct:
        if rt: return call_op(*args)
        return call_op(args)
    else:
        if rt: return call_op[0](*args, *call_op[1:])
        return call_op[0](args, *call_op[1:])


@ovsic(op_call_args)
def _op_call_args(call_op, args=(), defr=None):  # pragma: no cover
    """
    ``op_call_args`` overload for the Numba implementation.

    :param call_op: Callable or operator sequence.
    :param args: Arguments to apply.
    :param defr: Default return value when ``call_op`` is ``None``.
    :returns: An overload implementation for ``op_call_args``.
    """
    # ruff: disable[F821]
    if call_op is _N:
        if defr is _N or defr is None: return call_op
        else: return defr

    ct = isinstance(call_op, types.Callable)
    rt = isinstance(args, (types.BaseTuple, types.LiteralList))

    if ct:
        if rt: return lambda call_op, args=(), defr=None: call_op(*args)
        return lambda call_op, args=(), defr=None: call_op(args)
    else:
        if rt: return lambda call_op, args=(), defr=None: call_op[0](*args,*call_op[1:])
        return lambda call_op, args=(), defr=None: call_op[0](args,*call_op[1:])


def op_args(call_op: Op, args: CSeq | Any = (), defr: Any = None) -> Any:
    """
    Previously called ``op_call_args_v2`` as ``op_call_args``'s successor, only difference being we send
    ``call_op``'s own arguments before ``args``.

    This method is more flexible than ``op_call_args`` because it allows the underlying callable to still have
    a variable portion of arguments when used at different entry points. While any arguments in ``call_op[1:]``
    e.g. array work memory addresses or config values, can be treated like generic or unobserved constants.

    Example:

    .. code-block:: python

        @nbux.jt
        def callop(a,b,c,d,*args):
            pass

        op=(callop, 1,2,3)

        @nbux.jt
        def testop(op):

            nbux.op_args(op, (7,9,6))
            nbux.op_args(op, (.1,.2))

    Note how in the example, 7 and 0.1 would occupy argument ``d``, and the rest fall into the variable ``*args``
    sequence. Args length can be used within a numba block, and its elements are accessed statically. But
    ``op_call_args`` would fail for the same setting.


    :param call_op: The callable, or callable operator and attached parameters.
    :param args: The other implementation arguments that are treated as external or problem-specific inputs.
    :param defr: Default return value when ``call_op`` is ``None``.
    :returns: Function output.
    """

    if isinstance(call_op, NoneType):
        if defr is None: return call_op
        else: return defr

    ct = callable(call_op)
    rt = isinstance(args, CSeqRuntime)
    if ct:
        if rt: return call_op(*args)
        return call_op(args)
    else:
        if rt: return call_op[0](*call_op[1:], *args)
        return call_op[0](*call_op[1:], args)


@ovsic(op_args)
def _op_args(call_op, args=(), defr=None):  # pragma: no cover
    """
    ``op_args`` overload for the Numba implementation.

    :param call_op: Callable or operator sequence.
    :param args: Arguments to apply.
    :param defr: Default return value when ``call_op`` is ``None``.
    :returns: An overload implementation for ``op_args``.
    """
    # ruff: disable[F821]
    if call_op is _N:
        if defr is _N or defr is None: return call_op
        else: return defr

    ct = isinstance(call_op, types.Callable)
    rt = isinstance(args, (types.BaseTuple, types.LiteralList))

    if ct:
        if rt: return lambda call_op, args=(), defr=None: call_op(*args)
        return lambda call_op, args=(), defr=None: call_op(args)
    else:
        if rt: return lambda call_op, args=(), defr=None: call_op[0](*call_op[1:], *args)
        return lambda call_op, args=(), defr=None: call_op[0](*call_op[1:], args)


@rgc
def aligned_buffer(n_bytes: int, align: int = 64) -> np.ndarray:
    """
    Return an aligned ``uint8`` view of length ``n_bytes``.

    A slightly oversized buffer is allocated and then manually aligned by
    slicing so that ``result.ctypes.data % align == 0``. The extra capacity is
    not exposed by the returned view.

    This function is also in ``numpy_buffermap`` but is included here so it is
    not a required dependency.

    :param n_bytes: Logical size of the returned view (in bytes).
    :param align: Desired byte alignment (power-of-two is assumed).
    :returns: A buffer view that is aligned to an ``align``-byte boundary.
    """
    raw = np.empty(n_bytes + align, dtype=np.uint8)
    offset = (-raw.ctypes.data) & (align - 1)
    return raw[offset : offset + n_bytes]


def prim_info(dt: Any, field: int) -> Any:
    """
    Return type-specific info for NumPy type ``dt``, given integer field selector.

    (kind, field) match cases:

    - ('i' or 'u', 0): min
    - ('i' or 'u', 1): max
    - ('f', 0): min
    - ('f', 1): max
    - ('f', 2): eps
    - ('b', 0): False
    - ('b', 1): True
    - ('c', 0): min (complex with min real/imag)
    - ('c', 1): max (complex with max real/imag)
    - ('c', 2): eps (complex with float eps)
    - (any, 3): itemsize (bytes)
    - others: None

    This only supports np.dtype object instances, or anything with the kind field.
    So get the type first with ``type_ref``.

    :param dt: A NumPy dtype (or dtype-like).
    :param field: Field selector (see list above).
    :returns: The requested field value, or ``None``.
    """
    if not hasattr(dt, "kind"): dt = np.dtype(dt)

    match (dt.kind, field):
        # Integer & unsigned integer
        case ("i" | "u", 0):
            return np.iinfo(dt).min
        case ("i" | "u", 1):
            return np.iinfo(dt).max
        # Floating point
        case ("f", 0):
            return np.finfo(dt).min
        case ("f", 1):
            return np.finfo(dt).max
        case ("f", 2):
            return np.finfo(dt).eps
        # Boolean
        case ("b", 0):
            return False
        case ("b", 1):
            return True
        # Complex types
        case ("c", 0):
            return complex(np.finfo(dt).min, np.finfo(dt).min)
        case ("c", 1):
            return complex(np.finfo(dt).max, np.finfo(dt).max)
        case ("c", 2):
            return np.finfo(dt).eps  # hmmm
        # Universal: byte size
        case (_, 3):
            return dt.itemsize
        # Fallback
        case _:
            return None


np_tinfo = prim_info


@ovsic(prim_info)
def _prim_info(typ, res):
    """
    Overloads for primitives info. Implementation for numba mode.

    :param typ: type received from a function like type_ref in a nopython block.
    :param res: 0 min, 1 max, 2 epsilon/precision.
    :returns: The requested field value.
    """
    if isinstance(res, (nb.types.Literal, int)):
        ref = res if isinstance(res, int) else res.literal_value  # `type(res) is` unliked by pyrefly
        tpref = as_dtype(typ)
        infoval = np_tinfo(tpref, ref)  # where we query
        return lambda typ, res: infoval
    return lambda typ, res: nb.literally(res)  # literal value request makes this compile time but still cacheable.


@jtic
def placerange(r: np.ndarray, start: int = 0, step: int = 1) -> None:
    """
    Like numpy arange but for existing arrays. Start and step may be float values.

    :param r: Output array.
    :param start: Starting value.
    :param step: Step value.
    :returns: None.
    """
    for i in range(r.shape[0]): r[i] = start + i * step


@rgi
def swap(x: np.ndarray, i: int, j: int) -> None:
    """Array element swap shorthand.

    :param x: 1D array to perform element swap on.
    :param i: First element index.
    :param j: Second element index.
    :returns: None.
    """
    t = x[i]
    x[i] = x[j]
    x[j] = t


def force_const(val: Any) -> Any:
    """
    Within a numba block this forces referenced variables to become literal, mainly
    kwargs and args from the function header. Numba procedures compiled with a
    ``force_const`` will still cache, but it seems that it will not save signatures with
    different values meaning it will recompile from the python scope each time the value
    is changed. This is true even before the interpreter restarts.

    Therefore ``force_const`` may not have a meaningful use case, because non-cached numba functions called from a
    numba scope with constant value arguments will already compile with branch reductions and hot paths.

    :param val: Value to treat as a literal/compile-time constant.
    :returns: ``val``.
    """
    return val


@ovs(force_const)
def _force_const(val):  # pragma: no cover
    if isinstance(val, types.Literal):
        # tv=val.literal_value
        # print('const',tv)
        return lambda val: val
    else: return lambda val: nb.literally(val)


def run_py(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Numba's base python definition is inside the ``py_func`` field, if it exists we try to call it here.

    :param func: callable.
    :param args: Variable unnamed ordered args.
    :param kwargs: Variable named unordered kwargs.
    :returns: The function result.
    """
    if hasattr(func, "py_func"): func = func.py_func

    return func(*args, **kwargs)


def run_numba(func: Callable[..., Any], *args: Any, verbose: bool = False, **kwargs: Any) -> Any:
    """
    First attempts to call the numba dispatcher in fully compiled (no-python) mode.

    If that fails it tries to run as a python function. Even if the function signature isn't
    supported in no-python mode, it can still provide performance benefits as the numba subroutines
    will be compiled separately.

    :param func: callable.
    :param args: Variable unnamed ordered args.
    :param verbose: Announce if the no-python dispatch failed for the
        function before running in python mode.
    :param kwargs: Variable named unordered kwargs.
    :returns: The function result.
    """
    try:
        return func(*args, **kwargs)
    except (nb_error.TypingError, nb_error.UnsupportedError):
        if verbose: print(f"Failed to run full-numba for {func.__name__}, attempting in python.")
        return run_py(func, *args, **kwargs)


### FORCED PARALLEL OR SYNC BLOCK


def _ov_pl_factory(
    sync_impl: Callable[..., Any], pl_impl: Callable[..., Any], ov_def: Callable[..., Any]
) -> Callable[..., Any]:
    """
    The method I use to be completely sure that there are two separate implementations for parallel bool.
    Register a numba overload that routes to ``sync_impl`` or ``pl_impl``
    according to the value (or literal value) of the boolean ``parallel``
    keyword that must be present in ``ov_def``'s signature.

    :param sync_impl: Synchronous implementation.
    :param pl_impl: Parallel implementation.
    :param ov_def: Python-only function to overload (must define ``parallel`` or ``pl`` keyword).
    :returns: The constructed overload function.
    """
    sig = inspect.signature(ov_def)
    params = list(sig.parameters.values())

    # ------------------------------------------------------------------
    # Build the textual version of the two signatures we need:
    #   1) the overload stub itself  – must exactly match *ov_def*
    #   2) the lambda we will return – same, but 'parallel' forced False
    # ------------------------------------------------------------------
    lp = None

    def _pstr(p):  # render *p* without annotations
        nonlocal lp
        base = p.name
        if base in ("parallel", "pl"): lp = base
        if p.kind is p.VAR_POSITIONAL: base = "*" + base
        elif p.kind is p.VAR_KEYWORD: base = "**" + base
        if p.default is not inspect._empty: base += "=" + repr(p.default)
        return base

    full_params = ", ".join(_pstr(p) for p in params)
    if lp is None: raise ValueError(f"Parallel overloads separator failed to find keyword for: def {ov_def.__name__}")
    call_args = ", ".join(p.name for p in params if p.name not in ("parallel", "pl"))
    lambda_paramlist = full_params

    code = f"""
@ovsi(ov_def)
def gener_ov({full_params}):
    #print('requesting')
    if isinstance({lp}, (bool, types.Literal)):
        dv = {lp} if isinstance({lp}, bool) else {lp}.literal_value
        if dv:
            #print("Is Parallel")
            return lambda {lambda_paramlist}: pl_impl({call_args})
        else:
            #print("Is Sync")
            return lambda {lambda_paramlist}: sync_impl({call_args})
    #print("Requesting Literal")
    return lambda {lambda_paramlist}: nb.literally({lp})

"""
    scope = dict(nb=nb, types=types, np=np, ovsi=ovsi, sync_impl=sync_impl, pl_impl=pl_impl, ov_def=ov_def)
    exec(textwrap.dedent(code), scope)
    return ov_def


def ir_force_separate_pl(
    sync_impl: Callable[..., Any], pl_impl: Callable[..., Any]
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    This is a *decorator factory* that generates an overloads to **force** compile in
    fully synchronous or parallel implementations based on the ``pl`` or ``parallel`` argument at
    compile time.

    In the current version numba still compiles in both the parallel and synchronous blocks even if
    the conditional branch that routes to either implementation is a constant/static value. Meaning that
    there will be the compilation bloat and decision scope that LLVM won't be able to optimize away, this is
    heavier on the system than a simple and lightweight single threaded complition would be.

    See :module:`nbux._rng` for use examples.

    This is a paragraph that contains `a link`_.

    .. _a link: https://domain.invalid/

    :param sync_impl: The (only) synchronous implementation of the numba function.
    :param pl_impl: The parallel implementation of the numba function.
    :returns: A decorator that constructs an overloads function around the python-only version.
    """
    return lambda ov_def: _ov_pl_factory(sync_impl, pl_impl, ov_def)


### INDEX LOWERING OPS - a form of implicit internal broadcast indexing for arrays (maybe tuples).


def l_1_0(x: np.ndarray | tuple[Any, ...], i1: int = 0) -> Any:
    if isinstance(x, (np.ndarray, tuple)) and len(x) >= 1: return x[i1]
    return x


@ovsic(l_1_0)
def _l_1_0(x, i1: int = 0):  # pragma: no cover
    # Same thing but d is manual, no literal cast so compilation can be a little quicker.
    def _impl(x, i1: int = 0): return x

    if isinstance(x, types.Array) and x.ndim >= 1:

        def _impl(x, i1: int = 0): return x[i1]

    return _impl


def l_1_1(x: np.ndarray | tuple[Any, ...], i1: int = 0) -> Any:
    if type(x) is np.ndarray and len(x.shape) >= 2: return x[i1]
    if isinstance(x, tuple) and len(x) >= 1 and isinstance(x[0], tuple): return x[i1]
    return x


@ovsic(l_1_1)
def _l_1_1(x, i1: int = 0):  # pragma: no cover
    def _impl(x, i1: int = 0): return x

    if isinstance(x, types.Array) and x.ndim >= 2:

        def _impl(x, i1: int = 0): return x[i1]

    return _impl


def l_1_2(x: np.ndarray | tuple[Any, ...], i1: int = 0) -> Any:
    if type(x) is np.ndarray and len(x.shape) >= 3: return x[i1]
    if isinstance(x, tuple) and len(x) >= 1 and isinstance(x[0], tuple) and isinstance(x[0][0], tuple): return x[i1]
    return x


@ovsic(l_1_2)
def _l_1_2(x, i1: int = 0):  # pragma: no cover
    def _impl(x, i1: int = 0): return x

    if isinstance(x, types.Array) and x.ndim >= 3:

        def _impl(x, i1: int = 0): return x[i1]

    return _impl


def l_12_0(x: np.ndarray | tuple[Any, ...], i1: int = 0, i2: int = 0) -> Any:
    if type(x) is np.ndarray:
        if len(x.shape) >= 2: return x[i1, i2]
        elif len(x.shape) == 1: return x[i1]
    if isinstance(x, tuple):
        if len(x) >= 1 and isinstance(x[0], tuple): return x[i1][i2]
        elif len(x) == 1: return x[i1]
    return x


@ovsic(l_12_0)
def _l_12_0(x, i1: int = 0, i2: int = 0):  # pragma: no cover
    # Same thing but d is manual, no literal cast so compilation can be a little quicker.
    def _impl(x, i1: int = 0, i2: int = 0): return x

    if isinstance(x, types.Array):
        if x.ndim >= 2:

            def _impl(x, i1: int = 0, i2: int = 0): return x[i1, i2]
        elif x.ndim == 1:

            def _impl(x, i1: int = 0, i2: int = 0): return x[i1]

    return _impl


def l_21_0(x: np.ndarray | tuple[Any, ...], i1: int = 0, i2: int = 0) -> Any:
    if type(x) is np.ndarray:
        if len(x.shape) >= 2: return x[i1, i2]
        elif len(x.shape) == 1: return x[i2]
    if isinstance(x, tuple):
        if len(x) >= 1 and isinstance(x[0], tuple): return x[i1][i2]
        elif len(x) == 1: return x[i2]
    return x


@ovsic(l_21_0)
def _l_21_0(x, i1: int = 0, i2: int = 0):  # pragma: no cover
    # Same thing but d is manual, no literal cast so compilation can be a little quicker.
    def _impl(x, i1: int = 0, i2: int = 0): return x

    if isinstance(x, types.Array):
        if x.ndim >= 2:

            def _impl(x, i1: int = 0, i2: int = 0): return x[i1, i2]
        elif x.ndim == 1:

            def _impl(x, i1: int = 0, i2: int = 0): return x[i2]

    return _impl


def l_12_d(x: np.ndarray | tuple[Any, ...], i1: int = 0, i2: int = 0, d: int = 0) -> Any:
    if isinstance(d, nb.types.Literal): d = d.literal_value
    if type(x) is np.ndarray:
        if len(x.shape) >= 2 + d: return x[i1, i2]
        elif len(x.shape) == 1 + d: return x[i1]
    if isinstance(x, tuple):
        if d == 0 and len(x) >= 1 and isinstance(x[0], tuple): return x[i1][i2]
        elif d == 0 and len(x) == 1: return x[i1]
    return x


_verbs = False


@ovsic(l_12_d)
def _l_12_d(x, i1: int = 0, i2: int = 0, d: int = 0):  # pragma: no cover
    def _impl(x, i1: int = 0, i2: int = 0, d: int = 0): return x

    if isinstance(x, types.Array):
        if isinstance(d, (nb.types.Literal, int)):
            dv = d if isinstance(d, int) else d.literal_value
            if _verbs: print("d is ", dv)
            if x.ndim >= 2 + dv:

                def _impl(x, i1: int = 0, i2: int = 0, d: int = 0): return x[i1, i2]
            elif x.ndim == 1 + dv:

                def _impl(x, i1: int = 0, i2: int = 0, d: int = 0): return x[i1]

            return _impl
        if _verbs: print("Requesting literal value for d")
        return lambda x, i1=0, i2=0, d=0: nb.literally(d)
    return _impl
