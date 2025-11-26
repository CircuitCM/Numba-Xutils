from typing import Sequence, Callable

from numba.np.numpy_support import as_dtype

import aopt.utils.configs as cfg
import numpy as np
import numba as nb
from numba import types
from numba.extending import intrinsic,register_jitable,overload
from numba.core import cgutils
import warnings
from functools import wraps
import os
_N = types.none
from numba.core.errors import NumbaPerformanceWarning
def njit_no_parallelperf_warn(**njit_kwargs):
    """
    Drop‑in replacement for @nb.njit(parallel=True) that suppresses the
    ‘parallel=True but no parallel transform’ warning **for this function only**.
    """
    jt=nb.njit(**njit_kwargs)
    def decorator(func):
        name = func.__name__ #might have to become __qualname__
        pattern = f"(?s)(?=.*parallel=True)(?=.*{name}).*"
        warnings.filterwarnings("ignore",category = NumbaPerformanceWarning,message  = pattern)
        return jt(func)
    return decorator

def rg_no_parallelperf_warn(rrg):
    def decorator(func):
        name = func.__name__
        pattern = f"(?s)(?=.*parallel=True)(?=.*{name}).*"
        warnings.filterwarnings("ignore",category = NumbaPerformanceWarning,message  = pattern)
        return rrg(func)
    return decorator

#This only changes once at import time.
#--- Numba Global Fastmath : you can frequently get a 2x performance boost, for virtually no loss in error range (only 4x larger than 2**-{# your bit precision range})
_fm=os.environ.get('NB_GFASTM','true')
_fm = False if not _fm or _fm.lower() in 'false' else eval(_fm) if any(i in _fm for i in ('[','{','(')) else True
#--- Numba Global Error Model : 'numpy'|'python' will do less checks but frequently has much faster jitted code, because it isn't forced to fall back to larger data types, [see this article](). 
_erm=os.environ.get('NB_GERRMOD','numpy')

"""
Before continuation some notes on this section:

## Configurations
s|p : Sync or Parallel, the threading strategy.
c : Cache the compilation for new signatures.
n : nogil, does numba bypass the global interpreter lock of python (array writes are not thread safe!).
i : Manual/forced Numba-IR level inline. Convenient if types are causing compilations to fail, sometimes to force inlining of low level operations if LLVM won't do it.

## Decorators
jt - Numba jit using the base defaults and extension characters seen above.
rg - Register Jittable, these functions will compile into the Numba IR but run as python when called from the interpreter. This can actually improve performance significantly for the python function as Numba functions have a high overhead cost to call. you can call a jitted functions py func by `jitfunc.py_func(*args, **kwargs)`.
ov - Override decorators. See the numba docs for coverage on this.

From my past experience, these behaviors might be true, but need to be verified from source:
- Cache:
    - Great for reducing initial startup times by not needing to recompile the entire library. This even significantly benefits compilation speeds of functions that can't be cached but utilize cached functions.
    - If your procedure is compiling with reference to an already cached function, it will **always** treat it as a function pointer and not benefit from LLVM's unified compilations. Therefore you shouldn't cache small external functions, caching should be at the scope of larger procedures.
- Inline:
    - Inline decorated functions will make the first compile slower. If you want to be very certain the procedure is being compiled as one scope, by sacrificing a little compilation speed, this inline option will help. Usually Inline doesn't do anything.
- Nogil:
    - GIL bypass seems to occur before execution of the compiled source. Meaning that any nogil decorations on inner linked functions will have no effect. For Numba library developers it means that you only need to wrap the outermost scope that you expect users to call in to.
- Parallel:
    - Will **not** cache if you modify the block size of dispatch. Normally leaving it on automatic (0) works well for large vectorized array operations, however if you use it more like a dispatch or work queue you may want significantly smaller blocks (optimal for asymmetric or long running tasks). You could change block size outside of the parallel block though. It also won't cache if you request thread scope IDs.
    - Will run parallel in scope. So if you call a @jtp function from within a @jt function you will get parallel benefits, but expecting parallel dispatch from a @jt function within @jtp block is not applicable.
    - Using a parallel decorator but a branch to decide if the block should be run parallel or not, will **always** compile in the parallel and sync branch, even if the branch comes from a constant. If you want the low overhead benefits of the sync branch alone, then you need to define them as different functions, they could then be tied in with a custom overloads. To make this situation generic or at least less boilerplate to implement, I developed a hacky approach below.
    - If you have multiple layers of `prange` only the outer one will be parallelized, **unless** all inner ranges are the same, (see here)[]. Example:
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

_dft=dict(fastmath=_fm, error_model=_erm) #base python arguments.
jit_s=_dft
jit_sn=jit_s|dict(nogil=True)
jit_sc=jit_s|dict(cache=True)
jit_scn=jit_sc|dict(nogil=True)
jit_si=jit_s|dict(inline='always')
jit_sci=jit_si|dict(cache=True)
jit_scin=jit_sci|dict(nogil=True)

jit_p= _dft | dict(parallel=True)
jit_pn=jit_p|dict(nogil=True)
jit_pc=jit_p|dict(cache=True)
jit_pcn=jit_pc|dict(nogil=True)
jit_pi=jit_p|dict(inline='always')
jit_pci=jit_pi|dict(cache=True)

# --- JIT DECORATORS
jt=nb.njit(**jit_s) #plain jit
jtc=nb.njit(**jit_sc) #cache
jtn=nb.njit(**jit_sn) #nogil
jtnc=nb.njit(**jit_scn) #nogil and cache
jti=nb.njit(**jit_si) #inline
jtic=nb.njit(**jit_sci) #inline and cache - might have no effect as inline forces code injection.
#there is barely a concievable need for both inline and nogil in the same decorator.
#jtinc=nb.njit(**jit_scin)

jtp=nb.njit(**jit_p) #jit parallel.
jtp_s=njit_no_parallelperf_warn(**jit_p) #jit parallel - silence no parallel warning.
jtpc=nb.njit(**jit_pc)
jtpc_s=njit_no_parallelperf_warn(**jit_pc)
#we might want parallel nogil if we eg run less python threads than number of cores and we limit thread dispatch in the parallel block
jtpn=nb.njit(**jit_pn)
jtpn_s=njit_no_parallelperf_warn(**jit_pn)
jtpnc=nb.njit(**jit_pcn)
jtpnc_s=njit_no_parallelperf_warn(**jit_pcn)
# Parallel + inline are not relevant together - inline will push the code the the next function scope, this scope will need a non-inline parallel to run the prange
# jtpi=nb.njit(**jit_pi)
# jtpi_s=njit_no_parallelperf_warn(**jit_pi)
# jtpic=nb.njit(**jit_pci)
# jtpic_s=njit_no_parallelperf_warn(**jit_pci)

# --- REGISTER JITTABLE DECORATORS
_rg=register_jitable
rg=_rg(**jit_s) #base sync
rgc=_rg(**jit_sc) #cache
#in theory nogil is never utilized with register jittable, if you call rg from python scope it will run the py func.
# rgn=_rg(**jit_sn) #nogil
# rgnc=_rg(**jit_scn) #nogil cache
rgi=_rg(**jit_si) #Inline
rgic=_rg(**jit_sci) #inline cachce

rgp=_rg(**jit_p)
rgp_s=rg_no_parallelperf_warn(rgp)
rgpc=_rg(**jit_pc)
rgpc_s=rg_no_parallelperf_warn(rgpc)

# --- OVERLOADS DECORATORS
#I'm pretty sure caching is redundant for overloads, but assuming not and including.
ovs=lambda impl: overload(impl, jit_options=jit_s)
ovsi=lambda impl: overload(impl,jit_options=jit_s,inline='always')
ovsc=lambda impl: overload(impl,jit_options=jit_sc)
ovsic=lambda impl: overload(impl,jit_options=jit_sc,inline='always')
#It's also possible parallel is never directly utilized by overloads.
ovp=lambda impl: overload(impl,jit_options=jit_p)
ovpc=lambda impl: overload(impl,jit_options=jit_pc)


#shorthand
fb_=np.frombuffer

#See if you can make this more performant in the future
@rgi
def ffb_(buffr,sidx,eidx,dtype):
    typ=type_ref(buffr)
    ogo=prim_info(typ,3)
    ogn=prim_info(dtype,3)
    tm=ogn//ogo

def compiletime_parallelswitch():
    pass

@intrinsic
def stack_empty_impl(typingctx,size,dtype):
    def impl(context, builder, signature, args):
        ty=context.get_value_type(dtype.dtype)
        ptr = cgutils.alloca_once(builder, ty,size=args[0])
        return ptr

    sig = types.CPointer(dtype.dtype)(types.int64,dtype) #int64 is the os level pointer dtype, may need to change
    return sig, impl

def stack_empty(size,shape,dtype):
    return np.empty(shape,dtype=dtype)

@jtc
def stack_empty_(size, shape, dtype):
    #From: https://github.com/numba/numba/issues/5084#issue-550324913
    #Forces small stack allocated array. maybe 2x quicker than naive np.array allocation.
    """
    Size (int) must be a fixed at compile time.
    It is not possible to change it during execution.

    The shape (tuple) size can be as large as the size or smaller.
    This can be dynamically changed during execution.

    The datatype also have to be fixed in this implementation.

    The carray can't be returned from a function.
    """
    arr_ptr=stack_empty_impl(size,dtype)
    arr=nb.carray(arr_ptr,shape)
    return arr

@overload(stack_empty, **cfg.jit_s,cache=True)
def _stack_empty(size,shape,dtype):
    #same as above, but now calling stack_empty in a python area will just return a normal c-array with shape and dtype.
    return lambda size, shape, dtype: stack_empty_(size, shape, dtype)

#can probably use this to replace the self archive tracking for de-bpesta mutation.
@intrinsic
def nb_val_ptr(typingctx, data):
    def impl(context, builder, signature, args):
        ptr = cgutils.alloca_once_value(builder,args[0])
        return ptr
    sig = types.CPointer(data)(data)
    return sig, impl

@intrinsic
def nb_ptr_val(typingctx, data):
    def impl(context, builder, signature, args):
        val = builder.load(args[0])
        return val
    sig = data.dtype(data)
    return sig, impl

@intrinsic
def nb_array_ptr(typingctx, arr_typ):
    elptr = types.CPointer(arr_typ.dtype)
    def codegen(ctx, builder, sig, args):
        return ctx.make_helper(builder, sig.args[0], args[0]).data
    return elptr(arr_typ), codegen

@jtic 
def buffer_nelems_andp(arr):
    """This is basically a method to get the gaps in a discontinuous array. It's a method to return the contiguous backing array to be used for iterations where mutation of the discontuous portion isn't actually relevant but can improve total loop performance."""
    ptr=nb_array_ptr(arr)
    size = arr.size
    if arr.ndim<2:return size,ptr
    # fast C vs F test via stride magnitudes
    s1,s2=(-1,-2) if abs(arr.strides[0]) >= abs(arr.strides[-1]) else (0,1) #we use this to detect if C or F ordered
    tail_prod = size // arr.shape[s1]
    first_len = abs(arr.strides[s2]) // abs(arr.strides[s1])
    nelems = tail_prod * first_len
    #There is so much less implementation code than np.frombuffer in numba, because it's assumed the entire thing is contiguous
    return nelems,ptr


#WARNING fastmath can produce different results.
#The only one that is a little faster than np.round is the int32
def ri64(rd:float): return int(rd + .5)
ri32=ri64
ovsic(ri64)(lambda rd: (lambda rd: nb.int64(rd + .5)))
ovsic(ri32)(lambda rd: (lambda rd: nb.int32(rd + nb.float32(.5))))

def ri64s(rd:float,s): return int((rd/s) + .5)*s
ri32s=ri64s
overload(ri64s, **cfg.jit_s,cache=True)(lambda rd,s: (lambda rd,s: nb.int64((rd/s) + .5)*s))
overload(ri32s, **cfg.jit_s,cache=True)(lambda rd,s: (lambda rd,s: nb.int32((rd/s) + nb.float32(.5))*s))



@rgic
def display_round(f,m=1,s=10):
    """
    A method to round floats so that when printed from numba's stdout it doesn't show roundoff tail error.
    
    Note: this still doesn't work, looking for something that does.
    """
    # so that we don't get roundoff tails and mess up the string, we  instead float -> int -> float it.
    # mv and s are integers that represent base 10 factors, mv is b10 order offset and s is significant digits, s=0 no digits, mv=1 no offset.
    m10 = 10. ** (m - 1)
    s10= 10. ** s
    return np.float64(np.int64(f * s10*m10 + .5)) / s10
    

def type_ref(arg):
    #print(idxr.dtype,idxr.dtype.type)
    if isinstance(arg,np.ndarray): return arg.dtype.type
    else: return type(arg)

@ovsic(type_ref)
def _type_ref(arg):
    #now supports primitive types, literals and array memory type.
    if isinstance(arg,types.Literal): #we should never even get this result
        typ=arg._literal_type_cache
        return lambda arg: typ
    elif isinstance(arg,types.Array):
        typ = arg.dtype
        return lambda arg: typ
    else:
        typ=arg #it only sees type in this scope not value
        return lambda arg: typ

def if_val_cast(typ,val):
    if isinstance(val,(np.ndarray,Sequence)):
        return val
    else:
        return typ(val)

@ovsic(if_val_cast)
def _if_val_cast(typ,val):
    if isinstance(val,types.IterableType):return lambda typ,val:val
    else:return lambda typ,val:typ(val)
    
def op_call(cal:Callable|tuple,_def=True):
    #_def is for default value
    if isinstance(cal,Callable):
        return cal()
    elif isinstance(cal,tuple|list):
        if isinstance(cal[0],Callable):
            return cal[0](*cal[1:])
    if _def is not None:
        return _def
    return cal 

@ovsic(op_call)
def _op_call(cal,_def=True):
    if isinstance(cal,types.Callable):
        return lambda cal,_def=True: cal()
    elif isinstance(cal,types.BaseTuple|types.LiteralList):
        if isinstance(cal[0],types.Callable):
            return lambda cal,_def=True: cal[0](*cal[1:])
    if _def is not _N:
        return lambda cal,_def=True: _def
    return lambda cal, _def=True: cal


def op_call_args(cal,args):
    ct=isinstance(cal,Callable) #otherwise tuple|list
    rt=isinstance(args,tuple|list) #otherwise single element.
    if ct and rt:
        return cal(*args)
    if ct and not rt:
        return cal(args)
    if not ct and rt:
        return cal[0](*args,*cal[1:])
    #if not ct and not rt:
    return cal[0](args,*cal[1:])

@ovsic(op_call_args)
def _op_call_args(cal,args):
    ct=isinstance(cal,types.Callable) #otherwise tuple|list
    rt=isinstance(args,types.BaseTuple|types.LiteralList) #otherwise single element.
    #print('Here',ct,rt)
    if ct and rt:
        return lambda cal,args: cal(*args)
    if ct and (not rt):
        return lambda cal,args: cal(args)
    if (not ct) and rt:
        return lambda cal,args: cal[0](*args,*cal[1:])
    #if not ct and not rt:
    return lambda cal,args: cal[0](args,*cal[1:])


def prim_info(dt, field):
    """
    Return type-specific info for NumPy type `dt`, given integer field selector.
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
    """
    if not hasattr(dt, 'kind'):
        dt = np.dtype(dt)

    match (dt.kind, field):
        # Integer & unsigned integer
        case ('i' | 'u', 0): return np.iinfo(dt).min
        case ('i' | 'u', 1): return np.iinfo(dt).max
        # Floating point
        case ('f', 0): return np.finfo(dt).min
        case ('f', 1): return np.finfo(dt).max
        case ('f', 2): return np.finfo(dt).eps
        # Boolean
        case ('b', 0): return False
        case ('b', 1): return True
        # Complex types
        case ('c', 0): return complex(np.finfo(dt).min, np.finfo(dt).min)
        case ('c', 1): return complex(np.finfo(dt).max, np.finfo(dt).max)
        case ('c', 2): return np.finfo(dt).eps
        # Universal: byte size
        case (_, 3): return dt.itemsize
        # Fallback
        case _: return None

np_tinfo=prim_info

@ovsic(prim_info)
def _prim_info(typ,res):
    """
    Primitives info.
    
    :param typ: type received from a function like type_ref in a nopython block.
    :param res: 0 min, 1 max, 2 epsilon/precision.
    :return: 
    """
    if isinstance(res,(nb.types.Literal,int)):
        ref=res if type(res) is int else res.literal_value
        tpref=as_dtype(typ)
        #tpref=np.dtype(str(typ.dtype)) #gives us the underlying primitive type. maybe use numbas np_support func tho.
        infoval=np_tinfo(tpref,ref) #where we query 
        return lambda typ,res: infoval
    return lambda typ,res:nb.literally(res)

@jti
def placerange(r,start=0,step=1):
    for i in range(r.shape[0]):
        r[i] = start + i*step



@rgi
def swap(x,i,j):
    t=x[i]
    x[i]=x[j]
    x[j]=t
    

def force_const(val): return val
@ovs(force_const)
def _force_const(val):
    if isinstance(val,types.Literal):
        #tv=val.literal_value
        #print('const',tv)
        return lambda val: val
    else:
        return lambda val: nb.literally(val)


def run(func,*args,**kwargs):
    return func(*args,**kwargs)

def run_py(func,*args,**kwargs):
    if hasattr(func,'py_func'):
        func = func.py_func
    run(func,*args,**kwargs)

def run_fallback(func,*args,verbose=False,**kwargs):
    try:
        return run(func,*args,**kwargs)
    except:
        if verbose:
            print('Failed to run numba mode, attempting python.')
        return run_py(func,*args,**kwargs)

### FORCED PARALLEL OR SYNC BLOCK
import inspect
import textwrap

def _ov_pl_factory(sync_impl, pl_impl, ov_def):
    """
    The method I use to be completely sure that there are two separate implementations for parallel bool.
    Register a numba‑overload that routes to *sync_impl* or *pl_impl* according
    to the value (or Literal value) of the boolean *parallel* keyword that must
    be present in *ov_def*’s signature.
    """
    sig        = inspect.signature(ov_def)
    params     = list(sig.parameters.values())

    # ------------------------------------------------------------------
    # Build the textual version of the two signatures we need:
    #   1) the overload stub itself  – must exactly match *ov_def*
    #   2) the lambda we will return – same, but 'parallel' forced False
    # ------------------------------------------------------------------
    lp=None
    def _pstr(p):               # render *p* without annotations
        nonlocal lp
        base = p.name
        if base in ('parallel', 'pl'):
            lp=base
        if p.kind is p.VAR_POSITIONAL: base = "*"+base
        elif p.kind is p.VAR_KEYWORD:  base = "**"+base
        if p.default is not inspect._empty:
            base += "=" + repr(p.default)
        return base

    full_params      = ", ".join(_pstr(p)            for p in params)
    if lp is None: raise ValueError(f'Parallel overloads separator failed to find keyword for: def {ov_def.__name__}')
    call_args        = ", ".join(p.name              for p in params
                                 if p.name not in ("parallel",'pl'))
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
    scope = dict(nb=nb, types=types,np=np,ovsi=ovsi,
                 sync_impl=sync_impl, pl_impl=pl_impl,ov_def=ov_def)
    exec(textwrap.dedent(code), scope)
    return ov_def

def ir_force_separate_pl(sync_impl, pl_impl):
    return lambda ov_def: _ov_pl_factory(sync_impl,pl_impl,ov_def)
 
### INDEX LOWERING OPS - a form of implicit internal broadcast indexing for arrays.

def l_1_0(x, i1=0):
    if type(x) is np.ndarray and len(x.shape)>=1: return x[i1]
    return x

@ovsic(l_1_0)
def _l_1_0(x, i1=0):
    #Same thing but d is manual, no literal cast so compilation can be a little quicker.
    def _impl(x,i1=0): return x
    if isinstance(x, types.Array) and x.ndim >= 1: 
            def _impl(x,i1=0): return  x[i1]
    return _impl

def l_1_1(x, i1=0):
    if type(x) is np.ndarray and len(x.shape)>=2: return x[i1]
    return x

@ovsic(l_1_1)
def _l_1_1(x, i1=0):
    def _impl(x,i1=0): return x
    if isinstance(x, types.Array) and x.ndim >= 2: 
            def _impl(x,i1=0): return  x[i1]
    return _impl

def l_1_2(x, i1=0):
    if type(x) is np.ndarray and len(x.shape)>=3: return x[i1]
    return x

@ovsic(l_1_2)
def _l_1_2(x, i1=0):
    def _impl(x,i1=0): return x
    if isinstance(x, types.Array) and x.ndim >= 3: 
            def _impl(x,i1=0): return  x[i1]
    return _impl


def l_12_0(x, i1=0, i2=0):
    if type(x) is np.ndarray:
        if len(x.shape)>=2: return x[i1, i2]
        elif len(x.shape)==1: return x[i1]
    return x

@ovsic(l_12_0)
def _l_12_0(x, i1=0, i2=0):
    #Same thing but d is manual, no literal cast so compilation can be a little quicker.
    def _impl(x,i1=0, i2=0):
        return x
    if isinstance(x, types.Array):
        if x.ndim>=2:
            def _impl(x,i1=0, i2=0):return x[i1, i2]
        elif x.ndim == 1: 
            def _impl(x,i1=0, i2=0): return  x[i1]
    return _impl

def l_21_0(x, i1=0, i2=0):
    if type(x) is np.ndarray:
        if len(x.shape)>=2: return x[i1, i2]
        elif len(x.shape)==1: return x[i2]
    return x

@ovsic(l_21_0)
def _l_21_0(x, i1=0, i2=0):
    #Same thing but d is manual, no literal cast so compilation can be a little quicker.
    def _impl(x,i1=0, i2=0):
        return x
    if isinstance(x, types.Array):
        if x.ndim>=2:
            def _impl(x,i1=0, i2=0):return x[i1, i2]
        elif x.ndim == 1: 
            def _impl(x,i1=0, i2=0): return  x[i2]
    return _impl

def l_12_d(x, i1=0, i2=0,d=0):
    if isinstance(d,nb.types.Literal):
        d=d.literal_value
    if type(x) is np.ndarray:
        if len(x.shape)>=2+d: return x[i1, i2]
        elif len(x.shape)==1+d: return x[i1]
    return x

_verbs=False 

@ovsic(l_12_d)
def _l_12_d(x, i1=0, i2=0,d=0):
    def _impl(x,i1=0, i2=0,d=0):return x
    if isinstance(x, types.Array):
        if isinstance(d,(nb.types.Literal,int)):
            dv=d if type(d) is int else d.literal_value
            if _verbs:
                print('d is ',dv)
            if x.ndim >= 2+dv: 
                def _impl(x,i1=0, i2=0,d=0): return  x[i1,i2]
            elif x.ndim == 1+dv: 
                def _impl(x,i1=0, i2=0,d=0): return  x[i1]
            return _impl
        if _verbs:
            print('Requesting literal value for d')
        return lambda x,i1=0, i2=0,d=0: nb.literally(d)
    return _impl
