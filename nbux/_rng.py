import numba as nb
import numpy as np
import math as mt
import random as rand

from docutils.parsers.rst.roles import unimplemented_role
from numba import literal_unroll
from numba.misc.literal import literal_unroll_impl

import nbux._utils as nbu
import aopt.utils.configs as cfg
from aopt.calculations import jtc
from aopt.utils.numba_ext import eigh_inplace, pinv, innermul_cself, fo, outermul_cself, cholesky_fsolve_aplace, \
    cholesky_fsolve_inplace
from aopt.utils.numba import placerange, ri64, type_ref, prim_info, ri64s
import aopt.calculations as calc


@nbu.jt
def durstenfeld_p_shuffle(a, k=None):
    """
    Perform up to k swaps of the Durstenfeld shuffle on array 'a'.
    Shuffling should still be unbiased even if a isn't changed back to sorted.
    """
    n = a.shape[0]
    num_swaps = n-1 if k is None else k
    for i in range(num_swaps):
        j = rand.randrange(i,n)
        # Swap in-place
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp

##Legacy
@nbu.jtic
def jt_uniform_rng(a,b):
    return rand.uniform(a,b)

## 

@nbu.jtic
def _ss(f):
    rand.seed(f)
    np.random.seed(f)


def set_seed(seed):
    if seed is not None:
        _ss(seed)
        rand.seed(seed)
        np.random.seed(seed)

_N = nbu.types.none

#no idk if uniform or int would be quicker here
scaled_rademacher_rng=nbu.rgic(lambda l=-1.,u=1. : l if rand.random()<.5 else u)

@nbu.rgic
def normal_rng_protect(mu=0.,sig=1.,pr=.001):
    n=rand.gauss(mu,sig)
    sr=sig*pr
    if abs(n)<sr:
        return mt.copysign(sr,n)
    return n

# itrs,ptr=nbu.buffer_nelems_andp(a) is for handling non-contiguous arrays in distributing random generation over threads. various lapack and blas calls are faster when ld{abc...} have extra space so that buffers align with SIMD operations. However the drawback is that (at least for now) these rngs will still generate into extra buffer space, overall this will still probably be faster than multiple subindexes on discontinuous parallel blocks. For very small arrays there might be barely noticeable overhead because of the itrs calculation, but in that case maybe direct calls to the rng stream would be better anyway. Also these implementations are subject to performance improvements eg through optimized and parallel mkl array streams in the future.
#NOTE: this whole module assumes true C or F ordering where C may have buffer gaps at the -1 idx and F can have buffer gaps at the 0 idx, otherwise there can't be any other bgaps.

#I implement with separate pl and sync functions so 1. the rng gens can be cached, and 2. with inline='always' so that if parallel is a constant value, the pl or sync routine gets inlined alone for smaller asm profile.
#for now these are only f64 custom, implicit type casting used if arrays are smaller size, or you can see if manually setting config types changes signature of underlying rngs.

### Gauss
@nbu.jtic#ic
def place_gauss_s(a,mu=0.,sig=0.):
    itrs,ptr=nbu.buffer_nelems_andp(a)
    for i in range(itrs):ptr[i]=rand.gauss(mu,sig)

@nbu.jtpic
def place_gauss_pl(a,mu=0.,sig=0.,):
    itrs, ptr = nbu.buffer_nelems_andp(a)
    #ld = nb.set_parallel_chunksize(mt.ceil(a.size / nb.get_num_threads()))
    for i in nb.prange(itrs): ptr[i]=rand.gauss(mu,sig)
    #nb.set_parallel_chunksize(ld)
    
# @nbu.jtc #doesn't work both functions still compiled in.
# def place_gauss(a,mu=0.,sigma=1.,parallel=False):
#     if nbu.force_const(parallel): place_gauss_pl(a,mu,sigma)
#     else:place_gauss_s(a, mu, sigma)

#Only method I found to be certain the rng implements are compiling separately.
@nbu.ir_force_separate_pl(place_gauss_s,place_gauss_pl)
def place_gauss(a,mu=0.,sigma=1.,parallel=False):
    if parallel: place_gauss_pl(a,mu,sigma)
    else:place_gauss_s(a, mu, sigma)

#EXAMPLES
@nbu.jtpic_s
def _place_gauss(a,mu=0.,sigma=1.,parallel=False): 
    #because of the overhead of the literal value request even after jitting, there is an extra ~80 ms calling this from the python interpreter, so use place_gauss python-mode for py_func calls.
    place_gauss(a, mu, sigma, parallel) #already compiled
 
@nbu.jtc   
def _place_gauss_pl1(a,mu=0.,sigma=1.): #makes it just as quick as original. No 80ms overhead after const is cached inside.
    place_gauss(a, mu, sigma, True)
    
## offshoot unbiased random orthogonal sample

@nbu.rgpic
def random_orthogonals(a,ortho_mem,parallel=False):
    #note in the future for this to be cached, ortho_mem should be a single block of array memory, unpack the needed memory later on the final interface.
    place_gauss(a,parallel=parallel)
    calc.orthnorm_f(a,*ortho_mem)
    


### Uniform  - defaults are set to ev=0, std = 1  
@nbu.jtic #abbreviated decorators
def place_uniform_s(a,l=-(3.**.5),u=3.**.5):
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in range(itrs):ptr[i]=rand.uniform(l,u)

@nbu.jtpic
def place_uniform_pl(a,l=-(3.**.5),u=3.**.5):
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in nb.prange(itrs): ptr[i]=rand.uniform(l,u)

#a method I devised to force implementations to actually not include the compilation for the excluded bool method
@nbu.ir_force_separate_pl(place_uniform_s,place_uniform_pl)
def place_uniform(a, l=-(3.**.5),u=3.**.5, parallel=False):
    if parallel: place_uniform_pl(a,l,u)
    else:place_uniform_s(a,l,u)

### Gauss with 0 protection.
@nbu.jtic
def place_gauss_no0_s(a,mu=0.,sig=0.):
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in range(itrs):ptr[i]=rand.gauss(mu,sig)

@nbu.jtpic
def place_gauss_no0_pl(a,mu=0.,sig=0.,):
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in nb.prange(itrs): ptr[i]=rand.gauss(mu,sig)

@nbu.ir_force_separate_pl(place_gauss_no0_s,place_gauss_no0_pl)
def place_gauss_no0(a,mu=0.,sigma=1.,parallel=False):pass

### Rademacher.
@nbu.jtic
def place_rademacher_s(a, l=-1., u=1.):
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in range(itrs):ptr[i]=scaled_rademacher_rng(l, u)

@nbu.jtpic
def place_rademacher_pl(a,  l=-1., u=1.):
    itrs, ptr = nbu.buffer_nelems_andp(a) 
    for i in range(itrs):ptr[i]=scaled_rademacher_rng(l, u)

@nbu.ir_force_separate_pl(place_rademacher_s,place_rademacher_pl)
def place_rademacher(a,l=-1.,u=1.,parallel=False):
    if parallel: place_rademacher_pl(a,l,u)
    else:place_rademacher_s(a,l,u)
