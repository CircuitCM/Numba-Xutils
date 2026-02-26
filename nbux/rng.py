# pyrefly: ignore-errors
# Array pointers are acting up
from __future__ import annotations

import math as mt
import random as rand

try:
    import aopt.calculations as calc
except ModuleNotFoundError:
    calc = None
import numba as nb
import numpy as np

import nbux.utils as nbu


@nbu.jt
def durstenfeld_p_shuffle(a: np.ndarray, k: int | None = None) -> None:
    """
    Perform up to k swaps of the Durstenfeld shuffle on array 'v'.
    Shuffling should still be unbiased even if a isn't changed back to sorted.

    :param a: Array to shuffle in-place.
    :param k: Number of swaps (defaults to ``a.shape[0] - 1``).
    :returns: None.
    """
    n = a.shape[0]
    num_swaps = n - 1 if k is None else k
    for i in range(num_swaps):
        j = rand.randrange(i, n)
        # Swap in-place
        tmp = a[i]
        a[i] = a[j]
        a[j] = tmp


##Legacy
@nbu.jtic
def jt_uniform_rng(a: float, b: float) -> float:
    """Return a single uniform random value between ``a`` and ``b``."""
    return rand.uniform(a, b)


##


@nbu.jtc
def _ss(f) -> None:
    rand.seed(f)
    np.random.seed(f)


def set_seed(seed: int | None) -> None:
    """Set both ``random`` and ``numpy.random`` seeds for both python and jit execution, from a python scope.  
     Or just jit execution from a jit scope.  
     
     Typically calling set_seed from python scope is best, only needs to be compiled into jit routines for specific needs."""
    if seed is not None:
        _ss(seed)
        rand.seed(seed)
        np.random.seed(seed)
        
@nbu.ovs(set_seed)
def impl_set_seed(seed: int | None):
    return _ss


_N = nbu.types.none

# no idk if uniform or int would be quicker here
scaled_rademacher_rng = nbu.rgic(lambda low=-1.0, high=1.0: low if rand.random() < 0.5 else high)


@nbu.rgic
def normal_rng_protect(mu: float = 0.0, sig: float = 1.0, pr: float = 0.001) -> float:
    """
    Draw a Gaussian sample and clamp values too close to zero.

    :param mu: Mean of the Gaussian draw.
    :param sig: Standard deviation of the Gaussian draw.
    :param pr: Protection ratio relative to ``sig`` used for the zero clamp.
    :returns: Gaussian sample with zero-protection applied.
    """
    n = rand.gauss(mu, sig)
    sr = sig * pr
    if abs(n) < sr: return mt.copysign(sr, n)
    return n


# itrs,ptr=nbu.buffer_nelems_andp(v) is for handling non-contiguous arrays in distributing random generation over
# threads. various lapack and blas calls are faster when ld{abc...} have extra space so that buffers align with SIMD
# operations. However the drawback is that (at least for now) these rngs will still generate into extra buffer space,
# overall this will still probably be faster than multiple subindexes on discontinuous parallel blocks. For very small
# arrays there might be barely noticeable overhead because of the itrs calculation, but in that case maybe direct calls
# to the rng stream would be better anyway. Also these implementations are subject to performance improvements eg
# through optimized and parallel mkl array streams in the future.
# NOTE: this whole module assumes true C or F ordering where C may have buffer gaps at the -1 idx and F can have buffer
# gaps at the 0 idx, otherwise there can't be any other bgaps.

# I implement with separate pl and sync functions so 1. the rng gens can be cached, and 2. with inline='always' in the
# override so that if parallel is a constant value, the pl or sync routine gets inlined alone for smaller asm profile.
# for now these are only f64 custom, implicit type casting used if arrays are smaller size, or you can see if manually
# setting config types changes signature of underlying rngs.

# For sync and parallel decorators, I go with
# Sync: jtic, because we still want the load boost when calling place_gauss from the interpreter (calling into it as
# pyfunc), but the benefits full compilation, which can only be assumed with inlining if we are caching the function
# already.
# Parallel: jtpc, again cache for python scope call. But we can also use the cached version for jitting scope, because
# the overhead of calling parallel cores will already be >> than calling into an external cfunc. However it might be the
# case that you lose control of setting parallel chunk size outside of this function scope, but haven't checked that.


### Gauss
@nbu.jtic
def _place_gauss_s(a: np.ndarray, mu: float = 0.0, sig: float = 0.0) -> None:
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in range(itrs): ptr[i] = rand.gauss(mu, sig)


@nbu.jtpc
def _place_gauss_pl(
    a: np.ndarray,
    mu: float = 0.0,
    sig: float = 0.0,
) -> None:  # pragma: no cover
    itrs, ptr = nbu.buffer_nelems_andp(a)
    # ld = nb.set_parallel_chunksize(mt.ceil(v.size / nb.get_num_threads()))
    for i in nb.prange(itrs): ptr[i] = rand.gauss(mu, sig)
    # nb.set_parallel_chunksize(ld)


# Only method I found to be certain the rng implements are compiling separately.
@nbu.ir_force_separate_pl(_place_gauss_s, _place_gauss_pl)
def place_gauss(a: np.ndarray, mu: float = 0.0, sigma: float = 1.0, parallel: bool = False) -> None:
    """
    Fill ``a`` in-place with Gaussian samples.

    :param a: Target array buffer.
    :param mu: Mean of the Gaussian distribution.
    :param sigma: Standard deviation of the Gaussian distribution.
    :param parallel: Use parallel implementation when true.
    :returns: None.
    """
    if parallel: _place_gauss_pl(a, mu, sigma)
    else: _place_gauss_s(a, mu, sigma)


# EXAMPLES
@nbu.jtc  # doesn't work, both the parallel and sync functions still compiled in. Seen in byte code.
def _place_gauss(v: np.ndarray, mu: float = 0.0, sigma: float = 1.0, parallel: bool = False) -> None:
    if nbu.force_const(parallel): _place_gauss_pl(v, mu, sigma)
    else: _place_gauss_s(v, mu, sigma)


@nbu.jtpc_s
def _2place_gauss(a, mu: float = 0.0, sigma: float = 1.0, parallel: bool = False) -> None:  # pragma: no cover
    # because of the overhead of the literal value request even after jitting, there is an extra ~80 ms calling
    # this from the python interpreter, so use place_gauss python-mode for py_func calls.
    place_gauss(a, mu, sigma, parallel)  # already compiled


@nbu.jtc
def _place_gauss_pl1(a, mu: float = 0.0, sigma: float = 1.0) -> None:  # pragma: no cover
    # makes it just as quick as original. No 80ms overhead after const is cached inside.
    place_gauss(a, mu, sigma, True)


## offshoot unbiased random orthogonal sample


# idk yet what this decorator should be
@nbu.rgc
def _random_orthogonals(a: np.ndarray, ortho_mem: tuple[np.ndarray, ...], parallel: bool = False) -> None:
    # note in the future for this to be cached, ortho_mem should be a single block of array memory, unpack the needed
    # memory later on the final interface.
    place_gauss(a, parallel=parallel)
    calc.orthnorm_f(a, *ortho_mem)


### Uniform  - defaults are set to ev=0, std = 1
@nbu.jtic  # abbreviated decorators
def place_uniform_s(a: np.ndarray, low: float = -(3.0**0.5), high: float = 3.0**0.5) -> None:
    """Synchronously fill ``a`` in-place with uniform samples on ``[low, high]``."""
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in range(itrs): ptr[i] = rand.uniform(low, high)


@nbu.jtpc
def place_uniform_pl(a: np.ndarray, low: float = -(3.0**0.5), high: float = 3.0**0.5) -> None:  # pragma: no cover
    """Parallel fill of ``a`` in-place with uniform samples on ``[low, high]``."""
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in nb.prange(itrs): ptr[i] = rand.uniform(low, high)


# A method I devised to force implementations to actually not include the compilation for the excluded bool method
@nbu.ir_force_separate_pl(place_uniform_s, place_uniform_pl)
def place_uniform(
    a: np.ndarray,
    low: float = -(3.0**0.5),
    high: float = 3.0**0.5,
    parallel: bool = False,
) -> None:
    """
    Fill ``a`` in-place with uniform samples on ``[low, high]``.

    :param a: Target array buffer.
    :param low: Lower bound of the uniform distribution.
    :param high: Upper bound of the uniform distribution.
    :param parallel: Use parallel implementation when true.
    :returns: None.
    """
    if parallel: place_uniform_pl(a, low, high)
    else: place_uniform_s(a, low, high)


### Rademacher.
@nbu.jtic
def _place_rademacher_s(a: np.ndarray, low: float = -1.0, high: float = 1.0) -> None:
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in range(itrs): ptr[i] = scaled_rademacher_rng(low, high)


@nbu.jtpc
def _place_rademacher_pl(a: np.ndarray, low: float = -1.0, high: float = 1.0) -> None:  # pragma: no cover
    itrs, ptr = nbu.buffer_nelems_andp(a)
    for i in range(itrs): ptr[i] = scaled_rademacher_rng(low, high)


@nbu.ir_force_separate_pl(_place_rademacher_s, _place_rademacher_pl)
def place_rademacher(a: np.ndarray, low: float = -1.0, high: float = 1.0, parallel: bool = False) -> None:
    """
    Fill ``a`` in-place with Rademacher-style two-point samples.

    :param a: Target array buffer.
    :param low: Lower point value.
    :param high: Upper point value.
    :param parallel: Use parallel implementation when true.
    :returns: None.
    """
    if parallel: _place_rademacher_pl(a, low, high)
    else: _place_rademacher_s(a, low, high)
