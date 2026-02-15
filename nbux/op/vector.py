"""
Numba Extensions and Utils Vector operations API. A module that holds an assortment of generic vector operations useful
for implementing performant numerical routines. It uses a naming convention that extends the axpy BLAS operator to other
kernels.

Array vectors: x, y, z. To qualify for this naming convention:
- x : Is always the write array and the first argument of the kernel.
- y : This will always be the optional second array.
- z : The optional third array.

Values: s1, s2, s3. In latex: $s_1, s_2, s_3$

Then we have char names:
- a : Add.
- p : Product.
- c : Copy. (x only).
- n : Negative. (Used infrequently, special for product -1, this is cosmetic b/c -1. * x == -x)
- d : Division. (not implemented for vector - vector at present, all kernels support (1/v) products).
- w : Power. (Not implemented yet, for whole scalars this can be replicated with v-v multi-products which are more
performant as well).

There's no subtract as it will likely compile down to the same thing as a -1. scalar product.
There's also no multi-scalar arithmetic, as that can be reduced to just one scalar externally.


Now the actual names, only applied when there is a return array. If isn't returned x, then we don't use this convention.
So:
    Choose one of the chars and x, examples:
    | ax* -> x += ...
    | cx* -> x = ...
    | px* -> x *= ...
    | nx -> x = -x

On the left hand side, the *scalar values* are unnamed and represent a left char op, we also include any other char
ops. Only vectors are named:
    | nxy -> x = -y
    | axpy -> x += s1*y
Three vector arrays are the most complex kernels we implement here, this allows us to build triads and may
provide a performance boost anywhere from 10-100% when applied correctly:
    | cxaxapy -> x = s1 + x + s2*y, but this is just | axapy -> x += s1 + s2*y
    | cxpyapz -> x = s1*y + s2*z, example triad it has around 2x throughput compared to x = s1*y; x += s2*z, when x
    and y are not contiguous. But surprisingly:
    | axpyapz -> x += s1*y + s2*z; can also have a 20-50% improvement in performance. Even though the reads are now
    the bottleneck. Considering reads vs writes are 3-1, and architecture is 2-1, optimal use can still gain 1.5x.

Why triads can be faster:
The Core registry is where data is loaded by the (two) read ports, operated on by the other ports, and written out by
the often singular write port. A port can handle (roughly) one AVX512 instruction set per core cycle, and all ports may
be used within that cycle. [For more info on
this](https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(client)#Scheduler_Ports_.26_Execution_Units).
This allows for best performance on accumulations x += y. But It's less known you can reference three independent blocks
of memory, two reads, one write (without read), and have zero instruction set bottlenecks.

Note: Llvm likely fully compile-reduces a kernel like cxapyapz when we set to unused kwargs s1=0., s2=1., s3=1.
So in the future we could eliminate some implementations but for visual reasons keeping their reduced versions around.
"""

from __future__ import annotations

import math as mt
from typing import Tuple

import numpy as np

import nbux.utils as nbu

# two dashes are old definitions or just unecessary.
# methods like x[:]=s, x[:]=y, x[:]+=s, x[:]+=y and other single-variable broadcasts should have no loss of performance
# by not being represented as a kernel. The rest of the kernels should have some memory or performance benefit from
# their broadcast
# counterparts.

# USER is responsible for selection of scalar floating value types.
# fix these


@nbu.jti
def dot(x: np.ndarray, y: np.ndarray) -> float:
    r"""Vector dot product: $v \leftarrow x^T y$"""
    n = x.shape[0]
    s = nbu.type_ref(x)(0.0)
    for i in range(n): s += x[i] * y[i]
    return s


@nbu.jti
def ndot(x: np.ndarray, y: np.ndarray) -> float:
    r"""Vector negate dot product: $v \leftarrow - x^T y$"""
    n = x.shape[0]
    s = nbu.type_ref(x)(0.0)
    for i in range(n): s -= x[i] * y[i]
    return s


@nbu.jti
def doti(x: np.ndarray) -> float:
    r"""Vector dot product with itself: $v \leftarrow x^T x$"""
    n = x.shape[0]
    s = nbu.type_ref(x)(0.0)
    for i in range(n): s += x[i] * x[i]
    return s


@nbu.jti
def tridot(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    r"""Vector dot product: $v \leftarrow x^T y,\; b \leftarrow x^T z$"""
    n = x.shape[0]
    s1 = s2 = nbu.type_ref(x)(0.0)
    for i in range(n):
        t = x[i]
        s1 += t * y[i]
        s2 += t * z[i]
    return s1, s2


@nbu.jti
def l2nm(x: np.ndarray) -> float:
    """L2 Euclidean vector norm shorthand."""
    return mt.sqrt(doti(x))


@nbu.jti
def cxy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""`x = y`, $x := y$."""
    n = y.shape[0]
    for i in range(n): x[i] = y[i]
    return x


@nbu.jti
def nxy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r"""`x = -y`, $x := -y$."""
    n = y.shape[0]
    for i in range(n): x[i] = -y[i]
    return x


@nbu.jti
def nx(x: np.ndarray) -> np.ndarray:
    r"""`x = -x`, $x := -x$."""
    n = x.shape[0]
    for i in range(n): x[i] = -x[i]
    return x


@nbu.jti
def cxpy(x: np.ndarray, y: np.ndarray, s1: float) -> np.ndarray:
    r"""`x = s1 * y`, $x := s_1 y$."""
    n = y.shape[0]
    s1 = nbu.type_ref(y)(s1)
    for i in range(n): x[i] = s1 * y[i]
    return x


@nbu.jti
def cxay(x: np.ndarray, y: np.ndarray, s1: float) -> np.ndarray:
    r"""`x = s1 + y`, $x := s_1 + y$."""
    n = y.shape[0]
    s1 = nbu.type_ref(y)(s1)
    for i in range(n): x[i] = s1 + y[i]
    return x


@nbu.jti
def cxapy(x: np.ndarray, y: np.ndarray, s1: float, s2: float) -> np.ndarray:
    r"""`x = s1 + s2 * y`, $x := s_1 + s_2 y$."""
    n = y.shape[0]
    typ = nbu.type_ref(y)
    s1, s2 = typ(s1), typ(s2)
    for i in range(n): x[i] = s1 + s2 * y[i]
    return x


@nbu.jti
def axpy(x: np.ndarray, y: np.ndarray, s1: float) -> np.ndarray:
    r"""`x += s1 * y`, $x := x + s_1 y$."""
    n = y.shape[0]
    s1 = nbu.type_ref(y)(s1)
    for i in range(n): x[i] += s1 * y[i]
    return x


@nbu.jti
def axay(x: np.ndarray, y: np.ndarray, s1: float) -> np.ndarray:
    r"""`x += s1 + y`, $x := x + s_1 + y$."""
    n = y.shape[0]
    s1 = nbu.type_ref(y)(s1)
    for i in range(n): x[i] += s1 + y[i]
    return x


@nbu.jti
def axapy(x: np.ndarray, y: np.ndarray, s1: float, s2: float) -> np.ndarray:
    r"""`x += s1 + s2 * y`, $x := x + s_1 + s_2 y$."""
    n = y.shape[0]
    typ = nbu.type_ref(y)
    s1, s2 = typ(s1), typ(s2)
    for i in range(n): x[i] += s1 + s2 * y[i]
    return x


@nbu.jti
def pxaxpy(x: np.ndarray, y: np.ndarray, s1: float, s2: float) -> np.ndarray:
    r"""`x = s1 * x + s2 * y`, $x := s_1 x + s_2 y$."""
    n = y.shape[0]
    typ = nbu.type_ref(y)
    s1, s2 = typ(s1), typ(s2)
    for i in range(n): x[i] = s1 * x[i] + s2 * y[i]
    return x


@nbu.jti
def pxaxy(x: np.ndarray, y: np.ndarray, s1: float) -> np.ndarray:
    r"""`x = s1 * x + y`, $x := s_1 x + y$."""
    n = y.shape[0]
    typ = nbu.type_ref(y)
    s1 = typ(s1)
    for i in range(n): x[i] = s1 * x[i] + y[i]
    return x


# NOTE turns out triads seem to perform better than two applications of two separate arrays.
# but greater than triads seems to not add significantly, unless array is tiny.
# YES because intel cpu's have two load ports and one store port
# https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(client)#Scheduler_Ports_.26_Execution_Units
@nbu.jti
def cxpyapz(x: np.ndarray, y: np.ndarray, z: np.ndarray, s1: float, s2: float) -> np.ndarray:
    r"""`x = s1 * y + s2 * z`, $x := s_1 y + s_2 z$."""
    n = y.shape[0]
    s1, s2 = nbu.type_ref(y)(s1), nbu.type_ref(z)(s2)
    for i in range(n): x[i] = s1 * y[i] + s2 * z[i]
    return x


# pxaxpy and cxpyapz should see a 2x performance boost from counterparts


# this is actual a 3 port load technically but still seems to improve in benchmarking.
@nbu.jti
def axpyapz(x: np.ndarray, y: np.ndarray, z: np.ndarray, s1: float, s2: float) -> np.ndarray:
    r"""`x += s1 * y + s2 * z`, $x := x + s_1 y + s_2 z$."""
    n = y.shape[0]
    s1, s2 = nbu.type_ref(y)(s1), nbu.type_ref(z)(s2)
    for i in range(n): x[i] += s1 * y[i] + s2 * z[i]
    return x


@nbu.jti
def cxpyaz(x: np.ndarray, y: np.ndarray, z: np.ndarray, s1: float) -> np.ndarray:
    r"""`x = s1 * y + z`, $x := s_1 y + z$."""
    n = y.shape[0]
    s1 = nbu.type_ref(y)(s1)
    for i in range(n): x[i] = s1 * y[i] + z[i]
    return x


@nbu.jti
def cxyaz(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    r"""`x = y + z`, $x := y + z$."""
    n = y.shape[0]
    for i in range(n): x[i] = y[i] + z[i]
    return x


# slight improvement here as well... which is strange.
@nbu.jti
def cxypz(x: np.ndarray, y: np.ndarray, z: np.ndarray, s1: float, s2: float) -> tuple[np.ndarray, np.ndarray]:
    r"""`x = s1 * z; y = s2 * z`, $x := s_1 z,\; y := s_2 z$."""
    n = z.shape[0]
    typ = nbu.type_ref(z)
    s1, s2 = typ(s1), typ(s2)
    for i in range(n):
        s = z[i]
        x[i] = s1 * s
        y[i] = s2 * s
    return x, y


@nbu.jti
def axypz(x: np.ndarray, y: np.ndarray, z: np.ndarray, s1: float, s2: float) -> tuple[np.ndarray, np.ndarray]:
    r"""`x += s1 * z; y += s2 * z`, $x := x + s_1 z,\; y := y + s_2 z$."""
    n = z.shape[0]
    typ = nbu.type_ref(z)
    s1, s2 = typ(s1), typ(s2)
    for i in range(n):
        s = z[i]
        x[i] += s1 * s
        y[i] += s2 * s
    return x, y


@nbu.jti
def vmax(x: np.ndarray) -> float:
    r"""Finds the maximum value in a vector: $v \leftarrow \max(x_i)$"""
    typ = nbu.type_ref(x)
    s = typ(nbu.prim_info(typ, 0))
    for e in x:
        if e > s: s = e
    return s


@nbu.jti
def vmin(x: np.ndarray) -> float:
    r"""Finds the minimum value in a vector: $v \leftarrow \min(x_i)$"""
    typ = nbu.type_ref(x)
    s = typ(nbu.prim_info(typ, 1))
    for e in x:
        if e < s: s = e
    return s


@nbu.jti
def vminmax(x: np.ndarray) -> tuple[float, float]:
    r"""Finds the minimum and maximum value in a vector: $\rightarrow \min(x_i),\max(x_i)$"""
    typ = nbu.type_ref(x)
    smin = typ(nbu.prim_info(typ, 1))
    smax = typ(nbu.prim_info(typ, 0))
    for e in x:
        if e < smin: smin = e
        if e > smax: smax = e
    return smin, smax


@nbu.jti
def argminmax(x: np.ndarray) -> tuple[float, float, int, int]:
    r"""Finds the minimum and maximum value in a vector: $\rightarrow \min_i(x_i),\max_j(x_j), i, j$"""
    typ = nbu.type_ref(x)
    smin = typ(nbu.prim_info(typ, 1))
    smax = typ(nbu.prim_info(typ, 0))
    i = -1
    j = -1
    for n in range(x.shape[0]):
        e = x[n]
        if e < smin:
            smin = e
            i = n
        if e > smax:
            smax = e
            j = n
    # we can
    return smin, smax, i, j


@nbu.jtic
def dtrace(x: np.ndarray) -> float:
    """Square Diagonal Trace"""
    t = nbu.type_ref(x)(0)
    for i in range(x.shape[0]): t += x[i, i]
    return t


@nbu.jtic
def dadd(x: np.ndarray, s: float) -> None:
    """Square diagonal Add."""
    for i in range(x.shape[0]): x[i, i] += s


@nbu.jtic
def dvadd(x: np.ndarray, y: np.ndarray) -> None:
    """Square diagonal vector Add."""
    for i in range(x.shape[0]): x[i, i] += y[i]


@nbu.jtic
def dmult(x: np.ndarray, s: float) -> None:
    """Square diagonal Multiply."""
    for i in range(x.shape[0]): x[i, i] *= s


@nbu.jtic
def dvmult(x: np.ndarray, y: np.ndarray) -> None:
    """Square diagonal vector Multiply."""
    for i in range(x.shape[0]): x[i, i] *= y[i]


@nbu.jti
def __argminmax_2k(x: np.ndarray):  # this is much slower than the simple 1 loop versions.
    """Two step calculation, this only costs 1.5n comparisons instead of 2n comparisons of the previous method.
    But in practice it's of course much slower than `argminmax`.
    """
    typ = nbu.type_ref(x)
    smin = typ(nbu.prim_info(typ, 1))  # +∞ sentinel
    smax = typ(nbu.prim_info(typ, 0))  # -∞ sentinel
    i = -1
    j = -1
    n = x.shape[0]

    # iterate in pairs, starting from 0 for numba-compatibility
    for k in range(0, n - 1, 2):
        a = x[k]
        b = x[k + 1]
        if a < b:
            if a < smin:
                smin = a
                i = k
            if b > smax:
                smax = b
                j = k + 1
        else:
            if b < smin:
                smin = b
                i = k + 1
            if a > smax:
                smax = a
                j = k

    # handle tail element if n is odd
    if n & 1:
        e = x[-1]
        if e < smin:
            smin = e
            i = n - 1
        elif e > smax:
            smax = e
            j = n - 1

    return smin, smax, i, j
