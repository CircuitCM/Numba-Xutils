import aopt.utils.numba as nbu
import numpy as np
import math as mt


# two dashes are old definitions or just unecessary.
# methods like dst[:]=a, dst[:]=src, dst[:]+=a, dst[:]+=src and other single variable broadcasts should have no loss of performance
# by not being represented as a kernel. The rest of the kernels should have some memory or performance benefit from their broadcast
# counterparts.

# USER is responsible for selection of scalar floating value types.

@nbu.jti
def dot(x: np.ndarray, y: np.ndarray) -> float:
    sr"""Vector dot product: $v \leftarrow x^T y$"""
    n = x.shape[0]
    v = nbu.type_ref(x)(0.0)
    for i in range(n):
        v += x[i] * y[i]
    return v

@nbu.jti
def ndot(x: np.ndarray, y: np.ndarray) -> float:
    sr"""Vector negate dot product: $v \leftarrow - x^T y$"""
    n = x.shape[0]
    v = nbu.type_ref(x)(0.0)
    for i in range(n):
        v -= x[i] * y[i]
    return v


@nbu.jti
def doti(x: np.ndarray) -> float:
    sr"""Vector dot product with itself: $v \leftarrow x^T x$"""
    n = x.shape[0]
    v = nbu.type_ref(x)(0.0)
    for i in range(n):
        v += x[i] * x[i]
    return v

@nbu.jti
def tridot(x: np.ndarray, y: np.ndarray,z: np.ndarray) -> (float,float):
    sr"""Vector dot product: $a \leftarrow x^T y,\; b \leftarrow x^T z$"""
    n = x.shape[0]
    v1=v2 = nbu.type_ref(x)(0.0)
    for i in range(n):
        t=x[i]
        v1 += t * y[i]
        v2 += t * z[i]
    return v1,v2

@nbu.jti
def l2nm(x: np.ndarray) -> float:return mt.sqrt(doti(x)) 


@nbu.jti
def __cxy(dst: np.ndarray, src: np.ndarray, ):
    sr"""Copy vector: $x \leftarrow y$"""
    n = src.shape[0]
    for i in range(n):
        dst[i] = src[i]
    return dst


@nbu.jti
def cxny(dst: np.ndarray, src: np.ndarray, ):
    sr"""Copy negative y to x: $x \leftarrow -y$"""
    n = src.shape[0]
    for i in range(n):
        dst[i] = -src[i]
    return dst


@nbu.jti
def nx(dst: np.ndarray):
    sr"""Negate self (in-place): $x \leftarrow -x$"""
    n = dst.shape[0]
    for i in range(n):
        dst[i] = -dst[i]
    return dst


@nbu.jti
def cxpy(dst: np.ndarray, v1: float, src: np.ndarray):
    sr"""Copy value product y to x: $x \leftarrow v_1 \cdot y$"""
    n = src.shape[0]
    v1 = nbu.type_ref(src)(v1)
    for i in range(n):
        dst[i] = v1 * src[i]
    return dst


@nbu.jti
def cxay(dst: np.ndarray, v1: float, src):
    sr"""Copy value added y to x: $x \leftarrow v_1 + y$ """
    n = src.shape[0]
    v1 = nbu.type_ref(src)(v1)
    for i in range(n):
        dst[i] = v1 + src[i]
    return dst


@nbu.jti
def cxapy(dst: np.ndarray, v1: float, v2: float, src):
    sr"""Copy product of y value added, to x: $x \leftarrow a + b \cdot y$"""
    n = src.shape[0]
    typ = nbu.type_ref(src)
    v1, v2 = typ(v1), typ(v2)
    for i in range(n):
        dst[i] = v1 + v2 * src[i]
    return dst


@nbu.jti
def axpy(dst: np.ndarray, a: float, src: np.ndarray):
    sr"""Add product of y to x: $x \leftarrow a \cdot y + x$"""
    n = src.shape[0]
    a = nbu.type_ref(src)(a)
    for i in range(n):
        dst[i] += a * src[i]
    return dst


@nbu.jti
def axay(dst: np.ndarray, v1: float, src: np.ndarray):
    sr"""Add value added y to x: $x \leftarrow (a + y) + x$"""
    n = src.shape[0]
    v1 = nbu.type_ref(src)(v1)
    for i in range(n):
        dst[i] += v1 + src[i]
    return dst


@nbu.jti
def axapy(dst: np.ndarray, v1: float, v2: float, src: np.ndarray):
    sr"""Add product of y value added, to x: $x \leftarrow (a + b \cdot y) + x$"""
    n = src.shape[0]
    typ = nbu.type_ref(src)
    v1, v2 = typ(v1), typ(v2)
    for i in range(n):
        dst[i] += v1 + v2 * src[i]
    return dst

@nbu.jti
def pxaxpy(dst: np.ndarray, v1: float, v2: float, src: np.ndarray):
    sr"""Value product to x add value product of y to x: $x \leftarrow v_1 \cdot x + v_2 \cdot y$"""
    n = src.shape[0]
    typ = nbu.type_ref(src)
    v1, v2 = typ(v1), typ(v2)
    for i in range(n):
        dst[i] = v1*dst[i] + v2*src[i] #not a copy technically would be px
    return dst

#NOTE turns out triads seem to perform better than two applications of two separate arrays.
#but greater than triads seems to not add significantly, unless array is tiny.
#YES because intel cpu's have two load ports and one store port
#https://en.wikichip.org/wiki/intel/microarchitectures/skylake_(client)#Scheduler_Ports_.26_Execution_Units
@nbu.jti
def cxapypz(dst: np.ndarray, v1: float, v2: float, src1: np.ndarray, src2: np.ndarray):
    sr"""Copy product y add product z: $x \leftarrow v_1 \cdot x + v_2 \cdot y$"""
    n = src1.shape[0]
    v1, v2 = nbu.type_ref(src1)(v1), nbu.type_ref(src2)(v2)
    for i in range(n):
        dst[i] = v1*src1[i] + v2*src2[i] #not a copy technically would be px
    return dst
#pxaxpy and cxapypz should see a 2x performance boost from counterparts

#this is actual a 3 port load technically but still seems to improve in benchmarking.
@nbu.jti
def axapypz(dst: np.ndarray, v1: float, v2: float, src1: np.ndarray, src2: np.ndarray):
    sr"""(to x) add product y add product z: $x \leftarrow v_1 \cdot x + v_2 \cdot y$"""
    n = src1.shape[0]
    v1, v2 = nbu.type_ref(src1)(v1), nbu.type_ref(src2)(v2)
    for i in range(n):
        dst[i] += v1*src1[i] + v2*src2[i]
    return dst

@nbu.jti
def cxapyz(dst: np.ndarray, v1: float, src1: np.ndarray, src2: np.ndarray):
    sr"""Copy product y add product z: $x \leftarrow v_1 \cdot x + \cdot y$"""
    n = src1.shape[0]
    v1 = nbu.type_ref(src1)(v1)
    for i in range(n):
        dst[i] = v1*src1[i] + src2[i] #not a copy technically would be px
    return dst

#slight improvement here as well.
@nbu.jti
def cxypz(dst1: np.ndarray, dst2: np.ndarray, a: float, b: float, src: np.ndarray):
    sr"""Copy (to x,y) products of a single source: 
    $x_i = a \cdot s_i,\; y_i = b \cdot s_i$"""
    n = src.shape[0]
    typ=nbu.type_ref(src)
    a, b = typ(a), typ(b)
    for i in range(n):
        s = src[i]
        dst1[i] = a * s
        dst2[i] = b * s
    return dst1, dst2


@nbu.jti
def axypz(dst1: np.ndarray, dst2: np.ndarray, a: float, b: float, src: np.ndarray):
    sr"""Add (to x,y) products of a single source:
    $x_i \mathrel{+}= a \cdot s_i,\; y_i \mathrel{+}= b \cdot s_i$"""
    n = src.shape[0]
    typ=nbu.type_ref(src)
    a, b = typ(a), typ(b)
    for i in range(n):
        s = src[i]
        dst1[i] += a * s
        dst2[i] += b * s
    return dst1, dst2
        
@nbu.jti
def vmax(x: np.ndarray) -> float:
    sr"""Finds the maximum value in a vector: $v \leftarrow \max(x_i)$"""
    typ=nbu.type_ref(x)
    v = typ(nbu.prim_info(typ,0))
    for e in x: 
        if e >v:v=e
    return v

@nbu.jti
def vmin(x: np.ndarray) -> float:
    sr"""Finds the minimum value in a vector: $v \leftarrow \min(x_i)$"""
    typ=nbu.type_ref(x)
    v =typ(nbu.prim_info(typ,1))
    for e in x:
        if e < v: v = e
    return v

@nbu.jti
def vminmax(x: np.ndarray) -> (float,float):
    sr"""Finds the minimum and maximum value in a vector: $\rightarrow \min(x_i),\max(x_i)$"""
    typ=nbu.type_ref(x)
    vm = typ(nbu.prim_info(typ,1))
    vx = typ(nbu.prim_info(typ,0))
    for e in x:
        if e < vm: vm = e
        if e>vx: vx = e
    return vm,vx

@nbu.jti
def argminmax(x: np.ndarray) -> (float,float):
    sr"""Finds the minimum and maximum value in a vector: $\rightarrow \min_i(x_i),\max_j(x_j), i, j$"""
    typ=nbu.type_ref(x)
    vm = typ(nbu.prim_info(typ,1))
    vx = typ(nbu.prim_info(typ,0))
    i=-1
    j=-1
    for n in range(x.shape[0]):
        e=x[n]
        if e < vm: 
            vm = e
            i=n
        if e>vx: 
            vx = e
            j=n
    #we can 
    return vm,vx,i,j

@nbu.jti
def __argminmax_2k(x: np.ndarray): #this is much slower than the simple 1 loop versions.
    sr"""Two step calculation, this only costs 1.5n comparisons instead of 2n comparisons of the previous method.
    But in practice it's of course much slower than `argminmax`.
    """
    typ = nbu.type_ref(x)
    vm = typ(nbu.prim_info(typ, 1))  # +∞ sentinel
    vx = typ(nbu.prim_info(typ, 0))  # -∞ sentinel
    i = -1
    j = -1
    n = x.shape[0]

    # iterate in pairs, starting from 0 for numba-compatibility
    for k in range(0, n - 1, 2):
        a = x[k]
        b = x[k + 1]
        if a < b:
            if a < vm:
                vm = a
                i = k
            if b > vx:
                vx = b
                j = k + 1
        else:
            if b < vm:
                vm = b
                i = k + 1
            if a > vx:
                vx = a
                j = k

    # handle tail element if n is odd
    if n & 1:
        e = x[-1]
        if e < vm:
            vm = e
            i = n - 1
        elif e > vx:
            vx = e
            j = n - 1

    return vm, vx, i, j

### --- IMPLICIT OPERATORS may move these to a separate module later
