from llvmlite import binding
from numba.core import typing
from numba.core.extending import overload

from _st_env import pre_start
pre_start()

import numpy as np
import numba as nb
from numba import types, literal_unroll
import random as rand
import time as t

from _for_benching import time_funcs, eval_and_plot_3d
from utils import stack_empty, numpy_to_tuples, type_ref
import math


@nb.njit(fastmath=True, error_model='numpy')
def rep_run(reps, func, *args):
    for _ in range(reps):
        func(*args)

@nb.njit(fastmath=True, error_model='numpy')
def arrayout_rep(inp, outp, func, *args):
    for i in range(inp.shape[0]):
        outp[i]=func(inp[i] ,*args)

@nb.njit(fastmath=True, error_model='numpy')
def array_rep(inp, func, *args):
    for i in range(inp.shape[0]):
        func(inp[i] ,*args)

@nb.njit(fastmath=True, error_model='numpy')
def arrayouti_rep(outp, func, *args):
    for i in range(outp.shape[0]):
        outp[i]=func(i,*args)

### Array Index Test Functions
tset = dict(fastmath=True, error_model='numpy')


@nb.njit(**tset)
def array_idx1(x):
    for i in range(x.shape[0]):
        rs = x[i]
        for v in range(x.shape[1]):
            # rs[v] += rand.random()
            rs[v] += rs[v] + 1e-8


@nb.njit(**tset)
def array_idx2(x):
    for i in range(x.shape[0]):
        for v in range(x.shape[1]):
            # x[i, v] += rand.random()
            x[i, v] += x[i, v] + 1e-8


@nb.njit(**tset)
def array_idx3(x):
    rs = x.reshape(-1)
    for i in range(rs.size):
        # rs[i] += rand.random()
        rs[i] += rs[i] + 1e-8


@nb.njit(**tset)
def array_idx4(x):
    s0 = x.shape[0]
    for g in range(x.size):
        i = g // s0
        v = g % s0
        x[i, v] += rand.random()


### Array Index Benchmark Initializations.
bset = dict(fastmath=True, error_model='numpy')

@nb.njit(**bset)
def bench_array_idx1(reps, rr):
    rep_run(reps, array_idx1, rr)

@nb.njit(**bset)
def bench_array_idx2(reps, rr):
    rep_run(reps, array_idx2, rr)

@nb.njit(**bset)
def bench_array_idx3(reps, rr):
    rep_run(reps, array_idx3, rr)

@nb.njit(**bset)
def bench_array_idx4(reps, rr):
    rep_run(reps, array_idx4, rr)

def array_indexing_benchmark():
    bench_ver_names = [f'Vec Array V{i}' for i in range(1, 4)]
    s1 = 3
    s2 = 14
    x = np.zeros((s1, s2), dtype=np.float64)

    def res_call():
        x[:] = 0.
        t.sleep(.05)

    b1 = lambda reps: bench_array_idx1(reps, x)
    b2 = lambda reps: bench_array_idx2(reps, x)
    b3 = lambda reps: bench_array_idx3(reps, x)
    #b4 = lambda reps: bench_array_idx4(reps, x) #significantly slower
    
    timing_run=5_000_000
    print(f's1={s1}, s2={s2}, timing_run={timing_run}')
    time_funcs((b1, b2, b3), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000, 
               timing_run=timing_run, 
               repeat_sequence=20,
               ops_mult=s1 * s2)
 
  
### Globals Example Functions
_jdgn=np.array([
    [0.286, 0.973, 0.384, 0.276, 0.973, 0.543, 0.957, 0.948, 0.543, 0.797, 0.936, 0.889, 0.006, 0.828, 0.399, 0.617, 0.939, 0.784, 0.072, 0.889],
    [0.645, 0.585, 0.31, 0.058, 0.455, 0.779, 0.259, 0.202, 0.028, 0.099, 0.142, 0.296, 0.175, 0.18, 0.842, 0.039, 0.103, 0.62, 0.158, 0.704],
    [4.284, 4.149, 3.877, 0.533, 2.211, 2.389, 2.145, 3.231, 1.998, 1.379, 2.106, 1.428, 1.011, 2.179, 2.858, 1.388, 1.651, 1.593, 1.046, 2.152]
],dtype=np.float64) #init in contiguous array.
#_jdgn=_jdgn[:,:12] #later example
_ja_ar =_jdgn[0]
_jb_ar =_jdgn[1]
_jc_ar =_jdgn[2]
_ja_tp=tuple(_ja_ar)
_jb_tp=tuple(_jb_ar)
_jc_tp=tuple(_jc_ar)
_jdgnt=np.zeros(_jdgn.T.shape,dtype=np.float64)
_jdgnt[:]=_jdgn.T #global array test 2
_jdgn_tp=numpy_to_tuples(_jdgnt)
_jdgn_ttp=tuple(_jdgnt.reshape(-1))
jt = nb.njit(fastmath=True, error_model='numpy')

@jt #Not used
def judge_eval0(x):
    _jdgn=np.array([
    [0.286, 0.973, 0.384, 0.276, 0.973, 0.543, 0.957, 0.948, 0.543, 0.797, 0.936, 0.889, 0.006, 0.828, 0.399, 0.617, 0.939, 0.784, 0.072, 0.889],
    [0.645, 0.585, 0.31, 0.058, 0.455, 0.779, 0.259, 0.202, 0.028, 0.099, 0.142, 0.296, 0.175, 0.18, 0.842, 0.039, 0.103, 0.62, 0.158, 0.704],
    [4.284, 4.149, 3.877, 0.533, 2.211, 2.389, 2.145, 3.231, 1.998, 1.379, 2.106, 1.428, 1.011, 2.179, 2.858, 1.388, 1.651, 1.593, 1.046, 2.152]
    ],dtype=np.float64)
    a_t=_jdgn[0]
    b_t=_jdgn[1]
    c_t=_jdgn[2]
    jvl=0.0
    for i in range(a_t.shape[0]):
        jvl+=((x[0] + x[1] * a_t[i] + (x[1] ** 2.0) * b_t[i]) - c_t[i])** 2.0
        
    return jvl

#### Array Functions
@jt #Local Arrays
def judge_eval1(x,a_t,b_t,c_t):
    jvl = 0.0
    for i in (range(len(a_t))):
        jvl += ((x[0] + x[1] * a_t[i] + (x[1] ** 2.0) * b_t[i]) - c_t[i]) ** 2.0
    return jvl

@jt # Global Arrays
def judge_eval2(x):
    #a_t = _ja_ar
    #b_t = _jb_ar
    #c_t = _jc_ar
    jvl = 0.0
    for i in (range(_ja_ar.shape[0])):
        jvl += ((x[0] + x[1] * _ja_ar[i] + (x[1] ** 2.0) * _jb_ar[i]) - _jc_ar[i]) ** 2.0
    return jvl

@jt #Local Optimal Array
def judge_eval3(x,jdgnt):
    jvl = 0.0
    for i in range(len(jdgnt)):
        jvl += ((x[0] + x[1] * jdgnt[i,0] + (x[1] ** 2.0) *  jdgnt[i,1]) -  jdgnt[i,2]) ** 2.0
    return jvl

@jt #Global Optimal Array
def judge_eval4(x):
    jvl = 0.0
    for i in range(len(_jdgnt)):
        #k=_jdgnt[i]
        jvl += ((x[0] + x[1] * _jdgnt[i,0] + (x[1] ** 2.0) *  _jdgnt[i,1]) -  _jdgnt[i,2]) ** 2.0
    #Pointing to k, or iterating on slices
    #worsens performance.
    return jvl

### Tuple Functions
# judge_eval5: Using judge_eval1 for the local tuple.
# 2D tuples can't be used in parallel regions, 
# which dq's testing judge_eval3 for tuples in my view. 

@jt #Global Tuples
def judge_eval6(x):
    a_t = _ja_tp
    b_t = _jb_tp
    c_t = _jc_tp
    jvl = 0.0
    for i in literal_unroll(range(len(a_t))):
        jvl += ((x[0] + x[1] * a_t[i] + (x[1] ** 2.0) * b_t[i]) - c_t[i]) ** 2.0
    return jvl

@jt #Global Tuple Iter
def judge_eval7(x):
    jvl = 0.0
    for jd in literal_unroll(_jdgn_tp):
        jvl += ((x[0] + x[1] * jd[0] + (x[1] ** 2.0) * jd[1]) - jd[2]) ** 2.0
    return jvl


@jt #Flat Global Tuple, Excluded performance never better than other tuple methods
def judge_eval8(x):
    jvl = 0.0
    for i in range(0,len(_jdgn_ttp),3):
        jvl += ((x[0] + x[1] * _jdgn_ttp[i] + (x[1] ** 2.0) * _jdgn_ttp[i+1]) - _jdgn_ttp[i+2]) ** 2.0
    return jvl


def globals_benchmark():
    #eval_and_plot_3d((judge_eval1,),((-3.5,3.5),(-3.5,3.5)),name='Judge Surface')
    #return
    bench_ver_names = [f'JV{i}' for i in range(1, 8)]
    timing_run = 2_000_000
    x=np.random.normal(0,1,(timing_run,4))
    inp=x[:,:3]
    outp =x[:,3]

    def res_call():
        outp[:]=0.
        t.sleep(.05)

    b1 = lambda reps: arrayout_rep(inp[:reps], outp[:reps], judge_eval1, _ja_ar, _jb_ar, _jc_ar)
    b2 = lambda reps: arrayout_rep(inp[:reps], outp[:reps], judge_eval2)
    b3 = lambda reps: arrayout_rep(inp[:reps], outp[:reps], judge_eval3, _jdgnt)
    b4 = lambda reps: arrayout_rep(inp[:reps], outp[:reps], judge_eval4)
    b5 = lambda reps: arrayout_rep(inp[:reps], outp[:reps], judge_eval1, _ja_tp, _jb_tp, _jc_tp)
    b6 = lambda reps: arrayout_rep(inp[:reps], outp[:reps], judge_eval6)
    b7 = lambda reps: arrayout_rep(inp[:reps], outp[:reps], judge_eval7)

    #Note: ops_mult does not account for transformations on the arrays
    #eg calls to expensive functions.
    #It is only a multiplier of the size you assign to ops_mult 
    #and the sum of all repetitions.
    #Therefore ops/sec is only comparable between competing benchmarks.
    print(f'timing_run={timing_run}')
    time_funcs((b1,b2,b3,b4,b5,b6,b7), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=16,
               ops_mult=20*3)
    
    
### Insert Sort Looping Example
jt = nb.njit(fastmath=True, error_model='numpy')

@jt #for while loop
def insert_sort1(arr): #second fastest
    n = arr.shape[0]
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key<arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

@jt #for for loop
def insert_sort2(arr):
    n = arr.shape[0]
    for i in range(1, n):
        key = arr[i]
        for j in range(i-1,-1,-1):
            if key>arr[j]:
                break
            arr[j + 1] = arr[j]
        else:
            j-=1
        arr[j+1] = key
        
@jt #while while loop
def insert_sort3(arr):
    n = arr.shape[0]
    i=1
    while i <n:
        key = arr[i]
        j = i - 1
        while j >= 0 and key<arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        i+=1
        
@jt #for while loop
def insert_sort4(arr): #fastest
    for i in range(1, arr.shape[0]):
        k = arr[i]
        j = i
        while j > 0 and k < arr[j - 1]:
            # Make place for moving A[i] downwards
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = k

def looptype_benchmark():
    bench_ver_names = [f'Sort Loop {i}' for i in range(1, 5)]
    timing_run = 1000_000
    ssz=20
    rp=np.random.normal(0,1,(timing_run,ssz))
    x=np.empty((timing_run,ssz),dtype=np.float64)

    def res_call():
        x[:]=rp
        t.sleep(.05)

    #Make sure they are the same
    # rp[1]=rp[0]
    # rp[2]=rp[0]
    # insert_sort1(rp[0])
    # x[0]=rp[0]
    # insert_sort2(rp[1])
    # x[1]=rp[1]
    # insert_sort1(rp[2])
    # x[2]=rp[2]
    # print(x[:3])

    b1 = lambda reps: array_rep(x[:reps], insert_sort1)
    b2 = lambda reps: array_rep(x[:reps], insert_sort2)
    b3 = lambda reps: array_rep(x[:reps], insert_sort3)
    b4 = lambda reps: array_rep(x[:reps], insert_sort4) #fastest version

    print(f'sort_size={ssz}, timing_run={timing_run}')
    time_funcs((b1,b2,b3,b4), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=10,
               ops_mult=ssz*ssz/2)


### Records vs 2D

@jt #Global Optimal Array
def judge_record(x):
    jvl = 0.0
    for i in range(len(_jdgnt)):
        jvl += ((x['x1'] + x['x2'] * _jdgnt[i,0] + (x['x2'] ** 2.0) *  _jdgnt[i,1]) -  _jdgnt[i,2]) ** 2.0
    return jvl

@jt #Global Optimal Array
def judge_array(x):
    jvl = 0.0
    for i in range(len(_jdgnt)):
        jvl += ((x[0] + x[1] * _jdgnt[i,0] + (x[1] ** 2.0) *  _jdgnt[i,1]) -  _jdgnt[i,2]) ** 2.0
    return jvl

@jt
def judge_record_bench(x,outp):
    arrayout_rep(x,outp,judge_record)

@jt
def judge_array_bench(x,outp):
    arrayout_rep(x,outp,judge_array)

def recordvarray_benchmark():
    bench_ver_names = [f'Sort Loop {i}' for i in range(1, 3)]
    timing_run = 50_000_000
    x=np.random.normal(0,1,(timing_run,2))
    xr = np.zeros(timing_run, dtype=[('x1', 'f8'), ('x2', 'f8')])
    xr['x1'] = x[:, 0]
    xr['x2'] = x[:, 1]
    outp=np.empty((timing_run,),dtype=np.float64)
    
    def res_call():
        t.sleep(.05)

    #b1 = lambda reps: judge_record_bench(xr[:reps], outp[:reps])# memory of inp may be further away than x than xr so may already have an additional penalty.
    b1 = lambda reps: arrayout_rep(xr[:reps], outp[:reps],judge_record)
    #b2 = lambda reps: judge_array_bench(x[:reps],outp[:reps])
    b2 = lambda reps: arrayout_rep(x[:reps], outp[:reps],judge_array)

    print(f'timing_run={timing_run}')
    time_funcs((b1,b2), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=20,
               ops_mult=60)


### Vectorized Numpy vs Loops
import math
_levy05_1idx = np.arange(1, 6)
@jt
def levy05_eval1(x):
    idx =_levy05_1idx
    a = idx * np.cos((idx - 1) * x[0] + idx)
    b = idx * np.cos((idx + 1) * x[1] + idx)
    return np.sum(a) * np.sum(b) + (x[0] + 1.42513) ** 2 + (x[1] + 0.80032) ** 2

@jt
def levy05_eval2(x):
    att,btt=0.0,0.0
    for i in range(1,6):
        att+= i*np.cos((i-1)*x[0])+i
        btt+=i*np.cos((i+1)*x[1])+i
    return att*btt + (x[0] + 1.42513) ** 2 + (x[1] + 0.80032) ** 2

@jt
def levy05_eval3(x):
    x0,x1=x[0],x[1]
    att,btt=0.0,0.0
    for i in range(1,6):
        att+= i*math.cos((i-1)*x0)+i
        btt+=i*math.cos((i+1)*x1)+i
    return att*btt + (x0 + 1.42513) ** 2 + (x1 + 0.80032) ** 2

def numpyvectorizedvsloop_benchmark():
    #eval_and_plot_3d((levy05_eval1,),((-8.,8.),(-8.,8.)),name='Levy5 Surface',num_points=500_000)
    #return
    bench_ver_names = [f'Levy5 {i}' for i in range(1, 4)]
    timing_run = 2_000_000
    rp=np.random.normal(0,1,(timing_run,3))
    x=rp[:,:2]
    outp=rp[:,2]

    def res_call():
        t.sleep(.05)

    b1 = lambda reps: arrayout_rep(x[:reps],outp[:reps], levy05_eval1)
    b2 = lambda reps: arrayout_rep(x[:reps],outp[:reps], levy05_eval2)
    b3 = lambda reps: arrayout_rep(x[:reps],outp[:reps], levy05_eval3)

    print(f'timing_run={timing_run}')
    time_funcs((b1,b2,b3), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=10,
               ops_mult=2*6)

### Vectorized Numpy vs Loops
@jt
def outerdot1(x):
    vlm=np.outer(x[0],x[1])
    x[2:]=vlm

@jt
def outerdot2(x,tmp):
    tmp=np.outer(x[0],x[1],out=tmp)
    x[2:]=tmp

@jt
def outerdot3(x):
    np.outer(x[0],x[1],out=x[2:])

# @jt
# def outerdot4(x):
#     np.linalg.outer(x[0],x[1],out=x[2:])

@jt
def outerdot5(x):
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            x[2 + i , j] = x[0, i] * x[1, j]
  
@jt
def custom_outer(a,b,out):
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            out[i , j] = a[i] * b[j]

@jt #evidence your own generic kernels can be faster than numpy routines.
def outerdot6(x):
    a=x[0]
    b=x[1]
    out=x[2:]
    custom_outer(a,b,out)



def numpyvectorizedvsloop2_benchmark():
    bench_ver_names = [f'OuterDot {i}' for i in range(1, 6)]
    timing_run = 100
    vsz=1024
    rp = np.random.normal(0, 1, (timing_run, 2, vsz))
    xio=np.empty((timing_run,2+vsz,vsz),dtype=np.float64)
    xi=xio[:,:2,:]
    xi[:]=rp
    tmp=np.empty((vsz,vsz),dtype=np.float64)
    # xo=xio[:,2:,:]

    def res_call():
        t.sleep(.05)

    #proof same result.
    # array_rep(xio, outerdot1)
    # print(xio[0])
    # tmp[:]=xio[0]
    # array_rep(xio, outerdot5)
    # print(xio[0]-tmp) 
    # return

    b1 = lambda reps: array_rep(xio[:reps], outerdot1)
    b2 = lambda reps: array_rep(xio[:reps], outerdot2,tmp)
    b3 = lambda reps: array_rep(xio[:reps], outerdot3)
    #b4 = lambda reps: array_rep(xio[:reps], outerdot4)
    b4 = lambda reps: array_rep(xio[:reps], outerdot5)
    b5 = lambda reps: array_rep(xio[:reps], outerdot6)

    print(f'timing_run={timing_run}')
    time_funcs((b1,b2,b3,b4,b5), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=10,
               ops_mult=vsz*vsz)


### Numpy vs native, RNGs
import random as rand
import math as mt
_rng = np.random.default_rng(seed=None)
jtp= nb.njit(fastmath=True, error_model='numpy',parallel=True)

@jt
def randint1_s(reps):
    return np.random.randint(0,100,(reps,))

@jtp
def randint1_pl(reps):
    return np.random.randint(0,100,(reps,))

@jt
def randint2_s(reps):
    x=np.empty((reps,),dtype=np.int64)
    for i in range(reps):
        x[i]=rand.randint(0,100)
    return x

@jtp
def randint2_pl(reps): #See hardware specs in the guide uses 16 threads, 8 cores.
    x=np.empty((reps,),dtype=np.int64)
    pl=nb.set_parallel_chunksize(mt.ceil(reps/nb.get_num_threads()))
    for i in nb.prange(reps):
        x[i]=rand.randint(0,100)
    nb.set_parallel_chunksize(pl)
    return x

@jt
def randint3(reps,rng): #does not parallelize, wrapping in parallel block or with different generators adds no speed improvement... maybe there is a class based lock?
    return rng.integers(0,100,(reps,),dtype=np.int64)

@jt
def randint4(reps,rng):
    # to make it a fair comparison to the loop methods, need to treat x as a reused temp array and rng as producing new arrays at each use.
    x = np.empty((reps,), dtype=np.int64)
    x[:]=rng.integers(0,100,(reps,),dtype=np.int64)
    return x

#Parallel regions numpy rng Generator's are not supported
#So can't have pl for 3 or 4.
#https://numba.readthedocs.io/en/stable/reference/numpysupported.html#random

@jt #Excluding, can be 30-50x slower, because calls cfunc.
def randint5(reps,rng):
    x=np.empty((reps),dtype=np.int64)
    for i in range(reps):
        x[i]=rng.integers(0,100,dtype=np.int64)
        #init arg with (1,) just in case.
        #x[i] = rng.integers(0, 100,(1,), dtype=np.int64)
    return x
  
def numpyvnative_randint_benchmark():
    cores=nb.get_num_threads()//2 #2 threads per core for my hardware.
    bench_ver_names =('Ri_s 1','Ri_pl 1','Ri_s 2','Ri_pl 2','Ri 3','Ri 4')
    thd_seq=(1,cores,1,cores,1,1,cores,1)
    timing_run = 5_000_000

    def res_call():
        t.sleep(.05)

    b1 = lambda reps: randint1_s(reps) #only difference is thread local state, likely explains small cost for sync performance.
    b2 = lambda reps: randint1_pl(reps)
    b3 = lambda reps: randint2_s(reps)
    b4 = lambda reps: randint2_pl(reps)
    b5 = lambda reps: randint3(reps,_rng)
    b6 = lambda reps: randint4(reps, _rng)
    #b8 = lambda reps: randint5(reps,_rng)

    print(f'timing_run={timing_run}')
    time_funcs((b1,b2,b3,b4,b5,b6), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=16,
               ops_mult=1,
               thread_seq=thd_seq)

###f64 vs f32 operations

def f64vf32_benchmark():
    bench_ver_names = ('f64 Outer','f32 Outer')
    timing_run = 1_000_000
    vsz = 5
    rp = np.random.normal(0, 1, (timing_run, 2, vsz))
    xio1 = np.empty((timing_run, 2 + vsz, vsz), dtype=np.float64)
    xio1[:, :2, :] = rp
    xio2 = np.empty((timing_run, 2 + vsz, vsz), dtype=np.float32)
    xio2[:, :2, :] = rp
    
    def res_call():
        t.sleep(.05)

    b2 = lambda reps: array_rep(xio1[:reps], outerdot6)
    b1 = lambda reps: array_rep(xio2[:reps], outerdot6)
    
    print(f'timing_run={timing_run}')
    time_funcs((b2,b1), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=50,
               ops_mult=vsz*vsz)

### Round to Ints
#roundint1=jt(lambda rd : nb.int64(rd + .5)) #no difference not including
#roundint2=jt(lambda rd : np.round(rd))

roundint1=jt(lambda rd : nb.int32(rd + nb.float32(.5)))
roundint2=jt(lambda rd : np.round(rd))
roundint3=jt(lambda rd : np.floor(rd + nb.float32(.5)))

def roundi_benchmark():
    bench_ver_names = [f'Round To Int {i}' for i in range(1, 4)]
    timing_run = 100_000_000
    tx=np.random.normal(0,5.,(timing_run,))
    fr=np.empty((timing_run,2),dtype=np.float32)
    x=fr[:,0]
    x[:]=tx
    out=fr[:,1].astype(dtype=np.int32,copy=False)

    def res_call():
        t.sleep(.05)

    b1 = lambda reps: arrayout_rep(x[:reps],out[:reps], roundint1)
    b2 = lambda reps: arrayout_rep(x[:reps],out[:reps], roundint2)
    b3 = lambda reps: arrayout_rep(x[:reps], out[:reps], roundint3)

    print(f'timing_run={timing_run}')
    time_funcs((b1,b2,b3), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=25,
               ops_mult=1)


### Unrolling tests found online.
@jt
def k1(a):
    for i in range(a.size):
        a[i] += 1

@jt
def k2(a):
    for i in range(a.size):
        a[i] *= 2

@jt
def k3(a):
    k1(a)
    k2(a)
    k1(a)
    k2(a)

@jt
def k123(a):
    for i in range(a.size):
        a[i] = 2 * (2 * (a[i] + 1) + 1)

@jt
def l1(a):
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = math.sin(a[i])
    return out

@jt
def l2(a):
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = math.cos(a[i])
    return out

@jt
def l3(a):
    return l1(l2(l1(l2(a))))

@jt
def l123(a):
    out = np.empty_like(a)
    for i in range(a.size):
        out[i] = math.sin(math.cos(math.sin(math.cos(a[i]))))
    return out

def aloopfusion_benchmark():
    bench_ver_names = [f'Loop Fusion {i}' for i in range(1, 3)]
    timing_run = 5_000_000
    x=np.random.normal(0,1,(timing_run,))
    def res_call():
        #x[:]=rp
        t.sleep(.05)

    b1 = lambda reps: l3(x[:reps])
    b2 = lambda reps: l123(x[:reps])

    print(f'timing_run={timing_run}')
    time_funcs((b1,b2), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=10,
               ops_mult=4)
    #Loop Fusion Manual Total Time: 1.415 seconds. (Estimated ops million)/second: 141.313
    #Loop Fusion Automatic w/ Arrays Total Time: 1.491 seconds. (Estimated ops million)/second: 134.129
    #While the difference still exists, it's not nearly as bad as it used to be.


@jt
def outer_s1(a,b):
    tp=type_ref(a)
    sm=tp(0.0)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            sm += a[i] * b[j]
    return sm

GTHDS=nb.get_num_threads()

@jtp
def outer_pl1(a,b):
    tp=type_ref(a) #an overloads to give us the type and use it at compile time
    ar = stack_empty(GTHDS,(GTHDS,),dtype=tp) #init a stack address at compile time, instead of creating new empty heap arrays.
    for i in range(GTHDS):
        ar[i]=tp(0.)
    pl=nb.set_parallel_chunksize(mt.floor(a.shape[0]/GTHDS))
    for i in nb.prange(a.shape[0]):
        td=nb.get_thread_id() #... check against the outer prange version, to minimize id calls.
        for j in range(b.shape[0]):
            ar[td] += a[i] * b[j]
    nb.set_parallel_chunksize(pl)
    sm=tp(0.)
    for i in range(GTHDS):
        sm+=ar[i]
    return sm


@jtp
def outer_pl2(a,b):
    #this is no better than outer_pl1, no need to over-engineer parallel regions.
    tp=type_ref(a)
    ar = stack_empty(GTHDS,(GTHDS,),dtype=tp)
    psz=mt.ceil(a.shape[0]/GTHDS)
    pl=nb.set_parallel_chunksize(1)
    for n in nb.prange(GTHDS):
        st=n*psz
        ed=min((n+1)*psz,a.shape[0])
        tsm=tp(0.) #now can be thread local.
        for i in range(st,ed):
            for j in range(b.shape[0]):
                tsm += a[i] * b[j]
        ar[nb.get_thread_id()]=tsm
    nb.set_parallel_chunksize(pl)
    sm=tp(0.)
    for i in range(GTHDS):
        sm+=ar[i]
    return sm

@jtp
def outer_pl3(a,b):
    ar = stack_empty(GTHDS,(GTHDS,),dtype=np.float64)
    for i in range(GTHDS):
        ar[i]=0.
    s1=a.shape[0]
    s2=b.shape[0]
    pl=nb.set_parallel_chunksize(mt.ceil(s1*s2/GTHDS))
    for n in nb.prange(s1*s2):
        i=n//s2
        j=n%s2
        ar[nb.get_thread_id()] += a[i] * b[j]
    nb.set_parallel_chunksize(pl)
    sm=0.
    for i in range(GTHDS):
        sm+=ar[i]
    return sm

def make_tiled_outer_sum(block_i=64, block_j=64):
    """
    Returns a JIT-compiled function that sums over a[i] * b[j],
    but in tile blocks of size (block_i, block_j).
    """
    @jt
    def tiled_outer_sum(a, b):
        tp = type_ref(a)
        nA = a.shape[0]
        nB = b.shape[0]
        total = tp(0.0)

        for i0 in range(0, nA, block_i):
            i_max = min(i0 + block_i, nA)

            for j0 in range(0, nB, block_j):
                j_max = min(j0 + block_j, nB)

                # Accumulate partial sums over this tile
                tile_sum = tp(0.0)
                for i in range(i0, i_max):
                    ai = a[i]   # hoist a[i] out of the inner loop
                    for j in range(j0, j_max):
                        tile_sum += ai * b[j]

                total += tile_sum
        return total

    return tiled_outer_sum

touters_64x64=make_tiled_outer_sum(64,64)

@jt
def run_outer(x,outerfunc):
    a=x[0]
    b=x[1]
    return outerfunc(a,b)  

def outerparallel_benchmark():
    bench_ver_names=('OSum_S64','OSum_S32','OSum_PL64','OSum_PL32','tiled')
    timing_run = 10
    vsz=4096
    rp = np.random.normal(0, 1, (timing_run, 2, vsz))
    x=np.empty((timing_run,),dtype=np.float64)
    x32 = np.empty((timing_run,), dtype=np.float32)
    rp32=np.array(rp,dtype=np.float32)

    def res_call():
        t.sleep(.05)

    print('plthreads',nb.get_num_threads())

    b1 = lambda reps: arrayout_rep(rp[:reps],x[:reps], run_outer,outer_s1)
    b2 = lambda reps: arrayout_rep(rp32[:reps], x32[:reps], run_outer, outer_s1)
    b3 = lambda reps: arrayout_rep(rp[:reps],x[:reps], run_outer, outer_pl1)
    b4 = lambda reps: arrayout_rep(rp32[:reps], x32[:reps], run_outer, outer_pl1)
    #b5 = lambda reps: arrayout_rep(rp[:reps], x[:reps], run_outer, touters_64x64)
    # b5 = lambda reps: arrayout_rep(rp[:reps], x[:reps], run_outer, outer_pl2)
    # b6 = lambda reps: arrayout_rep(rp32[:reps], x32[:reps], run_outer, outer_pl2)


    print(f'timing_run={timing_run}')
    time_funcs((b1,b2,b3,b4), bench_ver_names, res_call,
               compile_run=10,
               init_run=10,
               timing_run=timing_run,
               repeat_sequence=20,
               ops_mult=vsz*vsz)


#Integer range placement, rangei3 is the fastest (with step = 1), and it adapts implicitly to different dtypes unlike rangei1
MAXI64=np.iinfo(np.int64).max

@jt
def rangei1(x, start = 0, stop = MAXI64 , step = 1):
    idx = 0
    #stop = x.shape[0] if stop == MAXI64 else stop
    for i in range(start, stop, step):
        x[idx] = i
        idx += 1

@jt
def rangei2(x, start = 0, stop = MAXI64 , step = 1):
    #stop = x.shape[0] if stop == MAXI64 else stop
    x[:]=np.arange(start,stop,step)

@jt
def rangei3(x, start = 0 , step = 1):
    #stop = x.shape[0] if stop == MAXI64 else stop
    for i in range(x.shape[0]):
        x[i] = start + i*step

@jt
def rangei4(x, start = 0 , step = 1):
    #stop = x.shape[0] if stop == MAXI64 else stop
    for i in range(x.shape[0]):
        x[i] = i*step
    x[:]+=start



def arangei_benchmark():
    bench_ver_names = [f'IRange {i}' for i in range(1, 4)]
    timing_run = 10000
    psz=10000
    x=np.empty((timing_run,psz),dtype=np.int64)

    def res_call():
        t.sleep(.05)

    b3 = lambda reps: array_rep(x[:reps], rangei1,0,psz)
    b1 = lambda reps: array_rep(x[:reps], rangei2,0,psz)
    b2 = lambda reps: array_rep(x[:reps], rangei3,0,1)
    #b4 = lambda reps: array_rep(x[:reps], rangei4,0,1)

    print(f'timing_run={timing_run}, psz={psz}')
    time_funcs((b1,b2,b3), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=8,
               ops_mult=psz)


# --- Argmin/argmax implementations seem to be quickest in numba.
MAXF64=np.finfo(np.float64).max

@jt
def argmin1(x): #slowest because repeat array accesses in dynamic matter.
    cb=0
    for i in range(1,x.shape[0]):
        if x[cb] > x[i]: cb=i
    return cb

@jt
def argmin2(x): #Fastest
    return np.argmin(x)

@jt
def argmin3(x): #slower than argmin2 by hair's width.
    cb=0
    cmn=x[0]
    for i in range(1,x.shape[0]):
        if cmn > x[i]: 
            cmn=x[i]
            cb=i  
    return cb

def argmin_benchmark():
    bench_ver_names = [f'ArgMin {i}' for i in range(1, 4)]
    timing_run = 5_000_000
    psz=100
    x=np.random.normal(0,1,(timing_run,psz+1))
    otr=x[:,100]
    x=x[:,:100]

    def res_call():
        t.sleep(.05)

    b1 = lambda reps: arrayout_rep(x[:reps],otr[:reps], argmin1)
    b2 = lambda reps: arrayout_rep(x[:reps],otr[:reps], argmin2)
    b3 = lambda reps: arrayout_rep(x[:reps],otr[:reps], argmin3)

    print(f'timing_run={timing_run}, psz={psz}')
    time_funcs((b1,b2,b3), bench_ver_names, res_call,
               compile_run=100,
               init_run=1_000,
               timing_run=timing_run,
               repeat_sequence=20,
               ops_mult=psz)
    

@jt
def generic_cr_f_apply1(fac_arr,cr,f): #fastest, because 1 less boolean check.
    l = fac_arr.shape[0]
    j_rand = rand.randrange(0, l)
    for i in range(l):
        if rand.random() < cr:
            fac_arr[i] = f
        else:
            fac_arr[i] = 0.
    fac_arr[j_rand]=f


@jt
def generic_cr_f_apply2(fac_arr,cr,f):
    l = fac_arr.shape[0]
    j_rand = rand.randrange(0, l)
    for i in range(l):
        if rand.random() < cr or i==j_rand:
            fac_arr[i] = f
        else:
            fac_arr[i] = 0.

@jt
def generic_cr_f_apply3(fac_arr,cr,f):
    l = fac_arr.shape[0]
    j_rand = rand.randrange(0, l)
    for i in range(l):
        if rand.random() < cr or j_rand==i:
            fac_arr[i] = f
        else:
            fac_arr[i] = 0.


def crfapp_benchmark():
    #crossover + factor application for the differential evolution algorithm
    bench_ver_names = [f'CRFApp {i}' for i in range(1, 4)]
    timing_run = 100_000
    psz=1000
    x=np.empty((timing_run,psz),dtype=np.float64)

    def res_call():
        t.sleep(.05)
    cr,f=.8,.7
    b1 = lambda reps: array_rep(x[:reps],generic_cr_f_apply1,cr,f)
    b2 = lambda reps: array_rep(x[:reps],generic_cr_f_apply2,cr,f)
    b3 = lambda reps: array_rep(x[:reps], generic_cr_f_apply3, cr, f)

    print(f'timing_run={timing_run}, psz={psz}, cr={cr}, f={f}')
    time_funcs((b1,b2,b3), bench_ver_names, res_call,
               compile_run=1000,
               init_run=100_000,
               timing_run=timing_run,
               repeat_sequence=10,
               ops_mult=psz)


@jt
def covar_inplace(x, C, _tmp):
    # much slower than the lapack np.dot call, for large p and moderate to large d.
    # Meaning it's probably not worth using anything but the inplace np.dot call version.
    """
    x  : (p,d)  – modified in place (centred, *not* re‑shifted here)
    C  : (d,d)  – filled with 1/p * XᵀX   (full, symmetric)
    _tmp : (2,d) – scratch for the mean
    """
    return
    p, d = x.shape
    tpC = type_ref(C)
    tpt = type_ref(_tmp)  # overloads to get and use array types at compile time.

    mu = _tmp[0]
    mu[:] = tpt(0.)
    for i in range(p):
        row = x[i]
        for j in range(d):
            mu[j] += row[j]
    tp = tpt(p)
    mu[:] /= tp
    xtmp = _tmp[1]

    if d > 64:
        C.reshape(-1)[:] = tpC(0.0)
    else:
        for j in range(d):
            for k in range(d):
                C[j, k] = tpC(0.0)

    for i in range(p):  # outer loop over contiguous rows
        row = x[i]
        for j in range(d):
            xtmp[j] = ((row[j]) - mu[j]) / tp  # if p>d then might be less efficient but leaving it for tighter locality
        for j in range(d):
            vj=xtmp[j]
            for k in range(d):  # 2x faster because the loop isn't dynamic, maybe having it as a while loop for triangle would be faster?
                C[j, k] += vj * xtmp[k]

    # # If we want to symmetrize
    # if True:
    #     for j in range(d):
    #         for k in range(j, d):
    #             C[k, j] = C[j, k]


@jtp
def covar_inplace_vops(x, C, tmp):
    p, d = x.shape
    tpt = type_ref(tmp)  # overloads to get and use array types at compile time.
    tp = tpt(p)
    csize=20_000_000 #250_000
    if p*d>=csize and p*2>GTHDS and tmp.shape[0]>1:
        mu = tmp
        mu[:] = tpt(0.)
        ld = nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            thdd=nb.get_thread_id()
            for j in range(d):
                mu[thdd, j] += x[i,j]
        ms = mu[0]
        for i in range(1, GTHDS - 1):
            for j in range(d):
                ms[j] += mu[i, j]
        for j in range(d):
            ms[j] = (ms[j]+mu[-1, j])/tp
        for i in nb.prange(p):
            for j in range(d):
                x[i, j] -= ms[j]
        nb.set_parallel_chunksize(ld)
    else:
        ms=tmp[0]
        ms[:] = x[0]
        for i in range(1, p - 1):
            for j in range(d):
                ms[j] += x[i, j]
        for j in range(d):
            ms[j]=(ms[j]+x[-1, j])/tp
        for i in range(p):
            for j in range(d):
                x[i, j] -= ms[j]

    #Overall the performance benefit of these parallel blocks is negligible, only at millions of operations does it add a little
    #benefit. This might be because the lapack call of dot for large arrays occupies enough system recourses to undermine launching parallel blocks in rapid succession. Everything else has little impact on execution speed besides the np.dot call, for large d and p.
    np.dot(x.T, x, out=C)

    ct = C.reshape(-1)
    ttp=tp**2.
    if d * d >= 2**20:
        ld = nb.set_parallel_chunksize(mt.ceil(ct.size / GTHDS))
        for i in nb.prange(ct.size): ct[i] /= ttp
    else:
        for i in range(ct.size): ct[i] /= ttp

    if p * d >= csize:
        nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            for j in range(d): x[i, j] += ms[j]
        nb.set_parallel_chunksize(ld)
    else:
        for i in range(p):
            for j in range(d): x[i, j] += ms[j]
            

@jtp
def covar_inplace_vops2(x, C, tmp): #Somehow this version seems most quick
    p, d = x.shape
    tpt = type_ref(tmp)  # overloads to get and use array types at compile time.
    tp = tpt(p)
    csize=20_000_000
    if p*d>=csize and p*2>GTHDS and tmp.shape[0]>1:
        mu = tmp
        mu[:] = tpt(0.)
        ld = nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            thdd=nb.get_thread_id()
            for j in range(d):
                mu[thdd, j] += x[i,j]
        ms = mu[0]
        for i in range(1, GTHDS - 1):
            for j in range(d):
                ms[j] += mu[i, j]
        for j in range(d):
            ms[j] = (ms[j]+mu[-1, j])/tp
        for i in nb.prange(p):
            for j in range(d):
                x[i, j] = (x[i, j]-ms[j])/tp
        nb.set_parallel_chunksize(ld)
    else:
        ms=tmp[0]
        ms[:] = x[0]
        for i in range(1, p - 1):
            for j in range(d):
                ms[j] += x[i, j]
        for j in range(d):
            ms[j]=(ms[j] + x[-1,j])/tp
        for i in range(p):
            for j in range(d):
                x[i, j]= (x[i, j]-ms[j])/tp

    #For large # elements, everything else has little impact on execution speed besides the np.dot call and C or d adjustment.
    np.dot(x.T, x, out=C)

    if p * d >= csize:
        nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            for j in range(d): x[i, j]= x[i, j] * tp + ms[j]
        nb.set_parallel_chunksize(ld)
    else:
        for i in range(p):
            for j in range(d): x[i, j]= x[i, j] * tp + ms[j]


@jtp
def covar_inplace_vops3(x, C, tmp): #locality of loops dont help.
    p, d = x.shape
    tpt = type_ref(tmp)  # overloads to get and use array types at compile time.
    tp = tpt(p)
    ttp = tp**.5
    csize = 20_000_000  # 250_000
    if p * d >= csize and p * 2 > GTHDS and tmp.shape[0] > 1:
        mu = tmp
        mu[:] = tpt(0.)
        ld = nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            thdd = nb.get_thread_id()
            for j in range(d):
                mu[thdd, j] += x[i, j]
        ms = mu[0]
        for i in range(1, GTHDS):
            for j in range(d): ms[j] += mu[i, j]
        for j in range(d): ms[j] /=tp
        for i in nb.prange(p):
            for j in range(d):
                x[i, j] = (x[i, j] - ms[j]) / ttp
        nb.set_parallel_chunksize(ld)
    else:
        ms = tmp[0]
        ms[:] = 0.#x[0]
        for i in range(p):
            for j in range(d):ms[j] += x[i, j]
        for j in range(d):ms[j]/=tp
        for i in range(p):
            for j in range(d):
                x[i, j]=(x[i, j]-ms[j])/ttp
                

    # For large # elements, everything else has little impact on execution speed besides the np.dot call and C or d adjustment.
    np.dot(x.T, x, out=C)

    if p * d >= csize:
        nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            for j in range(d): x[i, j] = x[i, j] * ttp + ms[j]
        nb.set_parallel_chunksize(ld)
    else:
        for i in range(p):
            for j in range(d): x[i, j] = x[i, j] * ttp + ms[j]

# Load MKL shared library (adjust path if needed)
binding.load_library_permanently("mkl_rt.2.dll")

# Define CBLAS constants
# Type aliases for Numba types
_itt = types.intp    # native pointer-sized integer
_chtp = types.char   # single character

# CBLAS constants (wrapped in pointer-size integer type)
CBLAS_ROW_MAJOR = _itt(101)
CBLAS_COL_MAJOR = _itt(102)
CBLAS_NO_TRANS  = _itt(111)
CBLAS_TRANS     = _itt(112)
CBLAS_UPPER     = _itt(121)
CBLAS_LOWER     = _itt(122)
CBLAS_UNIT      = _itt(131)
CBLAS_NONUNIT   = _itt(132)
CBLAS_LEFT      = _itt(141)
CBLAS_RIGHT     = _itt(142)

# LAPACKE constants (encoded as character ordinals)
LAPACK_ROW_MAJOR = CBLAS_ROW_MAJOR
LAPACK_COL_MAJOR = CBLAS_COL_MAJOR
LAPACK_UPPER     = _chtp(ord('U'))
LAPACK_LOWER     = _chtp(ord('L'))
LAPACK_NO_TRANS  = _chtp(ord('N'))
LAPACK_TRANS     = _chtp(ord('T'))
LAPACK_CONJ_TRANS= _chtp(ord('C'))
LAPACK_NON_UNIT  = _chtp(ord('N'))
LAPACK_UNIT      = _chtp(ord('U'))
LAPACK_LEFT      = _chtp(ord('L'))
LAPACK_RIGHT     = _chtp(ord('R'))

"""
typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;
typedef enum {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
"""

# Define the signature for cblas_dsyrk
itt=types.intp
cblas_dsyrk_sig = typing.signature(
    types.void,
    itt,          # layout
    itt,          # uplo
    itt,          # trans
    itt,          # n
    itt,          # k
    types.double,        # alpha
    types.CPointer(types.double),  # A
    itt,          # lda
    types.double,        # beta
    types.CPointer(types.double),  # C
    itt           # ldc
)

# External binding to cblas_dsyrk
cblas_dsyrk_ext = types.ExternalFunction("cblas_dsyrk", cblas_dsyrk_sig)


@nb.njit(fastmath=True, error_model='numpy',parallel=True,nogil=True)
def syrk_numba(a, out, a_mult=1.,rem_mult=0.,order=CBLAS_ROW_MAJOR,uplo=CBLAS_UPPER,trans=CBLAS_TRANS):
    """
    In-place: C := alpha*A*A^T + beta*C on the chosen triangle.
    A, C: float64 arrays.
    Flags: int enums like 121 (Upper), 111 (NoTrans)
    """
    #layout = CBLAS_ROW_MAJOR
    # n, k = A.shape
    # lda = k

    if trans == CBLAS_NO_TRANS:
        n, k = a.shape
        lda = k
    else:
        k, n = a.shape
        lda = n

    ldc = out.shape[0]
    cblas_dsyrk_ext(
        order,
        uplo,
        trans,
        n,
        k,
        a_mult,
        a.ctypes,
        lda,
        rem_mult,
        out.ctypes,
        ldc
    )

@jt
def innermul_cself(a,out,a_mult=1.,rem_mult=0.,sym=False):
    #will write an inner product to out, so if a : (m,n) then out : (n,n)
    #c order otherwise first letter before self is f.
    #sym : fill lower triangle
    syrk_numba(a,out,a_mult,rem_mult)
    if sym:
        for j in range(a.shape[1]):
            for i in range(j+1,a.shape[1]):
                out[i, j] = out[j, i]
    

@jtp
def covar_inplace_vops4(x, C, tmp,place_back=True,symmetrize=True):
    """ Calculates the square covariance matrix by modifying the existing 2D array x, then places them back after C is calculated. 
    x may accumulate float precision errors on the order of (dtype epsilon) * x.shape[0] (this is very small). But take note of this if x has to be completely stationary, eg if used in deterministic algorithms with a worst case artifact multiplier.
    np.dot calls lapack routine for matrix product.
    
    :param x: (n,m) float array. 
    :param C: (m,m) float array. 
    :param tmp: (1,m) or (# threads,m) float array. Controls if x conditioning routine should be parallel. The np.dot lapack call will still be parallel even for a synchronous jit decorator.
    """
    #Figure out why this won't cache, maybe the parallel blocks?
    ispl= tmp.shape[0]>1
    p, d = x.shape
    tpt = type_ref(tmp)  # overloads to get and use array types at compile time.
    tp = tpt(p)
    #suspect the tradeoff might be so high because io is increased with the pl launch right before np.dot call which slows it down a bit.
    csize = 20_000_000
    if ispl and p * d >= csize:
        mu = tmp
        mu[:] = tpt(0.)
        ld = nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            thdd = nb.get_thread_id()
            for j in range(d):
                mu[thdd, j] += x[i, j]
        ms = mu[0]
        for i in range(1, GTHDS):
            for j in range(d): ms[j] += mu[i, j]
        for j in range(d): ms[j] /= tp
        for i in nb.prange(p):
            for j in range(d):
                x[i, j] = (x[i, j] - ms[j]) #/ tp
        nb.set_parallel_chunksize(ld)
    else:
        ms = tmp[0]
        ms[:] = 0.  # x[0] #apparently its faster to set to 0. Accumulate for exact dimensions, and finish with a separate /=, over reducing total loop count.
        for i in range(p):
            for j in range(d): ms[j] += x[i, j]
        for j in range(d): ms[j] /= tp
        for i in range(p):
            for j in range(d):
                x[i, j] = (x[i, j] - ms[j]) #/ tp

    # For large # elements, everything else has little impact on execution speed besides the np.dot call and C or d adjustment.
    innermul_cself(x,C,1./tp,sym=symmetrize)
    #np.dot(x.T, x, out=C)
    if place_back:
        if ispl and p * d >= csize:
            ld=nb.set_parallel_chunksize(mt.floor(p / GTHDS))
            for i in nb.prange(p):
                for j in range(d): x[i, j] = x[i, j] * tp + ms[j]
            nb.set_parallel_chunksize(ld)
        else:
            for i in range(p):
                for j in range(d): x[i, j] = x[i, j] * tp + ms[j]
    

@jtp
def _covar_inplace_vops3(x, C, tmp):
    #conditionals add a fair bit of perf overhead, maybe cause of kernel size.
    ispl = tmp.shape[0] > 1
    ms = tmp[0]
    p, d = x.shape
    tpt = type_ref(x) # x and tmp will have same types, but C can be another type eg f32 or even f16 if can't support more mem.
    tp = tpt(p)
    csize=10000_000_000 #250_000
    ndiv=p>d*12
    # if ndiv:
    #     def seti(d,x,ms,tp):
    #         for jj in range(d): x[jj] -= ms[jj]
    #     def seto(d,x,ms,tp):
    #         for jj in range(d): x[jj] += ms[jj]
    # else:
    #     def seti(d,x,ms,tp):
    #         for jj in range(d): x[jj] = (x[jj] - ms[jj]) / tp
    #     def seto(d,x,ms,tp):
    #         for jj in range(d): x[jj] = x[jj] * tp + ms[jj]
        
    if ispl and p*d>=csize and p*2>GTHDS:
        tmp[:] = tpt(0.)
        ld = nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        for i in nb.prange(p):
            thdd=nb.get_thread_id()
            for j in range(d):
                tmp[thdd, j] += x[i,j]
        for i in range(1, GTHDS - 1):
            for j in range(d):
                ms[j] += tmp[i, j]
        for j in range(d):
            ms[j] = (ms[j]+tmp[-1, j])/tp
        if ndiv:
            for i in nb.prange(p):
                for j in range(d): x[i,j] -= ms[j]
        else:
            for i in nb.prange(p):
                for j in range(d): x[i,j] = (x[i,j] - ms[j]) / tp
        # for i in nb.prange(p):
        #     seti(d,x[i],ms,tp)
        nb.set_parallel_chunksize(ld)
    else:
        ms[:] = x[0]
        for i in range(1, p - 1):
            for j in range(d):
                ms[j] += x[i, j]
        for j in range(d):
            ms[j]=(ms[j]+x[-1, j])/tp
        if ndiv:
            for i in range(p):
                for j in range(d): x[i,j] -= ms[j]
        else:
            for i in range(p):
                for j in range(d): x[i,j] = (x[i,j] - ms[j]) / tp

    #Overall the performance benefit of these parallel blocks is negligible, only at millions of operations does it add a little
    #benefit. This might be because the lapack call of dot for large arrays occupies enough system recourses to undermine launching parallel blocks in rapid succession. Everything else has little impact on execution speed besides the np.dot call, for large d and p.
    np.dot(x.T, x, out=C)

    #ctp = type_ref(C)
    #ttp = ctp(tp ** 2.)
    if ndiv:
        ctp=type_ref(C)
        ct = C.reshape(-1)
        ttp=ctp(tp**2.)
        if d * d >= 2**20:
            ld = nb.set_parallel_chunksize(mt.ceil(ct.size / GTHDS))
            for i in nb.prange(ct.size): ct[i] /= ttp
        else:
            for i in range(ct.size): ct[i] /= ttp

    if p * d >= csize:
        nb.set_parallel_chunksize(mt.floor(p / GTHDS))
        if ndiv:
            for i in nb.prange(p):
                for j in range(d):
                    x[i,j] += ms[j]
        else:
            for i in nb.prange(p):
                for j in range(d):
                    x[i, j] = x[i, j] * tp + ms[j]
        nb.set_parallel_chunksize(ld)
    else:
        if ndiv:
            for i in range(p):
                for j in range(d):
                    x[i,j] += ms[j]
        else:
            for i in range(p):
                for j in range(d):
                    x[i,j] = x[i,j] * tp + ms[j]
                
        # for i in range(p):
        #     seto(d,x[i],ms,tp)

def covar_benchmark():
    #crossover + factor application for the differential evolution algorithm
    bench_ver_names = [f'CovarBench {i}' for i in range(1, 5)]
    timing_run = 1000
    p=5000
    d=25
    # x  : (p,d)  – modified in place (centred, *not* re‑shifted here)
    #     C  : (d,d)  – filled with 1/p * XᵀX   (full, symmetric)
    #     _tmp : (2,d) – scratch for the mean
    x=np.random.normal(1,1,(timing_run,p,d))
    C=np.empty((d,d),dtype=np.float64)
    #C2=np.empty((d, d), dtype=np.float64)
    tmp = np.empty((GTHDS,d),dtype=np.float64)
    # 
    # covar_inplace_vops3(x[0],C,tmp)
    # covar_inplace_vops4(x[0], C2, tmp)
    # print(C[:5,:5])
    # print(C2[:5, :5])
    # print(C[:5, :5]-C2[:5, :5])
    # print(sum(sum(np.abs(C-C2))))
    # return #same calc

    def res_call():
        t.sleep(.05)
    #cr,f=.8,.7
    b1 = lambda reps: array_rep(x[:reps],covar_inplace_vops,C,tmp)
    b2 = lambda reps: array_rep(x[:reps],covar_inplace_vops2,C,tmp)
    b3 = lambda reps: array_rep(x[:reps],covar_inplace_vops3,C,tmp)
    #Can be significant faster when p*5>d but after that the other methods can tend slightly better due to cache locality.
    #regardless sryk can save a lot of time when there are many samples, and with fewer samples tradeoff sb less. (can switch too if it matters)
    b4 = lambda reps: array_rep(x[:reps],covar_inplace_vops4,C,tmp)
    #and it does appear the with the intel optimized install, calcs can be 20-30% faster than the original scipy cfuncs

    print(f'timing_run={timing_run}, p={p}, d={d}')
    time_funcs((b1,b2,b3,b4), bench_ver_names, res_call,
               compile_run=1,
               init_run=2,
               timing_run=timing_run,
               repeat_sequence=16,
               ops_mult=(p*(d+1)*d)+p*d + d*d)
    
    #print(C[:20, :20]-C2[:20, :20])
    #print(sum(sum(np.abs(C-C2))))


@jtp
def eu_pt1(x, out):
    csize = 250_000
    pt = x[max(nb.int64(x[0,0]),0)]

    def plc(ii):
        d = 0.0
        for j in range(x.shape[1]):
            d += x[ii, j] - pt[j]
        out[ii] = d ** .5

    if x.size >= csize:
        ld = nb.set_parallel_chunksize(mt.floor(x.shape[0] / 16))
        for i in nb.prange(x.shape[0]):
            plc(i)
        nb.set_parallel_chunksize(ld)
    else:
        for i in range(x.shape[0]):
            plc(i)
    
    return sum(out)

@jtp
def eu_pt2(x, out):
    csize = 250_000
    pt = x[max(nb.int64(x[x.shape[0], x.shape[1]]), 0)]

    if x.size >= csize:
        ld = nb.set_parallel_chunksize(mt.floor(x.shape[0] / 16))
        for i in nb.prange(x.shape[0]):
            d = 0.0
            for j in range(x.shape[1]):
                d += x[i, j] - pt[j]
            out[i] = d ** .5
        nb.set_parallel_chunksize(ld)
    else:
        for i in range(x.shape[0]):
            d = 0.0
            for j in range(x.shape[1]):
                d += x[i, j] - pt[j]
            out[i] = d ** .5

    return sum(out)


@jtp
def eu_pt3(x, out):
    csize = 3_000_000
    pt = x[max(nb.int64(x[0, 0]), 0)]

    def plc(ii):
        out[ii] = np.linalg.norm(x[ii] - pt)

    if x.size >= csize:
        ld = nb.set_parallel_chunksize(mt.floor(x.shape[0] / 16))
        for i in nb.prange(x.shape[0]):
            plc(i)
        nb.set_parallel_chunksize(ld)
    else:
        for i in range(x.shape[0]):
            plc(i)

    return sum(out)

def norm_benchmark():
    # crossover + factor application for the differential evolution algorithm
    bench_ver_names = [f'NormBench {i}' for i in range(1, 3)]
    timing_run = 100
    p = 1024
    d = 202
    # x  : (p,d)  – modified in place (centred, *not* re‑shifted here)
    #     C  : (d,d)  – filled with 1/p * XᵀX   (full, symmetric)
    #     _tmp : (2,d) – scratch for the mean
    x = np.random.normal(2, 2, (timing_run, p, d))
    # C2=np.empty((d, d), dtype=np.float64)
    tmp = np.empty((p,), dtype=np.float64)
    out = np.empty((timing_run,), dtype=np.float64)

    # covar_inplace(x[0],C,tmp)
    # covar_inplace_vops(x[0], C2, tmp,0)
    # print(C[:20,:20])
    # print(C2[:20, :20])
    # print(C[:20, :20]-C2[:20, :20])
    # print(sum(sum(np.abs(C-C2))))
    # return #same calc

    def res_call():
        t.sleep(.05)

    # cr,f=.8,.7
    b1 = lambda reps: arrayout_rep(x[:reps],out[:reps], eu_pt1,tmp) #no difference between 1 an 2
    b2 = lambda reps: arrayout_rep(x[:reps],out[:reps], eu_pt2,tmp)
    #b3 = lambda reps: arrayout_rep(x[:reps], out[:reps], eu_pt3, tmp)

    print(f'timing_run={timing_run}, p={p}, d={d}')
    time_funcs((b1, b2), bench_ver_names, res_call,
               compile_run=2,
               init_run=10,
               timing_run=timing_run,
               repeat_sequence=20,
               ops_mult=p*d)

@jt
def searchsorted1(j,x, s):
    xg=x[j]
    sv=s[j]
    i = 0
    n = x.shape[0]
    while i < n and xg[i] < sv:
        i += 1
    return i

@jt
def searchsorted2(j,x, s):
    xg=x[j]
    sv=s[j]
    i = 0
    while xg[i] < sv:
        i += 1
    return i

@jt
def searchsorted3(j,x, s):
    """
    Uses NumPy's binary search (searchsorted with side='left') to find
    the insertion index of x in sorted array arr.
    """
    return np.searchsorted(x[j], s[j], side='left')

def searchsorted_benchmark():
    #crossover + factor application for the differential evolution algorithm
    bench_ver_names = [f'SearchS {i}' for i in range(1, 4)]
    timing_run=p=100000*10
    d=160
    # x  : (p,d)  – modified in place (centred, *not* re‑shifted here)
    #     C  : (d,d)  – filled with 1/p * XᵀX   (full, symmetric)
    #     _tmp : (2,d) – scratch for the mean
    x=np.random.normal(1,1,(p,d))
    s=np.empty((p,),dtype=np.float64)
    rdg=np.random.randint(0,d,size=(p,))
    print(rdg.shape)
    s[:]=x[np.arange(p),rdg]
    print(s.shape)
    x.sort(axis=1)
    
    out=np.empty((p,),dtype=np.int64)

    def res_call():
        out[:]=0
        t.sleep(.05)
    #cr,f=.8,.7
    b2 = lambda reps: arrayouti_rep(out[:reps],searchsorted1,x,s)
    b1 = lambda reps: arrayouti_rep(out[:reps],searchsorted2,x,s)
    b3 = lambda reps: arrayouti_rep(out[:reps],searchsorted3,x,s)

    print(f'timing_run={timing_run}, p={p}, d={d}')
    time_funcs((b1,b2,b3,), bench_ver_names, res_call,
               compile_run=10,
               init_run=1000,
               timing_run=timing_run,
               repeat_sequence=20,
               ops_mult=d)

# Define CBLAS constants
# Type aliases for Numba types
_itt = types.intp    # native pointer-sized integer
_chtp = types.char   # single character

# CBLAS constants (wrapped in pointer-size integer type)
CBLAS_ROW_MAJOR = _itt(101)
CBLAS_COL_MAJOR = _itt(102)
CBLAS_NO_TRANS  = _itt(111)
CBLAS_TRANS     = _itt(112)
CBLAS_UPPER     = _itt(121)
CBLAS_LOWER     = _itt(122)
CBLAS_UNIT      = _itt(131)
CBLAS_NONUNIT   = _itt(132)
CBLAS_LEFT      = _itt(141)
CBLAS_RIGHT     = _itt(142)

# LAPACKE constants (encoded as character ordinals)
LAPACK_ROW_MAJOR = CBLAS_ROW_MAJOR
LAPACK_COL_MAJOR = CBLAS_COL_MAJOR
LAPACK_UPPER     = _chtp(ord('U'))
LAPACK_LOWER     = _chtp(ord('L'))
LAPACK_NO_TRANS  = _chtp(ord('N'))
LAPACK_TRANS     = _chtp(ord('T'))
LAPACK_CONJ_TRANS= _chtp(ord('C'))
LAPACK_NON_UNIT  = _chtp(ord('N'))
LAPACK_UNIT      = _chtp(ord('U'))
LAPACK_LEFT      = _chtp(ord('L'))
LAPACK_RIGHT     = _chtp(ord('R'))

MKL_ORDER_ROW_MAJOR       = _chtp(ord('R'))
MKL_ORDER_COL_MAJOR       = _chtp(ord('C'))
MKL_NOTRANS   = _chtp(ord('N'))  # 78
MKL_TRANS     = _chtp(ord('T'))  # 84
MKL_CONJTRANS = _chtp(ord('C'))  # 67
MKL_CONJ      = _chtp(ord('R'))  # 82

# layout
MKL_ORDER_ROW_MAJOR2 = _chtp(101)  # MKL_ROW_MAJOR
MKL_ORDER_COL_MAJOR2 = _chtp(102)  # MKL_COL_MAJOR

# transpose
MKL_NOTRANS2       = _chtp(111)    # MKL_NOTRANS
MKL_TRANS2         = _chtp(112)    # MKL_TRANS
MKL_CONJTRANS2     = _chtp(113)    # MKL_CONJTRANS
MKL_CONJ2          = _chtp(114)    # MKL_CONJ

itt = types.uintp
mkl_dimatcopy_sig = typing.signature(
    types.void,
    types.char,  # ordering
    types.char,  # trans
    itt,  # rows
    itt,  # cols
    types.double,  # alpha
    types.CPointer(types.double),  # AB
    itt,  # lda
    itt  # ldb
)

# External binding to cblas_dsyrk
mkl_dimatcopy = types.ExternalFunction("mkl_dimatcopy", mkl_dimatcopy_sig)
print(mkl_dimatcopy.__dict__)

itt = types.uintp
mkl_domatcopy_sig = typing.signature(
    types.void,
    types.char,  # ordering
    types.char,  # trans
    itt,  # rows
    itt,  # cols
    types.double,  # alpha
    types.CPointer(types.double),  # AB
    itt,  # lda
    itt  # ldb
)

mkl_domatcopy = types.ExternalFunction("mkl_domatcopy", mkl_domatcopy_sig)

@nb.njit(inline='always')
def _pick_lda_ldb(rows, cols, order, trans):
    if order == MKL_ORDER_COL_MAJOR:                         # column-major
        lda = max(1, rows)
        if trans in (MKL_TRANS, MKL_CONJTRANS):
            ldb = max(1, cols)
        else:                                          # N / R
            ldb = max(1, rows)
    else:                                              # row-major
        lda = max(1, cols)
        if trans in (MKL_TRANS, MKL_CONJTRANS):
            ldb = max(1, rows)
        else:                                          # N / R
            ldb = max(1, cols)

    # ---- in-place safe guard -------------------------------------------
    # If we do an in-place transpose we *must* keep lda == ldb,
    # otherwise MKL treats it as out-of-place and writes past the end.
    if lda != ldb:
        ldb = lda
    return lda, ldb

@jt
def imatcopy(a, alpha=1.0,
             order=MKL_ORDER_ROW_MAJOR,
             trans=MKL_TRANS):
    rows, cols = a.shape
    lda, ldb   = _pick_lda_ldb(rows, cols, order, trans)
    print(rows,cols)
    print(lda,ldb)
    mkl_dimatcopy(order, trans,
                  rows, cols,
                  alpha,
                  a.reshape(-1).ctypes,       # numba lowers this to the raw ptr
                  lda, ldb)

@jt
def copyf1(a):
    """In-place C→Fortran transpose using MKL."""
    imatcopy(a, order=MKL_ORDER_ROW_MAJOR, trans=MKL_NOTRANS)

@jt
def copyf2(a, trr):
    """Your original scratch-buffer version (kept for comparison)."""
    trr[:] = a
    a_T_F  = a.reshape(a.shape[::-1]).T   # Fortran-contiguous view
    a_T_F[:] = trr                        # copy back



    
def toff_benchmark():
    bench_ver_names = [f'CRFApp {i}' for i in range(1, 4)]
    timing_run = 500
    m=5
    n=5
    
    x=np.random.normal(0.,1.,(m,n))
    print(x.dtype)
    #x[1]=x[0]
    tmp=np.empty((m,n),dtype=np.float64)

    def res_call():
        t.sleep(.05)
    
    try:
        copyf1(x)
        print(x[:10,:10])
    except:
        print('failed imat1')
    try:
        copyf2(x)
        print(x[:10,:10])
    except:
        print('failed imat2')
    #copytrans2(x[1],tmp)
    print(x[ :10, :10])
    print(np.sum(np.abs(x[0]-x[1])))
    
    return
    
    b1 = lambda reps: array_rep(x[:reps],generic_cr_f_apply1,cr,f)
    b2 = lambda reps: array_rep(x[:reps],generic_cr_f_apply2,cr,f)
    b3 = lambda reps: array_rep(x[:reps], generic_cr_f_apply3, cr, f)

    print(f'timing_run={timing_run}, psz={psz}, cr={cr}, f={f}')
    time_funcs((b1,b2,b3), bench_ver_names, res_call,
               compile_run=1000,
               init_run=100_000,
               timing_run=timing_run,
               repeat_sequence=10,
               ops_mult=m*n)


p4o=np.array([1.000549459621945, -1.628487578121468, 2.486243732743884, -13.570479991819441, 0.017894429239572, 0.207331149646325, 0.000042785084111, 22.72573914371964, -0.338998884352377, 0.013561147304741, 0.000035858284938, -1.792855924513853, 2.085934258154098, -11.339740516942792, -0.263394466232209, 0.789626406029434, 0.002740111828052, 20.903260486632245, -0.786224682733048, -0.002451118780528, -0.00001488653939])#21 param rational, from a static/immutable array which the LLVM will be aware of as read only. can add a 1 for the separate coefficient so that it evals faster as well.

@jt
def tmv1(z, nu,):
    p = z * z
    r = 1.0 / nu

    num = p4o[0] \
          + p4o[1] * r + p4o[2] * (r * r) + p4o[3] * (r * r * r) \
          + p4o[4] * p * r + p4o[5] * p * (r * r) + p4o[6] * (p * p) * r \
          + p4o[7] * (r * r * r * r) \
          + p4o[8] * p * (r * r * r) + p4o[9] * (p * p) * (r * r) + p4o[10] * (p * p * p) * r

    den = 1.0 \
          + p4o[11] * r + p4o[12] * (r * r) + p4o[13] * (r * r * r) \
          + p4o[14] * p * r + p4o[15] * p * (r * r) + p4o[16] * (p * p) * r \
          + p4o[17] * (r * r * r * r) \
          + p4o[18] * p * (r * r * r) + p4o[19] * (p * p) * (r * r) + p4o[20] * (p * p * p) * r

    return z * (num / den)

p4o_k1=np.array([22.72573914371964, -13.570479991819441, -0.338998884352377, 2.486243732743884, 0.207331149646325, 0.013561147304741, -1.628487578121468, 0.017894429239572, 0.000042785084111, 0.000035858284938, 1.000549459621945, 20.903260486632245, -11.339740516942792, -0.786224682733048, 2.085934258154098, 0.789626406029434, -0.002451118780528, -1.792855924513853, -0.263394466232209, 0.002740111828052, -0.00001488653939, 1.])

@jt
def tmv1_k1(z: float, nu: float) -> float:
    p = z*z
    r = 1.0/nu
    # Numerator
    C4 = p4o_k1[0]
    C3 = p4o_k1[1] + p * p4o_k1[2]
    C2 = p4o_k1[3] + p * (p4o_k1[4] + p * p4o_k1[5])
    C1 = p4o_k1[6] + p * (p4o_k1[7] + p * (p4o_k1[8] + p * p4o_k1[9]))
    C0 = p4o_k1[10]
    num = (((C4 * r + C3) * r + C2) * r + C1) * r + C0
    # Denominator (r^4 present, no r^5)
    D4 = p4o_k1[11]
    D3 = p4o_k1[12] + p * p4o_k1[13]
    D2 = p4o_k1[14] + p * (p4o_k1[15] + p * p4o_k1[16])
    D1 = p4o_k1[17] + p * (p4o_k1[18] + p * (p4o_k1[19] + p * p4o_k1[20]))
    D0 = p4o_k1[21]
    den = (((D4 * r + D3) * r + D2) * r + D1) * r + D0
    return z * (num / den)

p4o_t1=(22.72573914371964, -13.570479991819441, -0.338998884352377, 2.486243732743884, 0.207331149646325, 0.013561147304741, -1.628487578121468, 0.017894429239572, 0.000042785084111, 0.000035858284938, 1.000549459621945, 20.903260486632245, -11.339740516942792, -0.786224682733048, 2.085934258154098, 0.789626406029434, -0.002451118780528, -1.792855924513853, -0.263394466232209, 0.002740111828052, -0.00001488653939, 1.)

@jt
def tmv1_t1(z: float, nu: float) -> float:
    p = z*z #same thing but it's a tuple
    r = 1.0/nu
    # Numerator
    C4 = p4o_t1[0]
    C3 = p4o_t1[1] + p * p4o_t1[2]
    C2 = p4o_t1[3] + p * (p4o_t1[4] + p * p4o_t1[5])
    C1 = p4o_t1[6] + p * (p4o_t1[7] + p * (p4o_t1[8] + p * p4o_t1[9]))
    C0 = p4o_t1[10]
    num = (((C4 * r + C3) * r + C2) * r + C1) * r + C0
    # Denominator (r^4 present, no r^5)
    D4 = p4o_t1[11]
    D3 = p4o_t1[12] + p * p4o_t1[13]
    D2 = p4o_t1[14] + p * (p4o_t1[15] + p * p4o_t1[16])
    D1 = p4o_t1[17] + p * (p4o_t1[18] + p * (p4o_t1[19] + p * p4o_t1[20]))
    D0 = p4o_t1[21]
    den = (((D4 * r + D3) * r + D2) * r + D1) * r + D0
    return z * (num / den)


_2ri= 1./mt.sqrt(2.0)

@jt
def _fasttv1_k(z_sig,dof,v1model):
    az=abs(z_sig)
    if dof < 3.:
        if az > 8.2590072: return 0. #erf will round to zero after this range with f64
        err=mt.erf(z_sig *_2ri)
        if dof<2.:
            return mt.tan((mt.pi / 2.0)*err)
        e=err*err
        den = 1.0 - e
        return mt.copysign(mt.sqrt(2.*e / den), err)

    #then poly
    p4,p3,p2,p1=0.004491152418700, 250.1508589461631, 0.243789262458383, 250.0952851341867
    if dof>p1:
        if dof>p1+az*(p2+az*(p3 + az*p4)): return z_sig
    
    return mt.copysign(v1model(az,dof),z_sig)

@jt
def fastt_v1(zdf):
    tk=0. #to make it less about writing arrays.
    for i in range(zdf.shape[0]):
        tk+= _fasttv1_k(zdf[i,0],zdf[i,1],tmv1)
    return tk

@jt
def fastt_v1k1(zdf):
    tk=0. #to make it less about writing arrays.
    for i in range(zdf.shape[0]):
        tk+= _fasttv1_k(zdf[i,0],zdf[i,1],tmv1_k1)
        
    return tk


@jt
def fastt_v1t1(zdf):
    tk = 0.  # to make it less about writing arrays.
    for i in range(zdf.shape[0]):
        tk += _fasttv1_k(zdf[i, 0], zdf[i, 1], tmv1_t1)

    return tk


def tmodel_benchmark():
    # crossover + factor application for the differential evolution algorithm
    bench_ver_names = [f'T-Estimator {i}' for i in range(1, 3)]
    timing_run = 10
    p = 400000
    # x  : (p,d)  – modified in place (centred, *not* re‑shifted here)
    #     C  : (d,d)  – filled with 1/p * XᵀX   (full, symmetric)
    #     _tmp : (2,d) – scratch for the mean
    z = np.random.uniform(.01, 6, (timing_run, p,))
    v = np.random.uniform(mt.log(3.1), mt.log(8000), (timing_run, p,)) #no erf call
    #v = np.random.uniform(1., 8000, (timing_run, p,)) #to test the model more than anything else
    zdf=np.empty((timing_run,p,2), dtype=np.float64)
    zdf[:,:,0]=z
    zdf[:, :, 1] = np.exp(v)
    # C2=np.empty((d, d), dtype=np.float64)
    out = np.empty((timing_run,), dtype=np.float64)

    # covar_inplace(x[0],C,tmp)
    # covar_inplace_vops(x[0], C2, tmp,0)
    # print(C[:20,:20])
    # print(C2[:20, :20])
    # print(C[:20, :20]-C2[:20, :20])
    # print(sum(sum(np.abs(C-C2))))
    # return #same calc

    def res_call():
        t.sleep(.05)

    # cr,f=.8,.7
    #reps=timing_run
    #this seems to be 
    b2 = lambda reps: arrayout_rep(zdf[:reps], out[:reps], fastt_v1k1,) #seems to win in median, so maybe better for singular calls.
    b1 = lambda reps: arrayout_rep(zdf[:reps], out[:reps], fastt_v1t1, ) #no difference keep as array or as tuple in numba, for faster numpy keep as tuple
    #b1 = lambda reps: arrayout_rep(zdf[:reps],out[:reps], fastt_v1,) #Almost no difference
    #b3 = lambda reps: arrayout_rep(x[:reps], out[:reps], eu_pt3, tmp)

    print(f'timing_run={timing_run}, p={p}')
    time_funcs((b1, b2), bench_ver_names, res_call,
               compile_run=2,
               init_run=10,
               timing_run=timing_run,
               repeat_sequence=100,
               ops_mult=p)

jti = nb.njit(fastmath=True, error_model='numpy',inline='always')

@jti
def axpy(dst: np.ndarray, a: float, src: np.ndarray):
    r"""Add product of y to x: $x \leftarrow a \cdot y + x$"""
    n = src.shape[0]
    for i in range(n):
        dst[i] += a * src[i]
    return dst

@jti
def cxpy(dst: np.ndarray, a: float, src: np.ndarray):
    r"""Add product of y to x: $x \leftarrow a \cdot y + x$"""
    n = src.shape[0]
    for i in range(n):
        dst[i] = a * src[i]
    return dst

@jti
def axfus(x,a,b):
    for i in range(x.shape[1]):
        x[0,i]=a*x[1,i]+b*x[2,i]

@jti
def axfus2(x,a,b):
    to=0.
    for i in range(x.shape[1]):
        t=x[0,i]+a*x[1,i]+b*x[2,i]
        to+=t*t
        x[0,i]=t
    x[0,-1]=to/1000000.
    
@jti
def axfus3(x,a,b):
    #to=0.
    for i in range(x.shape[1]):
        #t=x[0,i]
        x[1, i]=a*x[0,i]
        x[2, i]= b * x[0,i]

@jti
def axsec(x,a,b):
    #axpy(x[0],a,x[1])
    cxpy(x[0], a, x[1])
    axpy(x[0], b, x[2])
    
@jti
def axsec2(x,a,b):
    #axpy(x[0],a,x[1])
    cxpy(x[1], a, x[0])
    cxpy(x[2], b, x[0])

def double_axpytest():
    bench_ver_names = [f'Double Axpy {i}' for i in range(1, 3)]
    arrsz = 50_000
    ireps=20_000
    x=np.random.normal(0,1,(3,arrsz,))
    x1=np.empty((3,arrsz,))
    a=.2
    b=-.1
    def res_call():
        x1[:]=x
        t.sleep(.05)

    b1 = lambda reps: rep_run(reps,axfus,x1,a,b,)
    b2 = lambda reps: rep_run(reps,axsec,x1,a,b,)
    
    # b1 = lambda reps: arrayout_rep(reps,axfus3,x1,a,b,)
    # b2 = lambda reps: rep_run(reps,axsec2,x1,a,b,)

    print(f'timing_run={ireps} arrsz={arrsz}')
    time_funcs((b1,b2), bench_ver_names, res_call,
               compile_run=10,
               init_run=100,
               timing_run=ireps,
               repeat_sequence=10,
               ops_mult=arrsz)


if __name__ == '__main__':
    #looptype_benchmark()
    #recordvarray_benchmark()
    #numpyvectorizedvsloop_benchmark()
    #numpyvectorizedvsloop2_benchmark()
    #numpyvnative_randint_benchmark()
    #f64vf32_benchmark()
    #roundi_benchmark()
    
    #Not covered tests:
    #aloopfusion_benchmark()
    #outerparallel_benchmark()
    #arangei_benchmark()
    #argmin_benchmark()
    #crfapp_benchmark()
    #covar_benchmark()
    #toff_benchmark()
    #norm_benchmark()
    #searchsorted_benchmark()
    #tmodel_benchmark()
    double_axpytest()
