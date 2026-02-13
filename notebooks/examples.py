import numba as nb
import numpy as np
#from nevergrad.benchmark.experiments import parallel

### --- 1: OPERATOR STRATEGY
jt=nb.njit(fastmath=True, error_model='numpy')

@jt
def customprocessor_ver1(x,eval_func,stop_func,monitor_func=None):
    
    #Static processor settings
    a=None
    b=None
    
    #Processor memory, like arrays or local vars
    c=None
    d=None 
    
    #first eval
    f = eval_func(x)
    
    while not stop_func(f,x):
        #Put next x logic before monitor if we want to track before next eval.
        #--- choose new x
        if monitor_func is not None: monitor_func(f,x,c,d)
        #Or put next x logic here, for after eval tracking.
        #--- choose new x
        
        f = eval_func(x)

    #Optionally return result and final x vector
    #But the monitor can handle this.
    return f, x

@jt 
def eval2(x,*args):
    return 0

@jt 
def monitor2(f, x, c, d,*args):
    print(f,x,c,d,*args)
    return 0

@jt 
def stop2(f, x,*args):
    return False

@jt 
def customprocessor_ver2(x, eval_func, stop_func, monitor_func=None,*args):
    #if we type monitor func we now lose the ability to have kwargs if we want to have *args
    a = None
    b = None
    
    c = None
    d = None
    f = eval_func(x,*args)

    while not stop_func(f, x,*args):
        # -- choose new x here
        if monitor_func is not None: monitor_func(f, x, c, d,*args)
        # -- or here
        f = eval_func(x,*args)

    return f, x

@jt 
def stop2v2(f, x, ct_arr:np.ndarray, f_tracker:np.ndarray, *args):
    #now we assume f is improving when it decreases
    if ct_arr[0]==0:
        f_tracker[0]=f
        f_tracker[1]=f
    
    ct_arr[0]+=1
    #long ema
    f_tracker[0]*=.999
    f_tracker[0]+=f*.001
    
    #short ema
    f_tracker[1]*=.985
    f_tracker[1]+=f*.015
    
    #if short ema crosses above long ema > crit, stop
    crit=.05
    if (f_tracker[0]-f_tracker[1])>crit:
        return True
    else:
        return False

@jt 
def monitor2v2(f, x, c, d,ct_arr,f_tracker,*args):
    if ct_arr[0]%10==0:
        print(f,x,c,d,ct_arr,f_tracker,*args)


@jt 
def stop2v3(f, x, ct_arr:np.ndarray, f_tracker:np.ndarray,x_loc,dist_min, *args):
    nm=0.0
    for i in range(x.shape[0]):nm+=(x[i]-x_loc[i])**2.
    if (nm**.5)<dist_min: return True
    #imagine rest of 2v2 below

@jt 
def monitor2v3(f, x, c, d,ct_arr,f_tracker,x_loc,dist_min,*args):
    #...
    pass

@jt 
def stop3(f, x):
    pass

@jt 
def monitor3(f, x, c, d):
    pass

@jt
def customprocessor_ver3(x, 
                         eval_func, 
                         stop_func=stop3,
                         eval_args=(),
                         stop_args=(), 
                         monitor_func=monitor3,
                         monitor_args=()):

    a,b,c,d=None,None,None,None
    f = eval_func(x, *eval_args)

    while not stop_func(f, x, *stop_args):
        # -- choose new x here
        if monitor_func is not None: monitor_func(f, x, c, d, *monitor_args)
        # -- or here
        f = eval_func(x, *eval_args)

    return f, x

@jt
def customprocessor_ver4(x, 
                         eval_op, 
                         stop_op=(stop3,),
                         monitor_op=(monitor3,)):
    a,b,c,d=None,None,None,None
    #Numba treats tuples that contain different types like constant pointers in C: "&"
    #Can also think of them as being unrolled as anonymous signatures at compile time.
    #So there should be no performance difference between tuples used in this manner
    #and function parameter arguments.
    f = eval_op[0](x, *eval_op[1:])
    while not stop_op[0](f, x, *stop_op[1:]):
        
        if monitor_op is not None: monitor_op[0](f, x, c, d, *monitor_op[1:])
        f = eval_op[0](x, *eval_op[1:])
    return f, x


@jt
def customprocessor_ver5(x,
                         eval_op,
                         stop_op=(stop3,),
                         monitor_op=(monitor3,)):
    a, b, c, d = None, None, None, None
    f = eval_op[0](*eval_op[1:],x)
    while not stop_op[0](*stop_op[1:],f, x):
        #only difference is that input parameters are at the end.
        if monitor_op is not None: monitor_op[0](*monitor_op[1:],f, x, c, d)
        f = eval_op[0](*eval_op[1:],x)
    return f, x

# For example see aopt

### --- 2: LOCAL SCOPED FUNCTIONS
#These will just be examples from my existing works.

### Example 1 Sampler
import math as mt
import random as rand

jtp=nb.njit(fastmath=True, error_model='numpy',parallel=True)
_PL_THREADS=nb.get_num_threads()

@jtp
def uellipsoid_pbound_sampler(x:np.ndarray,c:np.ndarray|float,lb,ub,clip=4.,pl=True):
    ps=x.shape[0]
    pd=x.shape[1]
    def _upd(i):
        cm = 0.0
        for n in range(pd):
            v = rand.gauss(0., 1.)
            x[i, n] = v
            cm += v * v
        nw = (pd/cm) ** .5
        for n in range(pd):
        # reflecting over axis shouldn't change ellipsoid normality, but allow sampling to occur right up to the boundary.
            if x[i,n]>0.:
                rv=min(nw*x[i,n],clip)*c[n]
                x[i, n] = x[0, n]
                if (x[0, n]+rv)>ub[n]:
                    rv=-rv
            else:
                rv = max(nw * x[i, n], -clip) * c[n]
                x[i, n] = x[0, n]
                if (x[0, n] + rv) < lb[n]:
                    rv = -rv
            x[i, n]+=rv            
    if pl:
        ld = nb.set_parallel_chunksize(mt.floor((ps - 1) / _PL_THREADS))
        for i in nb.prange(1, ps):
            _upd(i)
        nb.set_parallel_chunksize(ld)
    else:
        for i in range(1,ps):
            _upd(i)


### Example 2 Vector Projection

@jt
def v1p1_grad_paramproj(x_tu_p: nb.types.Array(np.float32, 1, 'C', aligned=True),  # assuming f32
                        s_tu_p: nb.types.Array(np.float64, 1, 'C', readonly=True, aligned=True),
                        typ: tuple | np.ndarray,
                        lb=None, ub=None,  # enforce search space bounds if not already.

                        ):
    _lnsr = log(1. / 7.4)
    tempsoft = stack_empty(20, (20,), dtype=np.float64)

    def vl2_p(i):
        t = typ[i]
        if ub is not None:
            xv = min(max(s_tu_p[i], lb[i]), ub[i])
        else:
            xv = s_tu_p[i]
        match t:
            case 0 | 5:
                # 0 no transform.
                return xv
            case 4:
                # 4 log trans
                return log(xv)
            case 1 | 6:
                # exp trans.
                return exp(xv)
            case 3:
                # log time to ema alpha.
                return exp(_lnsr / exp(xv))
        return xv
            # case 2:
            #     pass

    if ub is not None:
        for i in range(10): tempsoft[i] = min(max(s_tu_p[i], lb[i]), ub[i])
        softmax_x(tempsoft[:10], x_tu_p[:10], tempsoft[10:])
        for i in range(10, 15): tempsoft[i - 10] = min(max(s_tu_p[i], lb[i]), ub[i])
        softmax_x(tempsoft[:5], x_tu_p[10:15], tempsoft[5:])
    else:
        softmax_x(s_tu_p[:10], x_tu_p[:10], tempsoft)
        softmax_x(s_tu_p[10:15], x_tu_p[10:15], tempsoft)
    
    # Removed ops to cut down on code size
    # ...

    x_tu_p[42] = v1 = vl2_p(40)
    v2 = vl2_p(41)
    x_tu_p[43] = v1 / v2
    x_tu_p[44] = 1. - tanh(v1) * .5

    x_tu_p[45] = vl2_p(42)
    x_tu_p[46] = vl2_p(43)
    x_tu_p[47] = vl2_p(44)
    x_tu_p[48] = vl2_p(45)
    x_tu_p[49] = s_tu_p[46] * s_tu_p[42] * -_lnsr
    x_tu_p[50] = s_tu_p[47] * s_tu_p[42] * -_lnsr
    x_tu_p[51] = s_tu_p[48] * s_tu_p[42] * -_lnsr
    
    #...

    for d in range(58, 202):
        x_tu_p[d + 4] = vl2_p(d)


### --- 2: IMPLICIT INDEXING W/ OVERLOADS

# Situation Demo, Basic example.

#ChatGPTs implementation of v 'clean' implementation for flexible operators.
