import numpy as np
import numba as nb
import math as mt

jt_pl = nb.njit(fastmath=True, error_model='numpy',parallel=True)

@jt_pl
def grid_eval_exec(bounds:np.ndarray|tuple,fitness:np.ndarray|int,eval_op)->tuple[np.ndarray,np.float32,np.float32]: #grid eval adjusted to widths of bounds.
    '''
    The executor for grid evaluation of the numba func.
    :param bounds: 1d array that is tot_dims*2, needs to be flat so it works in the parallel region.
    :param fitness: A flat array where output is placed.
    :param eval_op: Function to evaluate as an operator tuple -> (eval_func, *eval_config_args). Applied in numba like eval_op[0](x,*eval_op[1:])
    :return: 
    '''
    #Init size info:
    tot_pts=fitness.shape[0]
    tot_dims=len(bounds)//2
    dim_pts=nb.int64((tot_pts+1)**(1/tot_dims))
    nt=nb.get_num_threads()
    
    #init processing memory
    _cm=np.empty((nt,tot_dims),dtype=np.float64)
    _dc =np.empty((tot_dims,),dtype=np.int64)
    _dc[0]=1
    _mmx = np.empty((nt,2), dtype=np.float32)
    _mmx[:,0]=np.inf
    _mmx[:, 1] = -np.inf
    for d in range(1,tot_dims):
        _dc[d]=dim_pts ** d
    
    ld = nb.set_parallel_chunksize(mt.ceil(tot_pts/ nt))
    for i in nb.prange(0,tot_pts):
        tid=nb.get_thread_id()
        for d in range(0,tot_dims):
            # Compute the index along dimension `d`
            idx = (i // _dc[d]) % dim_pts
            # Map the index to the actual coordinate in dimension `d`
            _cm[tid, d] = bounds[d*2] + (idx / (dim_pts - 1)) * (bounds[d*2 +1] - bounds[d*2])
        fitness[i]=eval_op[0](_cm[tid],*eval_op[1:])
        if _mmx[tid,0]>fitness[i]:
            _mmx[tid, 0] = fitness[i]
        elif _mmx[tid,1]<fitness[i]:
            _mmx[tid, 1] = fitness[i]
    nb.set_parallel_chunksize(ld)
    
    mn=_mmx[:,0].min()
    mx=_mmx[:,1].max()

    return fitness,mn,mx

def grid_eval(eval_op,bounds,num_points=2**15,return_with_dims=True):
    bounds=np.array(bounds,np.float64) #for easy reshaping and indexing.
    dims=bounds.shape[0] #get dims for equal # pts per dim
    bounds=tuple(bounds.reshape(-1)) # flatten to tuple 1d so works in parallel region of exec.
    
    dim_pts = int((num_points + 1) ** (1 / dims))
    num_points=dim_pts**dims
    fitness=np.empty((num_points,),dtype=np.float32)
    fitness,mn,mx=grid_eval_exec(bounds,fitness,eval_op)
    if return_with_dims:
        fitness=fitness.reshape((dim_pts,)*dims)
    return fitness,mn,mx
