

def lars1_constraintsolve_dev(A, y,out,
                          At,T1,T2,T3,C,I,Ib, #required memory.
                          eps=1e-10, verbose=False,mxitrs=-1,l2cond=-1.):
    """
    For more efficient memory usage we can assume m<=n always. and m s/b >= 2.
    List‑free 1‑add LARS / homotopy algorithm (basis pursuit).
    
    for gradient solution m is # unique samples, n is gradient dimensions.

            Parameters
    ----------
    A : ndarray of shape (m, n)
        Measurement matrix (m <= n).
    y : ndarray of shape (m,)
        Observed measurements.
    out : ndarray of shape (n,)
        Solution vector.
    eps : float
        Stop if the residual norm ||y - A x||_2 < eps, or if lambda < ~1e-14.
    max_iter : int
        Maximum lars steps.
    verbose : bool
        If True, prints iteration details each step.
    """
      
    
    #older notes on memory dependency.
    #x perm dim n, accumulates the returned vector. can be individual.
    #gp 1: r/v buffer dim m
    #c buffer dim n
    # active_mask dim n but dtype = bool (or maybe int if quicker for indexing mask... or loop it)
    
    #at_i at most (m,n), but n will be shrunk so reshape first
    #G, sqr I reshaped subindexed buffer, at most n x n.
    
    #gp 2: d_full/s_I dim n reusable, separate from G as s_i is needed for it's use.
    
    #gp 1 and 2 can overlap..
    
    #d_I at most dim n, separate from gp 1/2 but maybe can be used in c.. about to see
    #can use G[0]
    
    #aiv dim n can be gp 1 2
    
    #dim_max is actually <= m always.
    #seems I cant avoid making c, which is another size n array.
    ctp=type_ref(T1) #calc type
    dim_max = At.shape[0]  # dim_max is how many dimensions remain significant in the allocated memory, the solution can still have various coefficients.
    m, n = A.shape
    if mxitrs == -1:
        if dim_max<m:
            mxitrs=m*.8#(.5**.5) #absolutely zero idea why this is v good stop for memory defficient solvers. even if m>n
            #if it goes >n-1 it explodes so...
        else:
            mxitrs=max(m,dim_max)
    tol = prim_info(ctp,2)*2.
    if l2cond==-1.:
        l2cond=tol*64.
    x = out #size n
    #x[:]=0. ASSUME USER SETS TO ZEROS
    np.dot(A.T, y, out=C)
    def maxabs_c():
        lam = ctp(0.0)
        for i in range(n):
            bc=abs(C[i])
            if lam<bc: lam=bc
        return lam

    lam=maxabs_c()
    #print('lam',lam)
    Ib[:]=False
    dtl= lam - tol
    atdx=0
    # initial placement, seems like this can often start of > 1
    for j in range(n):
        if abs(C[j])>=dtl and atdx<=dim_max:
            #print('kk',j,C[j])
            I[atdx]=j
            Ib[j]=True
            #T2[atdx]=math.copysign(C[j],1.)
            atdx+=1
    
    if atdx==0: #shouldn't be possible.
        return x
    
    for j in range(m):
        for i in range(atdx): #atdx ~ dimension reference n
            At[i,j]=A[j,I[i]]

    ctrs=0#atdx-1
    while True:
        # --- direction on active set
        Gt=T1[:atdx*atdx].reshape((atdx,atdx))
        S2 = T2[:atdx]
        for i in range(atdx):
            S2[i]=math.copysign(1.,C[I[i]])
        #T2 occupied.
        #print(I)
        #print('As',At[:atdx])
        outermul_cself(At[:atdx],Gt,sym=False) #At perm mem
        #T1 occupied.
        #print('b4',Gt)
        #print(epsr*sqr_trace(Gt)/atdx)
        sqr_dadd(Gt,l2cond*sqr_trace(Gt)/atdx) #conditioner so cholesky shouldn't ever blow up. Though we might need more than tll**.9...
        #print('G',Gt[:10,:10])
        #print('signs',S2)
        cholesky_fsolve_inplace(Gt,S2,fo.LAPACK_LOWER)
        #print('I',I[:atdx])
        #print('s2',S2)
        #S2 contains our solution. T2 still occupied.
        #T1 free.
        Ast=T1[:atdx*m].reshape((m,atdx)) #A[:atdx].T which is v view will be non-contiguous, make dot allocate v heap temp buffer, so we will copy it to T1 instead
        for i in range(atdx):
            for j in range(m):
                Ast[j,i]=At[i,j]
        #T1 occupied.
        v = np.dot(Ast,S2,out=T3) # v : T3 size m
        #T3 occupied.
        #T1 free.
        denr=np.dot(A.T, v, out=T1[-n:]) #size n im kinda sure.
        #T1 occupied.
        #T3 free.
        
        # --- update greedy magnitude, find next candidate.
        y_star=cfg.MAXF64
        nidx=-1
        for i in range(n):
            if not Ib[i]:
                denom1 = 1 - denr[i]
                denom2 = 1 + denr[i]
                if abs(denom1) > tol:
                    num1 = lam - C[i]
                    y1 = num1 / denom1
                    if y1 > tol and y1 < y_star:
                        y_star = y1
                        nidx=i
                if abs(denom2) > tol:
                    num2 = lam + C[i]
                    y2 = num2 / denom2
                    if y2 > tol and y2 < y_star:
                        y_star = y2
                        nidx = i
        #C free.
        #T1 free.
        # --- update x along S2
        if nidx == -1: y_star=tol
        for i in range(atdx):
            x[I[i]] += y_star * S2[i]
        #T2 free.
        
        # if nidx!=-1:
        #     # Update I
        #     I[atdx]=nidx
        #     atdx+=1
        #     Ib[nidx]=True
        # --- refresh residual, correlations, λ
        r=np.dot(A,x,out=T3) # size m
        #T3 occupied.
        rn=0.0
        for j in range(m):
            rv=y[j]-r[j]
            rn+=rv*rv
            r[j]=rv
        rn=rn**.5 
        np.dot(A.T, r, out=C) # size n
        #T3 freed.
        #C occupied.
        lam = maxabs_c()
        ctrs+=1
        
        if verbose:
            print(ctrs,'|I|=', atdx,
                  'γ=', y_star,
                  'idx=', nidx,
                  '||r||_2=', rn,
                  'λ=', lam)
        
        if ctrs>=mxitrs or rn < eps or lam < tol:
            break
        elif atdx>=dim_max and nidx != -1:
            #sort of failed experiment, if you need significantly more samples than necessary for expected sparsity this can reduce mem requirements by at most 50%.
            #but the much more obvious solution is to reduce the # of samples to match dimension # expectation.
            mnxidx=argminabs(x[I[:atdx]])
            x[mnxidx]=ctp(0.)
            ii=rfind_arg(I[:atdx],mnxidx,True)
            At[ii] = A.T[nidx]
            odx=I[ii]
            I[ii] = nidx
            Ib[odx] = False
            Ib[nidx] = True
        elif nidx != -1:
            #if continue now we add the new candidate.
            At[atdx]=A.T[nidx]
            I[atdx]=nidx
            atdx+=1
            Ib[nidx]=True
            #print(A.T[nidx],nidx)
        else:
             print('NIDX negative') #shouldnt be able to get here under normal conditions.
        

        # if atdx>2:
        #     break

    return x
