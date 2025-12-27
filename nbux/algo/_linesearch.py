import nbux._utils as nbu
from ..op._misc import quadratic_newton_coef
import math as mt
import numpy as np
import numba as nb


#For now I'm implementing the safe/bracketing methods with just this stopping criteria: when our point becomes "stationary enough". This is because for a complicated function or high order poly, the point can significantly impact the precision of the function output. However because all of these methods utilize some form of bisection fallback, we at least have a guarantee it will reach a stationary point. Depending on the optimizer there is some risk of premature stopping, but that should be extremely rare. That also means er_tol ~ root_point*(float64 epsilon). May implement a more refined stopping api in the future.

#nbu is my lib that extends numba in helpful ways.
#math native calls almost always have faster implements than the numpy equivalent, for numba scalar ops. eg mt.copysign. type casting not difference tho.
@nbu.rgi
def not0_bisect(f_op,lo,hi,max_iters=200,side=1,typ=np.float64):
    """A method that bisects any function by: not zero on left side and zero on right side, or vice versa if side=-1. Returns the value just before the zero boundary."""
    ict=np.int64(max_iters)
    lam = 0.5 * (lo + hi)
    _0=typ(0)
    while ict>0:
        f = nbu.op_call_args(f_op, lam)
        if side==1:
            if f == _0:hi = lam
            else:lo = lam
        else:
            if f == _0:lo = lam
            else:hi = lam
        lam = 0.5 * (lo + hi)
        ict-=1
    return lo if side==1 else hi

@nbu.rgi
def root_bisect(f_op,lo,hi,max_iters=60,typ=np.float64,er_tol = 1e-14):
    """Root bisect, for unstable/discrete-ish roots."""
    ict=np.int64(max_iters)
    lam = 0.5 * (lo + hi)
    _0=typ(0)
    while ict>0:
        f = nbu.op_call_args(f_op, lam)
        if f == _0:hi = lam
        else:lo = lam
        plam=lam
        lam = 0.5 * (lo + hi)
        if abs(plam - lam) < er_tol: break #should know exactly where and when it stops, so no need to send a reason.
        ict-=1
    return lo, hi
        
        

@nbu.rgi #register jittable with inlining, in interpreter runs as py func, but compiles with inlining when referenced in jit decorated scope.
def _bracketed_newton(fd_op, lo, hi, er_tol=1e-14, max_iters=20,sign=1):
    """Newton step with fallback to generic bisection if it steps out of bounds. This is a root finding algorithm. fd_op returns f, fp, value and value gradient.
    
    Variable calculations are all f64.
    
    :param fd_op: Can be a function or a function operator (tuple). It recieves a single scalar value for the point estimate.
    :param sign: =1 we expect to have f(lo)<f(root)<f(hi). if -1 we expect f(lo)>f(root)>f(hi). If this expectation is unknown, it controls the bracketing bias eg if f is all positives then the bracket will halve from left to right until reaching hi, if negatives vice versa. Note: both sides wrong could make convergence messy. However it can still work with a high likelihood if we only know the sign of either lo or hi, and that our target root has a root on the opposite sign side.
    
    Notes: Convergence is only guaranteed when there is a single root within the bracket.
    """
    #while loops are faster with premature breaks in numba.
    ict=np.int64(max_iters)
    lam = 0.5 * (lo + hi)
    while ict>0:
        f,fp  = nbu.op_call_args(fd_op,lam) #either its a function or a tuple with the callable in the first element and it's input args for the rest, this will place lam as the first argument in the function instead of needing fd_op[0](lam,*fd_op[1:])
        #f=mt.copysign(f,sign)
        if mt.copysign(f,sign) > 0:
            hi = lam
        else:
            lo = lam
            
        plam=lam
        # Newton proposal from current lam
        if -1e-24<fp<1e-24:lam = 0.5*(lo + hi)
        else:
            lam = lam - f/fp #I think we need f and fp to change signs which ends up not changing them..
            if not (lo< lam < hi):lam = 0.5*(lo + hi)
        
        if abs(plam-lam)<er_tol:break
        
        ict-=1

    return lam



@nbu.jt
def _bracketed_secant(f_op, lo, hi, er_tol=1e-14, max_iters=20,sign=1):
    """
    An older version of signedroot_secant, keeping around as it's simple.
    Basically the same thing as bracketed newton. f_op returns f. However the secant points use the two most recent points, instead of the updated lo hi brackets, this typically gets the most out of the secant method, while still guaranteeing convergence with bisection bracketing, we could make it more greedy as well by using increments >1/2 or <1/2.

    Variable calculations are all f64.

    :param f_op: Can be a function or a function operator (tuple). It receives a single scalar value for the point estimate.
    :param sign: =1 we expect to have f(lo)<f(root)<f(hi). if -1 we expect f(lo)>f(root)>f(hi). If this expectation is unknown, it controls the bracketing bias eg if f is all positives and sign=1, then the bracket will halve from right to left until reaching hi, if negatives and sign=1 then left to right. Note: If both sides are wrong then convergence is impossible. However, it still converges with a high likelihood if we only know the sign of either lo or hi, and that our target root has a root on the opposite sign side.

    Other Notes: Convergence is only guaranteed when there is a single root within the bracket. In a two root scenario because of how bracketing is handled, this method will always converge to the root with sign(slope)=sign (if it converges at all). Sign=-1: left root, =1: right root. Brent's method doesn't have this issue but is significantly slower. See alternative methods that can deal with multiple roots. 
    """
    fo, f = nbu.op_call_args(f_op, lo), nbu.op_call_args(f_op, hi)
    if sign == -1:
        fo, f = -f, -fo
        lamo, lam = hi, lo #root with negative slope target should use left most point to start in case right is ambiguous
    else:lamo, lam = lo, hi
    ict=np.int64(max_iters)
    while ict > 0:
        fd=(f-fo)
        fo=f
        lamn = lam - f * (lam - lamo) / fd
        lamo=lam
        lam=lamn
        if not (lo < lam < hi): lam = 0.5 * (lo + hi)
        
        if abs(lamo-lam)<er_tol:break
        
        f = nbu.op_call_args(f_op,lam) #either its a function or a tuple with the callable in the first element and it's input args for the rest, this will place lam as the first argument in the function instead of needing f_op[0](lam,*f_op[1:]), we want sign to be compile time so this should place only one of these f assignments in the instruction set.
        if sign==-1:
            f=-f
        
        if f > 0.:
            hi = lam
        else:
            lo = lam

        ict -= 1

    return lam

@nbu.jt
def signedroot_secant(f_op, lo, hi,br_rate=.5, max_iters=20, sign=1,eager=False,fallb=False,
                      br_tol=None,er_tol=None,rel_err=True,dtyp=None
                      ):
    """A bracketed secant method that achieves (empirically) faster convergence by knowing the sign of the function to the left and right of the root.
    It also allows us to select if the slope of our root is positive or negative when there are multiple roots. Which corresponds to finding
    local minima and maxima of the integrated line.

    The secant method uses the two most recent points instead of the updated lo hi brackets. This typically gets the most out of the 
    secant method, and demonstrates it's benefit under significant line asymmetry across the root. Convergence is still guaranteed 
    with bisection bracketing. Assume we know only which lo or hi has a positive sign with regards to the general problem, if left
    side is positive we are seeking a negatively sloped root sign:=-1 vice versa for right side and positive slope root.
    Then until the first time sign(value)==-1, we only take a bracketing step; this strategy allows us to converge to a root that has a sign congruent
    slope in a multi root situation. Eg in a convex problem to a -(slope) root this will always be the left root.

    Other Notes: Convergence is only guaranteed when there is a single root with a congruent slope in the bracket.
    However, the likelihood of converging to a congruent root, is still very high due to the initial side rejection strategy explained above,
    by decreasing the bracketing increment to a range that guarantees sampling a basin br_rate <.5, you once more recover guaranteed
    convergence to the signed root.

    Variable calculations are all f64.

    :param f_op: Can be a function or a function operator (tuple) that includes its arguments. It receives a single scalar value for the point estimate. 
    :param br_rate: (0,1). The bracket increment, at .5 it's classic bisection, if you expect roots to be clustered on the right then >.5 might be suitable.
    Left clustered <.5. But a smaller br_rate should always have more definite convergence.
    :param sign: =1 we expect to have f(lo)<f(root)<f(hi). if -1 we expect f(lo)>f(root)>f(hi). If this expectation is unknown,
    it controls the bracketing bias eg if f is all positives and sign=1, then the bracket will reduce from right to left at (1 - br_rate) until
    reaching hi, if negatives and sign=1 then left to right at br_rate. Note: If both sides are wrong then convergence will not occur in the single root case.
    :param p_strat: Interpolation strategy
        0 - Most recent point for lamo
        1 - least diff of value of lo, hi and lamb/current for lamo
        2 - current point and closest to zero between lo hi for lamo
        3 - least distance with current.

    """
    if dtyp is None: dtyp=nbu.type_ref(lo) # which will default to f64 unless dtyp is specified
    if br_tol is None:br_tol=nbu.prim_info(dtyp,2)**(2/3)
    if er_tol is None:er_tol=nbu.prim_info(dtyp,2)**(1/2)
    br_rate,lo,hi,er_tol,br_tol=dtyp(br_rate),dtyp(lo),dtyp(hi),dtyp(er_tol),dtyp(br_tol)
    _1=dtyp(1.)
    _0=dtyp(0.)
    sign = nbu.force_const(sign) #this will cause a recompilation every time a different sign from previous is called.
    flo, fhi = nbu.op_call_args(f_op, lo), nbu.op_call_args(f_op, hi)
    if sign == -1:
        op_bracket = (fhi < _0) or eager
        fo, f = fhi,flo
        lamo, lam = hi, lo #We know lo is positive, so we are more confident in giving it the step 2 interpolation point.
        lrt, hrt = _1 - br_rate, br_rate  #we want eagerness away from known side. so smaller=more conservative.
    else:
        op_bracket = (flo < _0) or eager
        fo, f = flo,fhi
        lamo, lam = lo, hi
        lrt, hrt = br_rate, _1 - br_rate
    #We init op_bracket by checking if the unknown point is negative, for this algo we assume that we know either lo or hi always has a positive sign for the general problem, my choice was due to the typical format of boundary solutions. If we seek a negative sloped root then we know our left side is positive, but the right side may have a basin or multiple roots (positive or negative areas), therefore we check if our right side has a negative bracketing location.
    
    #Translating to minima and maxima of integrators:
    #sign=1 right positive -> seeking minimum, starting on known positive slope point from right.
    #-func sign=1 right negative (mirrored positive) -> seeking maximum, known negative slope point from right.
    #sign=-1 left positive -> seeking a maximum, starting on known positive slope point from left.
    #-func sign=-1 left negative (mirrored positive) -> seeking minimum, known negative slope point from left.
    

    lamalt=lam-_1
    falt=f #if it wants to use secant for this point on first iteration this will correctly force it to the bracket step instead
    ict = int(max_iters)
    while ict > 0:
            fd = (f - fo)
            af=abs(f)
            nd=abs(fd) > (br_tol*max(af,abs(fo)) if rel_err else br_tol)
            _k=True
            #first is the precision good enough
            if nd:
                if not op_bracket:
                    lamb = (lo * lrt + hi * hrt)
                    _k=False
                    ll, lh = (lo, lamb) if sign == -1 else (lamb, hi)
                else:
                    ll,lh=lo,hi
                lamn = lam - f * (lam - lamo) / fd
                #if lamn is in bounds  then we skip the next two branches
                #otherwise it failed we enter next nd block
                nd= not (ll < lamn < lh)
            else:nd=True
            if nd and fallb:
                #print('Called secant fallback',ict,lamn,lam,lo,hi)
                #we check secant on the fallback bracket, succeeds then next
                #otherwise nd is true (fails again).
                ft=(f - falt)
                if abs(ft)>(br_tol*max(af,abs(falt)) if rel_err else br_tol):
                    lamn = lam - f * (lam - lamalt) / ft
                    nd = not (ll < lamn < lh)
                else:nd=True
            if nd:
                #print('Called bracket fallback',ict,lamn,lam,lo,hi)
                #final fallback after both secant points fail. either by imprecision or bounds.
                if _k:lamb=(lo * lrt + hi * hrt)
                lamn = lamb     
            lamo = lam
            fo = f
            lam=lamn
            lamalt,falt=(lo,flo) if lamo==hi else (hi,fhi)
            ict -= 1
        
            f = nbu.op_call_args(f_op,lam)
            s=f < _0
            op_bracket = op_bracket or s
            if (s if sign == -1 else not s):
                hi = lam
                fhi=f
            else:
                lo = lam
                flo=f
            
            if abs(lamo - lam) < er_tol:break

    return lam,lo,hi,2 if not op_bracket else 1 if ict==0 else 0 #2 is no lower bracket found, 1 failed to converge in time, 0 success


@nbu.jt
def posroot_nofallb_secant(f_op, lo, hi,br_rate=.5, max_iters=15,br_tol=None,er_tol=None,dtyp=None):
    #NOTE TO USERS: you can get smaller byte-codes and maybe faster methods, by propagating constants through a wrapper eg
    #But kwargs that are not called in the function header will also get statically compiled for that signature.
    """All constants in this example will compile out their conditionals, resulting in a more lightweight procedure."""
    return signedroot_secant(f_op, lo, hi, br_rate, max_iters, 1, True, False, br_tol, er_tol, True, dtyp)



@nbu.jt
def signedroot_quadinterp(f_op, lo, hi, br_rate=.5, max_iters=15, sign=1,eager=True,er_tol=None,br_tol=None,dtyp=None):
    """
    Signed Root quadratic interpolation scheme. Functions exactly like signedroot_secant but uses quadratic root solution. This method can save more than a few iterations when the problem has pronounced curvature at the root, otherwise the total # of steps is like the secant method and sometimes and rarely a bit worse if higher moments along the bracket dominate.
    
    While the secant method has a convergence order of ~O(1.6), quadinterp is ~O(1.8). Though this seems to be more like the average. With high root curvature it can even use only 1/2 the number of steps.
    """
    if dtyp is None: dtyp=nbu.type_ref(lo) # which will default to f64 unless dtyp is specified
    if br_tol is None:br_tol=nbu.prim_info(dtyp,2)**(2/3)
    if er_tol is None:er_tol=nbu.prim_info(dtyp,2)**(1/2)
    br_rate,lo,hi,er_tol,br_tol,signf=dtyp(br_rate),dtyp(lo),dtyp(hi),dtyp(er_tol),dtyp(br_tol),dtyp(sign)
    _1=dtyp(1.)
    _2=dtyp(2.)
    _4=dtyp(4.)
    _0=dtyp(0.)
    sign = nbu.force_const(sign) #this will cause a recompilation every time a different sign from previous is called.
    flo, fhi = nbu.op_call_args(f_op, lo), nbu.op_call_args(f_op, hi)
    if sign == -1:
        op_bracket = (fhi < _0) or eager
        fo, f = fhi,flo
        lamo, lam = hi, lo #We know lo is positive, so we are more confident in giving it the step 2 interpolation point.
        lrt, hrt = _1 - br_rate, br_rate  #we want eagerness away from known side. so smaller=more conservative.
    else:
        op_bracket = (flo < _0) or eager
        fo, f = flo,fhi
        lamo, lam = lo, hi
        lrt, hrt = br_rate, _1 - br_rate
    
    #Do one step of secant for the first iteration, then full quadratic.
    # --- Pre-loop: Perform one Secant/Bisection step to generate a 3rd point ---
    ict= int(max_iters)
    fd = (f - fo)
    af = abs(f)
    _k=True
    #Check secant
    if abs(fd) > br_tol * max(af, abs(fo)):
        #there is only one secant to test, no need for secant fallback strategy.
        lamn = lam - f * (lam - lamo) / fd
        if not op_bracket:
            lamb = (lo * lrt + hi * hrt)
            ll, lh = (lo, lamb) if sign == -1 else (lamb, hi)
            if not (ll < lamn < lh): lamn=lamb
        elif not (lo < lamn < hi): lamn=(lo * lrt + hi * hrt)
    else: lamn=(lo * lrt + hi * hrt)
        
    lamo = lam
    fo = f
    lam=lamn
    lamalt,falt=(lo,flo) if lamo==hi else (hi,fhi)
    ict -= 1

    f = nbu.op_call_args(f_op,lam)
    s=f < _0
    op_bracket = op_bracket or s
    if (s if sign == -1 else not s):
        hi = lam
        fhi=f
    else:
        lo = lam
        flo=f
    
    while ict > 0:
        #tbh we probably don't even need this check anymore, delta positive check could be enough.
        #nd=max(abs(f - fo),abs(f - falt)) > br_tol*max(abs(f),abs(fo),abs(falt))
        a, b, c = quadratic_newton_coef(lam, lamo, lamalt, f, fo, falt)
        delta = b*b - _4*a*c
        if delta>=_0:
            sqrt_delta = mt.sqrt(delta)
            lamn = (-b +signf*sqrt_delta) / (_2*a)
            if not op_bracket:
                lamb = (lo * lrt + hi * hrt)
                ll, lh = (lo, lamb) if sign == -1 else (lamb, hi)
                if not (ll < lamn < lh): lamn=lamb
            elif not (lo < lamn < hi): lamn=(lo * lrt + hi * hrt)
        else: lamn=(lo * lrt + hi * hrt)

        lamo = lam
        fo = f
        lam=lamn
        lamalt,falt=(lo,flo) if lamo==hi else (hi,fhi)
        ict -= 1
    
        f = nbu.op_call_args(f_op,lam)
        s=f < _0
        op_bracket = op_bracket or s
        if (s if sign == -1 else not s):
            hi = lam
            fhi=f
        else:
            lo = lam
            flo=f
        
        if abs(lamo - lam) < er_tol:break

    return lam,lo,hi,2 if not op_bracket else 1 if ict==0 else 0 #2 is no lower bracket found, 1 failed to converge in time, 0 success
    


@nbu.jt
def signedroot_newton(f_op, g_op, lo, hi, br_rate=.5, max_iters=12, sign=1,eager=True,er_tol=None,br_tol=None,rel_err=True,dtyp=None):
    """A bracketed Newton method that exploits prior knowledge about the side of the root and the desired root slope sign.

    Strategy mirrors the secant variant:
      * We bias steps using knowledge of which side of the root is positive/negative (via `sign`).
      * We preserve bisection-style bracketing to guarantee convergence when the Newton step misbehaves.
      * We only accept Newton steps that land in an admissible sub-interval (or the full bracket once op_bracket is True).
      * Before a negative-slope target (sign == -1) we flip signs so the effective problem is always increasing in the bracket:
        lo => negative, hi => positive. The derivative is also flipped consistently.

    Notes:
      * If the derivative goes tiny, we fall back to a bracketing step.
      * Convergence is only guaranteed when there is a single congruent-slope root in the bracket (same caveats as the secant version).
      * The bracket reduction eagerness is controlled by br_rate, exactly as in the secant version.

    Variable calculations are all f64.

    :param f_op: function or operator tuple; called via nbu.op_call_args(f_op, x) -> f(x).
    :param g_op: gradient operator; nbu.op_call_args(g_op, x) -> f'(x).
    :param br_rate: (0,1). Bracket increment (0.5 = classic bisection).
    :param sign: =1 means we expect f(lo)<f(root)<f(hi); =-1 means f(lo)>f(root)>f(hi).
    """
    if dtyp is None: dtyp=nbu.type_ref(lo) # which will default to f64 unless dtyp is specified
    if br_tol is None:br_tol=nbu.prim_info(dtyp,2)**(2/3)
    if er_tol is None:er_tol=nbu.prim_info(dtyp,2)**(1/2)
    br_rate,lo,hi,er_tol,br_tol=dtyp(br_rate),dtyp(lo),dtyp(hi),dtyp(er_tol),dtyp(br_tol)
    _1=dtyp(1.)
    _0=dtyp(0.)
    sign = nbu.force_const(sign)
    # Bias parameters mirror your secant version.
    if sign == 1:
        # Known side is right/hi (positive), eagerness toward the left.
        lrt, hrt = br_rate, _1 - br_rate
        lam = hi  # start Newton on the known side
    else:
        # Known side is left/lo (positive), we want eagerness toward the right/unknown side.
        lrt, hrt = _1 - br_rate, br_rate
        lam = lo  # start Newton on the known side

    # One f,g eval "outside" the loop; reused to propose the first Newton step.
    g = nbu.op_call_args(g_op, lam)
    f = nbu.op_call_args(f_op, lam)

    ict = int(max_iters)
    op_bracket = eager
    
    #We don't need to record all three relevant points besides for bracketing, so this looks a bit different.
    while ict > 0:
        # Proposed Newton step
        lamo = lam
        if abs(g)<(br_tol*abs(f) if rel_err else br_tol):
            lam = (lo*lrt + hi*hrt)
        else:
            lam = lam - f / g
            lamb = (lo * lrt + hi * hrt)
    
            # Define admissible region
            ll, lh = ((lamb, hi) if sign == 1 else (lo, lamb)) if not op_bracket else (lo, hi)
            if not (ll < lam < lh): lam = lamb
        
        g = nbu.op_call_args(g_op, lam)        
        f = nbu.op_call_args(f_op, lam)

        # Bracket update
        if (f > _0 if sign == 1 else f < _0):
            hi = lam
        else:
            op_bracket = True
            lo = lam

        ict -= 1

        # Termination on step size, consistent with your secant version.
        if abs(lamo - lam) < er_tol:
            break

    # Status: 0 on step-based convergence, 1 on iteration exhaustion, 2 failed to find opposite sign position.
    return lam, lo, hi, 2 if not op_bracket else 1 if ict==0 else 0


@nbu.jt
def signseeking_halley(f_op, g_op,c_op, lo, hi, br_rate=.5, max_iters=12, sign=1, eager=True,er_tol=None,br_tol=None,rel_err=True,dtyp=None):
    """A bracketed halley method that exploits prior knowledge about the side of the root and the desired root slope sign.
    
    Notes from Newton version, assume Halley:
    Strategy mirrors the secant variant:
      * We bias steps using knowledge of which side of the root is positive/negative (via `sign`).
      * We preserve bisection-style bracketing to guarantee convergence when the Newton step misbehaves.
      * We only accept Newton steps that land in an admissible sub-interval (or the full bracket once op_bracket is True).
      * Before a negative-slope target (sign == -1) we flip relations so the effective problem is always increasing in the bracket:
        lo => negative, hi => positive.

    Notes:
      * If the derivative goes tiny, we fall back to a bracketing step.
      * Convergence is only guaranteed when there is a single congruent-slope root in the bracket (same caveats as the secant version).
      * The bracket reduction eagerness is controlled by br_rate, exactly as in the secant version.

    Variable calculations are all f64.

    :param f_op: function or operator tuple; called via nbu.op_call_args(f_op, x) -> f(x).
    :param g_op: gradient operator; nbu.op_call_args(g_op, x) -> f'(x).
    :param g_op: c_op
    :param br_rate: (0,1). Bracket increment (0.5 = classic bisection).
    :param sign: =1 means we expect f(lo)<f(root)<f(hi); =-1 means f(lo)>f(root)>f(hi).
    """
    if dtyp is None: dtyp=nbu.type_ref(lo) # which will default to f64 unless dtyp is specified
    if br_tol is None:br_tol=nbu.prim_info(dtyp,2)#**(2/3)
    if er_tol is None:er_tol=nbu.prim_info(dtyp,2)**(1/2)
    br_rate,lo,hi,er_tol,br_tol=dtyp(br_rate),dtyp(lo),dtyp(hi),dtyp(er_tol),dtyp(br_tol)
    _1=dtyp(1.)
    _2=dtyp(2.)
    _0=dtyp(0.)
    _05=dtyp(0.5)
    #sign = nbu.force_const(sign)
    # Bias parameters mirror your secant version.
    if sign == 1:
        # Known side is right/hi (positive), eagerness toward the left.
        lrt, hrt = br_rate, _1 - br_rate
        lam = hi  # start halley on the known side
    else:
        # Known side is left/lo (positive), we want eagerness toward the right/unknown side.
        lrt, hrt = _1 - br_rate, br_rate
        lam = lo  # start halley on the known side

    # One f,g eval "outside" the loop; reused to propose the first Newton step.
    f = nbu.op_call_args(f_op, lam)
    g = nbu.op_call_args(g_op, lam)
    c = nbu.op_call_args(c_op, lam)

    ict = int(max_iters)
    op_bracket = eager

    while ict > 0:
        # Proposed Halley step
        lamo = lam
        denom=(g*g - _05*f*c)
        if abs(denom) < (br_tol*abs(f) if rel_err else br_tol):
            lam = (lo * lrt + hi * hrt)
        else:
            lam = lam - f*g / denom #see that if c~0 we get the newton update.
            lamb = (lo * lrt + hi * hrt)

            # Define admissible region
            ll, lh = ((lamb, hi) if sign == 1 else (lo, lamb)) if not op_bracket else (lo, hi)
            if not (ll < lam < lh): lam = lamb

        f = nbu.op_call_args(f_op, lam)
        g = nbu.op_call_args(g_op, lam)
        c = nbu.op_call_args(c_op, lam)

        # Bracket update
        if (f > _0 if sign == 1 else f < _0):
            hi = lam
        else:
            op_bracket = True
            lo = lam

        ict -= 1

        # Termination on step size which is the best proxy for achievable convergence.
        if abs(lamo - lam) < er_tol:
            break

    # Status: 0 on step-based convergence, 1 on iteration exhaustion, 2 failed to find opposite sign position.
    return lam, lo, hi, 2 if not op_bracket else 1 if ict == 0 else 0

@nbu.rgi
def brents_method(f_op, lo, hi, er_tol=1e-12, max_iters=50):
    """ NOTE: In reality this method appears to almost never outperform the bracketed secant or quadinterp method, converging in more steps. But all have smooth convergence guarantees from the bisection bracket. And the compiled kernel of bracketed secant will be significantly smaller. Maybe this implementation is not completely correct?
    
    Original Brent's method per Wikipedia (inverse quadratic interpolation + secant + bisection),
    no pre-bisection. Requires the caller to provide a valid bracket with f(lo)*f(hi) <= 0.
    Stopping: 'stationary enough' on b, i.e. |b - b_prev| < er_tol.
    """
    a = float(lo)
    b = float(hi)
    fa = nbu.op_call_args(f_op, a)
    fb = nbu.op_call_args(f_op, b)

    # Initialize
    c = a
    fc = fa
    d = b - a
    mflag = True
    ict = np.int64(max_iters)
    it = 1
    
    while ict > 0:

        if (fa != fc) and (fb != fc):
            # print('quad inv method',it)
            denom1 = (fa - fb) * (fa - fc)
            denom2 = (fb - fa) * (fb - fc)
            denom3 = (fc - fa) * (fc - fb)
            s = (a * fb * fc) / denom1 + (b * fa * fc) / denom2 + (c * fa * fb) / denom3
        else:
            denom = (fb - fa)
            s = b - fb * (b - a) / denom

        tol = er_tol

        # acceptance tests (per Wikipedia)
        cond1 = (s < (3.0 * a + b) * 0.25) or (s > b)
        if mflag:
            cond2 = abs(s - b) >= 0.5 * abs(b - c)
            cond4 = abs(b - c) < tol
        else:
            cond2 = abs(s - b) >= 0.5 * abs(c - d)
            cond4 = abs(c - d) < tol
        cond3 = mflag and (abs(s - b) >= 0.5 * abs(b - a))
        use_bisection = cond1 or cond2 or cond3 or cond4

        if use_bisection:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = nbu.op_call_args(f_op, s)

        d = c
        c = b
        fc = fb

        # update bracket
        if fa * fs < 0.0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        # keep |fb| <= |fa|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        it += 1

        if abs(a - b) < er_tol:
            break

        ict -= 1

    return b
