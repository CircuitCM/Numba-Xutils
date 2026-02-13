import numba as nb
import numpy as np

from nbux import _utils as nbu
from . import vector as opv

BLAS_PACK=False

@nbu.rg
def quadratic_newton_coef(x0, x1, x2, f0, f1, f2):
    """
    Quadratic polynomial coefficients using Newton's divided differences procedure.

    :param x0: First x sample.
    :param x1: Second x sample.
    :param x2: Third x sample.
    :param f0: ``f(x0)``.
    :param f1: ``f(x1)``.
    :param f2: ``f(x2)``.
    :returns: ``(a, b, c)`` coefficients for ``a*x**2 + b*x + c``.
    """
    # Compute divided differences
    c1 = (f1 - f0) / (x1 - x0)
    f12 = (f2 - f1) /(x2 - x1)
    a = (f12 - c1) / (x2 - x0)
    
    # Convert Newton form to Monomial form:
    # P(x) = c2(x-x0)(x-x1) + c1(x-x0) + c0
    #      = c2(x^2 - x(x0+x1) + x0x1) + c1(x - x0) + c0
    #      = c2*x^2 + (c1 - c2(x0+x1))*x + (c0 - c1*x0 + c2*x0*x1)
    b = c1 - a * (x0 + x1)
    c= f0 - c1 * x0 + a * x0 * x1
    
    return a, b, c


@nbu.rg
def cubic_newton_coef(
        x0, x1, x2, x3,
        f0, f1, f2, f3):
    """
    Cubic polynomial coefficients using Newton's divided differences procedure.

    :param x0: First x sample.
    :param x1: Second x sample.
    :param x2: Third x sample.
    :param x3: Fourth x sample.
    :param f0: ``f(x0)``.
    :param f1: ``f(x1)``.
    :param f2: ``f(x2)``.
    :param f3: ``f(x3)``.
    :returns: ``(a, b, c, d)`` coefficients for ``a*x**3 + b*x**2 + c*x + d``.
    """
    c1 = (f1 - f0)/(x1 - x0)
    f12 = (f2 - f1)/(x2 - x1)
    c2 = (f12 - c1)/(x2 - x0)
    f23 = (f3 - f2)/(x3 - x2)
    f123 = (f23 - f12)/(x3 - x1)
    r0=(f123 - c2) / (x3 - x0)

    #build p(x) = (((c3*(x - x2) + c2)*(x - x1) + c1)*(x - x0) + c0)

    # Multiply by (x - x2), then add c2  -> degree 1
    r1= -x2 * r0 + c2      # new constant

    # Multiply by (x - x1), then add c1  -> degree 2
    r2 = -x1 * r1 + c1     # const
    r3 =  r1 - x1 * r0     # x^1

    # Multiply by (x - x0), then add c0  -> degree 3
    a =  r0                # x^3
    b =  r3 - x0 * r0      # x^2
    c =  r2 - x0 * r3      # x^1
    d = -x0 * r2 + f0      # const

    return a, b, c, d


@nbu.rgi
def cubic_lagrange_coef(
    x0, x1, x2, x3,
    f0, f1, f2, f3
):
    """
    Cubic polynomial coefficients calculated by Lagrange interpolation.

    From local tests this is both slower and less accurate than cubic Newton
    differences.

    :param x0: First x sample.
    :param x1: Second x sample.
    :param x2: Third x sample.
    :param x3: Fourth x sample.
    :param f0: ``f(x0)``.
    :param f1: ``f(x1)``.
    :param f2: ``f(x2)``.
    :param f3: ``f(x3)``.
    :returns: ``(a, b, c, d)`` coefficients for ``a*x**3 + b*x**2 + c*x + d``.
    """
    # barycentric weights w_i = 1 / Π_{j≠i} (x_i - x_j)
    c1=(x0 - x1)
    c2=(x0 - x2)
    c3=(x0 - x3)
    c12=(x1 - x2)
    c13=(x1 - x3)
    c23=(x2 - x3)
    # + + + = +
    w0 = 1.0 / (c1*c2*c3)
    # - + + = -
    w1 = -1.0 / (c1*c12*c13)
    # - - + = +
    w2 = 1.0 / (c2*c12*c23)
    # - - - = -
    w3 = -1.0 / (c3*c13*c23)

    # Excluding-i symmetric sums:
    # s1^{(i)} = S1 - xi
    # s2^{(i)} = S2 - xi*(S1 - xi)
    # s3^{(i)} = (product of the other three) = S4/xi
    
    r0=f0*w0
    r1=f1*w1
    r2=f2*w2
    r3=f3*w3
    a = r0+r1+r2+r3
    S1 = x0 + x1 + x2 + x3
    # s10=x1 + x2 + x3
    # s11= x0 + x2 + x3
    # s12= x0 + x1 + x3
    # s13= x0 + x1 + x2
    s10 = S1 - x0
    s11 = S1 - x1
    s12 = S1 - x2
    s13 = S1 - x3
    b = -(r0*s10 + r1*s11 + r2*s12 + r3*s13)
    S2 = (x0*x1 + x0*x2 + x0*x3 + x1*x2 + x1*x3 + x2*x3)
    s20 = S2 - x0*s10
    s21 = S2 - x1*s11
    s22 = S2 - x2*s12
    s23 = S2 - x3*s13
    c =  (r0*s20 + r1*s21 + r2*s22 + r3*s23)
    S4 = x0 * x1 * x2 * x3
    s30 = S4 / x0
    s31 = S4 / x1
    s32 = S4 / x2
    s33 = S4 / x3
    d = -(r0*s30 + r1*s31 + r2*s32 + r3*s33)
    return a, b, c, d


@nbu.rgi
def horner_eval(x,coefs):
    """
    Horner evaluation kernel.

    Because it uses inline, ``coefs`` can be a tuple and receive full unrolling
    benefits, or an array.

    :param x: Evaluation point.
    :param coefs: Coefficients sequence.
    :returns: Polynomial value.
    """
    v=coefs[0]
    for c in nb.literal_unroll(coefs[1:]): v=v*x+c
    return v

@nbu.jtic
def sqr_lh(out):
    """
    Square matrix lower-half fill.

    :param out: Square matrix to fill in-place.
    :returns: None.
    """
    #doesn't matter if first or second axis, as this is v square matrix.
    #There is only one write port, but two load ports, so technically going contiguous on the write port should be more efficient.
    #test this later.
    for j in range(out.shape[0]):
        for i in range(0, j):
            out[j, i] = out[i, j]

@nbu.jtic
def sqr_uh(out):
    """
    Square matrix upper-half fill.

    :param out: Square matrix to fill in-place.
    :returns: None.
    """
    #doesn't matter if first or second axis, as this is v square matrix.
    #There is only one write port, but two load ports, so technically going contiguous on the write port should be more efficient.
    #test this later.
    for j in range(out.shape[0]):
        for i in range(j+1, out.shape[0]):
            out[j, i] = out[i, j]


from numba.core import types
    
if BLAS_PACK:
    def mmul_cself(a,out,a_mult=1.,rem_mult=0.,sym=False,outer=True):
        pass #to replaced after blas_lapack implementations are added.
    def cholesky_fsolve_inplace(a,b,uplo=types.char((ord('U')))):
        pass
    def potrs(L,x):
        pass
    
else:
    @nbu.jtc
    def mmul_cself(a,out,a_mult=1.,rem_mult=0.,sym=False,outer=True):
        """
        Matrix multiply self (C-ordered). This like BLAS syrk with a more friendly interface.

        If ``outer`` is True and ``a`` has shape ``(m, n)``, then ``out`` has
        shape ``(m, m)``. If ``outer`` is False (inner), then ``out`` has shape
        ``(n, n)``.

        :param a: Input matrix.
        :param out: Output matrix (written in-place).
        :param a_mult: Multiplier applied to ``a`` (or ``a.conj()``).
        :param rem_mult: Multiplier for combining with an existing ``out``.
        :param sym: If True, treat output as symmetric (currently unused).
        :param outer: Whether to compute the outer or inner product variant.
        :returns: ``out``.
        """
        at=nbu.type_ref(a)
        _0=at(0.)
        _1=at(1.)
        a_mult,rem_mult=at(a_mult),at(rem_mult)
        isc=at is np.complex64 or at is np.complex128
        if isc:
            ax=a.conj()
            if a_mult!=_1: ax*=a_mult
        elif a_mult!=_1: #np.dot will definitely copy if mem-address is the same, so might as well make our own copy
            ax=a*a_mult
        else:
            ax=a
        if outer:a,ax=a,ax.T
        else:a,ax=a.T,ax
        if rem_mult==_0:
            np.dot(a,ax,out=out)
        else:
            ac=np.dot(a,ax)
            opv.pxaxy(out.ravel(),rem_mult,ac.ravel()) #dimensions should match so this is faster than multi-looping.
        
        
        #if sym:sqr_lh(v,out)
        return out
    
    @nbu.jt
    def potrs(L,x):
        """
        An unoptimized ``potrs`` substitute.

        :param L: The lower factored matrix.
        :param x: The initial system vector (solution is overwritten in-place).
        :returns: ``x`` (the solution vector).
        """
        typ=nbu.type_ref(x)
        _0=typ(0.)
        n = x.shape[0]
    
        # --- Step 1: Forward Substitution ---
        # Solve L * y = b 
        # y is stored in b
        for i in range(n):
            # Dot product of row L[i, :i] and previous solutions y[:i]
            sv=opv.dot(L[i,:i],x[:i])
            x[i] = (x[i] - sv) / L[i, i]
    
        # --- Step 2: Backward Substitution ---
        # Solve L.T * x = y
        # x is stored in b (overwriting y)
        for i in range(n - 1, -1, -1):
            sum_val = _0
            # Dot product of col L[i+1:, i] (which is row i of L.T) and known x values
            for j in range(i + 1, n):
                sum_val += L[j, i] * x[j]  # Access L[j, i] for L.T
            x[i] = (x[i] - sum_val) / L[i, i]
            
        return x
        
    @nbu.jtc
    def cholesky_fsolve_inplace(a,b,uplo=types.char((ord('U')))):
        #col major upper is = row major lower for c array.
        L=np.linalg.cholesky(a)
        return potrs(L,b)
    
    ## Unit test : complex and real 
