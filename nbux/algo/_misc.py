import nbux._utils as nbu

@nbu.rg
def newton_to_monomial_quadratic(x0, x1, x2, f0, f1, f2):
    """
    Computes coefficients a, b, c for the quadratic polynomial ax^2 + bx + c
    passing through (x0, f0), (x1, f1), (x2, f2).
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
