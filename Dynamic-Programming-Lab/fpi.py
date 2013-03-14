## Filename: fpi.py
## Author David R. Pugh
## Based on code by John Stachurski

from scipy import mean, absolute as abs
from lininterp import LinInterp

def maximizer(h, a, b):
    g = lambda x: - h(x)               # Negate 
    return fminbound(g, a, b)          # and minimize

def T(sigma, w):
    " Implements the operator L T_sigma"
    vals = []
    for y in grid:
        Tw_y = U(y - sigma(y)) + rho * mean(w(f(sigma(y), W)))
        vals.append(Tw_y)
    return LinInterp(grip, vals)

def get_greedy(w):
    "Computes a w-greedy policy"
    vals = []
    for y in grid:
        h = lambda k: U(y - k) + rho * mean(w(f(k,W)))
        vals.append(maximizer(h, 0, y))
    return LinInterp(grid, vals)

def get_value(sigma, v):
    """

    Computes an approximation to v_sigma, the value of following policy sigma.
    Function v is an initial guess.

    """

    tol = 1e-2
    while 1:
        new_v = T(sigma, v)
        err = max(abs(new_v(grid) - v(grid)))
        if err < tol:
            return new_v
        v = new_v
