from sympy.solvers.solveset import linear_coeffs

from devito import Eq
from devito.finite_differences.differentiable import diffify

from wave_utils import freesurface
from FD_utils import laplacian, ssa_tti


def _solve(eq, target, **kwargs):
    """
    To be remved at next Devito release
    """
    if isinstance(eq, Eq):
        eq = eq.lhs - eq.rhs if eq.rhs != 0 else eq.lhs
    # Try first linear solver
    cc = linear_coeffs(eq.evaluate, target)
    return diffify(-cc[1]/cc[0])


def wave_kernel(model, u, fw=True, q=None):
    """
    Pde kernel corresponding the the model for the input wavefield

    Parameters
    ----------
    model: Model
        Physical model
    u : TimeFunction or tuple
        wavefield (tuple if TTI)
    fw : Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source
    """
    if model.is_tti:
        pde, fact = tti_kernel(model, u[0], u[1], fw=fw, q=q)
        fact += freesurface(model, fact) if model.fs else []
    else:
        pde = acoustic_kernel(model, u, fw, q=q)
        fact = []
        pde += freesurface(model, pde) if model.fs else []
    return pde, fact


def acoustic_kernel(model, u, fw=True, q=None):
    """
    Acoustic wave equation time stepper

    Parameters
    ----------
    model: Model
        Physical model
    u : TimeFunction or tuple
        wavefield (tuple if TTI)
    fw : Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source
    """
    u_n = u.forward if fw else u.backward
    udt = u.dt if fw else u.dt.T
    q = q or 0

    # Set up PDE expression and rearrange
    ulaplace = laplacian(u, model.irho)
    wmr = model.irho * model.m
    damp = model.damp
    stencil = _solve(wmr * (u.dt2 + damp * udt) - ulaplace - q, u_n)

    if 'nofsdomain' in model.grid.subdomains:
        return [Eq(u_n, stencil, subdomain=model.grid.subdomains['nofsdomain'])]
    return [Eq(u_n, stencil)]


def tti_kernel(model, u1, u2, fw=True, q=None):
    """
    TTI wave equation (one from my paper) time stepper

    Parameters
    ----------
    model: Model
        Physical model
    u1 : TimeFunction
        First component (pseudo-P) of the wavefield
    u2 : TimeFunction
        First component (pseudo-P) of the wavefield
    fw: Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source as a tuple (one value for each component)
    """
    m, damp, irho = model.m, model.damp, model.irho
    wmr = (irho * m)
    q = q or (0, 0)

    # Tilt and azymuth setup
    u1_n, u2_n = (u1.forward, u2.forward) if fw else (u1.backward, u2.backward)
    (udt1, udt2) = (u1.dt, u2.dt) if fw else (u1.dt.T, u2.dt.T)
    H0, H1, factp, factm = ssa_tti(u1, u2, model)

    # Stencils
    stencilp = _solve(wmr * (u1.dt2 + damp * udt1) - H0 - q[0], u1_n)
    stencilr = _solve(wmr * (u2.dt2 + damp * udt2) - H1 - q[1], u2_n)

    if 'nofsdomain' in model.grid.subdomains:
        pdea = freesurface(model, acoustic_kernel(model, u1, fw, q=q[0]))
        first_stencil = Eq(u1_n, stencilp, subdomain=model.grid.subdomains['nofsdomain'])
        second_stencil = Eq(u2_n, stencilr, subdomain=model.grid.subdomains['nofsdomain'])
    else:
        pdea = []
        first_stencil = Eq(u1_n, stencilp)
        second_stencil = Eq(u2_n, stencilr)

    return [first_stencil, second_stencil] + pdea, [factp, factm]
