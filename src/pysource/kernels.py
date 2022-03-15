from devito import Eq, solve, div, grad
from sympy import sqrt

from wave_utils import freesurface
from FD_utils import laplacian, sa_tti


def wave_kernel(model, u, fw=True, q=None, f0=0.015):
    """
    Pde kernel corresponding the the model for the input wavefield

    Parameters
    ----------
    model: Model
        Physical model
    u : TimeFunction or tuple
        wavefield (tuple if TTI or Viscoacoustic)
    fw : Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source
    f0 : Peak frequency
    """
    if model.is_tti:
        pde = tti_kernel(model, u[0], u[1], fw=fw, q=q)
    elif model.is_viscoacoustic:
        pde = SLS_2nd_order(model, u[0], u[1], fw=fw, q=q, f0=f0)
    else:
        pde = acoustic_kernel(model, u, fw, q=q)
    return pde


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
    stencil = solve(wmr * u.dt2 + damp * udt - ulaplace - q, u_n)

    if 'nofsdomain' in model.grid.subdomains:
        pde = [Eq(u_n, stencil, subdomain=model.grid.subdomains['nofsdomain'])]
        pde += freesurface(model, pde)
    else:
        pde = [Eq(u_n, stencil)]

    return pde


def SLS_2nd_order(model, u1, u2, fw=True, q=None, f0=0.015):
    """
    Viscoacoustic 2nd SLS wave equation

    Parameters
    ----------
    model: Model
        Physical model
    u1 : TimeFunction
        Pressure field
    u2 : TimeFunction
        Attenuation Memory variable
    fw: Bool
        Whether forward or backward in time propagation
    q : TimeFunction or Expr
        Full time-space source as a tuple (one value for each component)
    f0 : Peak frequency
    """
    qp, vp = model.qp, model.vp

    damp = model.damp

    q = q or 0

    b = model.irho

    s = model.grid.stepping_dim.spacing

    # The stress relaxation parameter
    t_s = (sqrt(1.+1./qp**2)-1./qp)/f0

    # The strain relaxation parameter
    t_ep = 1./(f0**2*t_s)

    # The relaxation time
    tt = (t_ep/t_s)-1.

    # Density
    rho = 1. / b

    # Bulk modulus
    bm = rho * vp**2

    p = u1
    r = u2

    if fw:

        pde_r = r + s * (tt / t_s) * rho * div(b * grad(p, shift=.5), shift=-.5) - \
            s * (1. / t_s) * r

        u_r = Eq(r.forward, damp * pde_r)

        pde_p = 2. * p - damp * p.backward + s**2 * bm * (1. + tt) * \
            div(b * grad(p, shift=.5), shift=-.5) - s**2 * vp**2 * \
            r.forward + s**2 * vp**2 * q

        u_p = Eq(p.forward, damp * pde_p)

        return [u_r, u_p]

    else:

        pde_r = r + s * (tt / t_s) * p - s * (1. / t_s) * r
        u_r = Eq(r.backward, damp * pde_r)

        pde_p = 2. * p - damp * p.forward + s**2 * vp**2 * \
            div(b * grad((1. + tt) * rho * p, shift=.5), shift=-.5) - s**2 * vp**2 * \
            div(b * grad(rho * r.backward, shift=.5), shift=-.5)
        u_p = Eq(p.backward, damp * pde_p)

        return [u_r, u_p]


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
    H0, H1 = sa_tti(u1, u2, model)

    # Stencils
    stencilp = solve(wmr * u1.dt2 + damp * udt1 - H0 - q[0], u1_n)
    stencilr = solve(wmr * u2.dt2 + damp * udt2 - H1 - q[1], u2_n)

    if 'nofsdomain' in model.grid.subdomains:
        pdea = freesurface(model, acoustic_kernel(model, u1, fw, q=q[0]))
        first_stencil = Eq(u1_n, stencilp, subdomain=model.grid.subdomains['nofsdomain'])
        second_stencil = Eq(u2_n, stencilr, subdomain=model.grid.subdomains['nofsdomain'])
    else:
        pdea = []
        first_stencil = Eq(u1_n, stencilp)
        second_stencil = Eq(u2_n, stencilr)

    return [first_stencil, second_stencil] + pdea
