from sympy import cos, sin
from devito import Eq

from wave_utils import freesurface
from FD_utils import laplacian, ssa_tti


def wave_kernel(model, u, fw=True, q=None, fs=False):
    """
    Pde kernel corresponding the the model for the input wavefield


    """
    if model.is_tti:
        pde = tti_kernel(model, u[0], u[1], fw=fw, q=q)
    else:
        pde = acoustic_kernel(model, u, fw, q=q)

    fs_eq = freesurface(u, model.nbl, forward=fw) if fs else []
    return pde + fs_eq


def acoustic_kernel(model, u, fw=True, q=None):
    """

    Acoustic wave equation time stepper
    """
    u_n, u_p = (u.forward, u.backward) if fw else (u.backward, u.forward)
    q = q or 0
    # Set up PDE expression and rearrange
    ulaplace = laplacian(u, model.irho)
    wmr = 1 / (model.irho * model.m)
    s = model.grid.time_dim.spacing
    stencil = model.damp * (2.0 * u - model.damp * u_p + s**2 * wmr * (ulaplace + q))
    return [Eq(u_n, stencil)]


def tti_kernel(model, u1, u2, fw=True, q=None):
    """
    TTI wave equation (one from my paper) time stepper
    """
    m, damp, irho = model.m, model.damp, model.irho
    wmr = 1 / (irho * m)
    q = q or (0, 0)
    # Tilt and azymuth setup

    u1_n, u1_p = (u1.forward, u1.backward) if fw else (u1.backward, u1.forward)
    u2_n, u2_p = (u2.forward, u2.backward) if fw else (u2.backward, u2.forward)
    H0, H1 = ssa_tti(u1, u2, model)
    # Stencils
    s = model.grid.stepping_dim.spacing
    stencilp = damp * (2 * u1 - damp * u1_p + s**2 * wmr * (H0 + q[0]))
    stencilr = damp * (2 * u2 - damp * u2_p + s**2 * wmr * (H1 + q[1]))
    first_stencil = Eq(u1_n, stencilp)
    second_stencil = Eq(u2_n, stencilr)
    return [second_stencil, first_stencil]
