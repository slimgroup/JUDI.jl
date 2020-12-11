import numpy as np
from sympy import cos, sin

from devito import Eq
from devito.tools import as_tuple

from wave_utils import sub_time, freesurface
from FD_utils import divs, grads


def func_name(freq=None, isic=False):
    """
    Get key for imaging condition/linearized source function
    """
    if freq is None:
        return 'isic' if isic else 'corr'
    else:
        return 'isic_freq' if isic else 'corr_freq'


def grad_expr(gradm, u, v, model, w=None, freq=None, dft_sub=None, isic=False):
    """
    Gradient update stencil

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    w: Float or Expr (optional)
        Weight for the gradient expression (default=1)
    freq: Array
        Array of frequencies for on-the-fly DFT
    factor: int
        Subsampling factor for DFT
    isic: Bool
        Whether or not to use inverse scattering imaging condition (not supported yet)
    """
    ic_func = ic_dict[func_name(freq=freq, isic=isic)]
    expr = ic_func(as_tuple(u), as_tuple(v), model, freq=freq, factor=dft_sub, w=w)
    if model.fs:
        eq_g = [Eq(gradm, gradm - expr, subdomain=model.grid.subdomains['nofsdomain'])]
        eq_g += freesurface(model, eq_g)
    else:
        eq_g = [Eq(gradm, gradm - expr)]
    return eq_g


def crosscorr_time(u, v, model, **kwargs):
    """
    Cross correlation of forward and adjoint wavefield

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    """
    w = kwargs.get('w') or u[0].indices[0].spacing * model.irho
    return w * sum(vv.dt2 * uu for uu, vv in zip(u, v))


def crosscorr_freq(u, v, model, freq=None, dft_sub=None, **kwargs):
    """
    Standard cross-correlation imaging condition with on-th-fly-dft

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    freq: Array
        Array of frequencies for on-the-fly DFT
    factor: int
        Subsampling factor for DFT
    """
    # Subsampled dft time axis
    time = model.grid.time_dim
    dt = time.spacing
    tsave, factor = sub_time(time, dft_sub)
    expr = 0

    for uu, vv in zip(u, v):
        ufr, ufi = uu
        # Frequencies
        nfreq = np.shape(freq)[0]
        fdim = ufr.dimensions[0]
        omega_t = lambda f: 2*np.pi*f*tsave*factor*dt
        # Gradient weighting is (2*np.pi*f)**2/nt
        w = lambda f: -(2*np.pi*f)**2/time.symbolic_max
        expr += sum(w(freq[ff])*(ufr._subs(fdim, ff)*cos(omega_t(freq[ff])) -
                                 ufi._subs(fdim, ff)*sin(omega_t(freq[ff])))
                    for ff in range(nfreq))*vv
    return expr


def isic_time(u, v, model, **kwargs):
    """
    Inverse scattering imaging condition

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    """
    w = u[0].indices[0].spacing * model.irho
    return w * sum(uu * vv.dt2 * model.m + inner_grad(uu, vv)
                   for uu, vv in zip(u, v))


def isic_freq(u, v, model, **kwargs):
    """
    Inverse scattering imaging condition

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    v: TimeFunction or Tuple
        Adjoint wavefield (tuple of fields for TTI)
    model: Model
        Model structure
    """
    freq = kwargs.get('freq')
    # Subsampled dft time axis
    time = model.grid.time_dim
    dt = time.spacing
    tsave, factor = sub_time(time, kwargs.get('factor'))
    expr = 0
    for uu, vv in zip(u, v):
        ufr, ufi = uu
        # Frequencies
        nfreq = np.shape(freq)[0]
        fdim = ufr.dimensions[0]
        omega_t = lambda f: 2*np.pi*f*tsave*factor*dt
        # Gradient weighting is (2*np.pi*f)**2/nt
        w = lambda f: -(2*np.pi*f)**2/time.symbolic_max
        w2 = factor / time.symbolic_max
        for ff in range(nfreq):
            cwt, swt = cos(omega_t(freq[ff])), sin(omega_t(freq[ff]))
            ufrf, ufif = ufr._subs(fdim, ff), ufi._subs(fdim, ff)
            idftu = (ufrf * cwt - ufif * swt)
            expr += w(freq[ff]) * idftu * vv * model.m - w2 * inner_grad(idftu, vv)
    return expr


def lin_src(model, u, isic=False):
    """
    Source for linearized modeling

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    model: Model
        Model containing the perturbation dm
    """
    ls_func = ls_dict[func_name(isic=isic)]
    return ls_func(model, as_tuple(u))


def basic_src(model, u, **kwargs):
    """
    Basic source for linearized modeling

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    model: Model
        Model containing the perturbation dm
    """
    w = -model.dm * model.irho
    if model.is_tti:
        return (w * u[0].dt2, w * u[1].dt2)
    return w * u[0].dt2


def isic_src(model, u, **kwargs):
    """
    ISIC source for linearized modeling

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    model: Model
        Model containing the perturbation dm
    """
    m, dm, irho = model.m, model.dm, model.irho
    dus = []
    for uu in u:
        du_aux = divs(dm * irho * grads(uu, so_fact=2), so_fact=2)
        dus.append(dm * irho * uu.dt2 * m - du_aux)
    if model.is_tti:
        return (-dus[0], -dus[1])
    return -dus[0]


def inner_grad(u, v):
    """
    Inner product of the gradient of two Function.

    Parameters
    ----------
    u: TimeFunction or Function
        First wavefield
    v: TimeFunction or Function
        Second wavefield
    """
    return sum([a*b for a, b in zip(grads(u, so_fact=2), grads(v, so_fact=2))])


ic_dict = {'isic_freq': isic_freq, 'corr_freq': crosscorr_freq,
           'isic': isic_time, 'corr': crosscorr_time}
ls_dict = {'isic': isic_src, 'corr': basic_src}
