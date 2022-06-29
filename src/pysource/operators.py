import numpy as np
from collections.abc import Hashable
from functools import partial

from devito import Constant, Operator, Function, info

from models import EmptyModel
from kernels import wave_kernel
from geom_utils import geom_expr
from fields import wavefield, forward_wavefield
from fields_exprs import (otf_dft, extended_rec,
                          extented_src, save_subsampled, weighted_norm)
from sensitivity import grad_expr, lin_src
from utils import opt_op


def name(model):
    if model.is_tti:
        return "tti"
    elif model.is_viscoacoustic:
        return "viscoacoustic"
    else:
        return ""


class memoized_func(object):
    """
    Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated). This decorator may also be used on class methods,
    but it will cache at the class level; to cache at the instance level,
    use ``memoized_meth``.

    Adapted from: ::

        https://github.com/devitocodes/devito/blob/master/devito/tools/memoization.py


    This version is made task safe to prevent access conflicts between different julia
    workers.

    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kw):
        if not isinstance(args, Hashable):
            # Uncacheable, a list, for instance.
            # Better to not cache than blow up.
            return self.func(*args, **kw)
        key = (self.func, args, frozenset(kw.items()))
        if key in self.cache:
            while True:
                try:
                    return self.cache[key]
                except RuntimeError:
                    pass

        value = self.func(*args, **kw)
        self.cache[key] = value
        return value

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)


@memoized_func
def forward_op(p_params, tti, visco, space_order, spacing, save, t_sub, fs, pt_src,
               pt_rec, dft, dft_sub, ws, full_q):
    """
    Low level forward operator creation, to be used through `propagator.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    info("Building forward operator")
    # Some small dummy dims
    model = EmptyModel(tti, visco, spacing, fs, space_order, p_params)
    nt = 10
    ndim = len(spacing)
    scords = np.ones((1, ndim)) if pt_src else None
    rcords = np.ones((1, ndim)) if pt_rec else None
    wavelet = np.ones((nt, 1))
    freq_list = np.ones((2,)) if dft else None
    q = wavefield(model, 0, save=True, nt=nt, name="qwf") if full_q else 0
    wsrc = Function(name='src_weight', grid=model.grid, space_order=0) if ws else None

    # Setting forward wavefield
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub)

    # Expression for saving wavefield if time subsampling is used
    eq_save = save_subsampled(model, u, nt, t_sub)

    # Add extended source
    q = extented_src(model, wsrc, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, q=q, f0=Constant('f0'))

    # Setup source and receiver
    g_expr = geom_expr(model, u, src_coords=scords, nt=nt,
                       rec_coords=rcords, wavelet=wavelet)

    # On-the-fly Fourier
    dft = otf_dft(u, freq_list, model.grid.time_dim.spacing, factor=dft_sub)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + dft + g_expr + eq_save,
                  subs=subs, name="forward"+name(model),
                  opt=opt_op(model))
    op.cfunction
    return op


@memoized_func
def adjoint_op(p_params, tti, visco, space_order, spacing, save, nv_weights, fs,
               pt_src, pt_rec, dft, dft_sub, ws, full_q):
    """
    Low level adjoint operator creation, to be used through `propagators.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    info("Building adjoint operator")
    # Some small dummy dims
    model = EmptyModel(tti, visco, spacing, fs, space_order, p_params)
    nt = 10
    ndim = len(spacing)
    scords = np.ones((1, ndim)) if pt_src else None
    rcords = np.ones((1, ndim)) if pt_rec else None
    wavelet = np.ones((nt, 1))
    freq_list = np.ones((2,)) if dft else None
    q = wavefield(model, 0, save=True, nt=nt, fw=False, name="qwf") if full_q else 0

    # Setting adjoint wavefield
    v = wavefield(model, space_order, save=save, nt=nt, fw=False)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, q=q, fw=False, f0=Constant('f0'))

    # On-the-fly Fourier
    dft = otf_dft(v, freq_list, model.grid.time_dim.spacing, factor=dft_sub)

    # Setup source and receiver
    g_expr = geom_expr(model, v, src_coords=rcords, nt=nt,
                       rec_coords=scords, wavelet=wavelet, fw=False)

    # Extended source
    wsrc = extended_rec(model, wavelet if ws else None, v)

    # Wavefield norm
    nv_t, nv_s = weighted_norm(v, weight=nv_weights) if nv_weights else ([], [])

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + wsrc + nv_t + dft + g_expr + nv_s,
                  subs=subs, name="adjoint"+name(model),
                  opt=opt_op(model))
    op.cfunction
    return op


@memoized_func
def born_op(p_params, tti, visco, space_order, spacing, save, pt_src,
            pt_rec, fs, t_sub, ws, dft, dft_sub, isic, nlind):
    """
    Low level born operator creation, to be used through `interface.py`
    Compute linearized wavefield U = J(m)* δ m
    and related quantities.
    """
    info("Building born operator")
    # Some small dummy dims
    model = EmptyModel(tti, visco, spacing, fs, space_order, p_params)
    nt = 10
    ndim = len(spacing)
    wavelet = np.ones((nt, 1))
    scords = np.ones((1, ndim)) if pt_src else None
    rcords = np.ones((1, ndim)) if pt_rec else None
    freq_list = np.ones((2,)) if dft else None
    wsrc = Function(name='src_weight', grid=model.grid, space_order=0) if ws else None
    f0 = Constant('f0')

    # Setting wavefield
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub)
    ul = wavefield(model, space_order, name="l")

    # Expression for saving wavefield if time subsampling is used
    eq_save = save_subsampled(model, u, nt, t_sub)

    # Add extended source
    q = extented_src(model, wsrc, wavelet)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, q=q, f0=f0)
    if getattr(model, 'dm', 0) == 0:
        pdel = []
    else:
        pdel = wave_kernel(model, ul, q=lin_src(model, u, isic=isic), f0=f0)
    # Setup source and receiver
    g_expr = geom_expr(model, u, rec_coords=rcords if nlind else None,
                       src_coords=scords, wavelet=wavelet)
    g_exprl = geom_expr(model, ul, rec_coords=rcords, nt=nt)

    # On-the-fly Fourier
    dft = otf_dft(u, freq_list, model.critical_dt, factor=dft_sub)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + g_expr + g_exprl + pdel + dft + eq_save,
                  subs=subs, name="born"+name(model),
                  opt=opt_op(model))
    op.cfunction
    return op


@memoized_func
def adjoint_born_op(p_params, tti, visco, space_order, spacing, pt_rec, fs, w,
                    save, t_sub, dft, dft_sub, isic):
    """
    Low level gradient operator creation, to be used through `propagators.py`
    Compute the action of the adjoint Jacobian onto a residual J'* δ d.
    """
    info("Building adjoint born operator")
    model = EmptyModel(tti, visco, spacing, fs, space_order, p_params)
    nt = 10
    ndim = len(spacing)
    residual = np.ones((nt, 1))
    rcords = np.ones((1, ndim)) if pt_rec else None
    freq_list = np.ones((2,)) if dft else None
    # Setting adjoint wavefieldgradient
    v = wavefield(model, space_order, fw=False)
    u = forward_wavefield(model, space_order, save=save, nt=nt, dft=dft, t_sub=t_sub)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, fw=False, f0=Constant('f0'))

    # Setup source and receiver
    r_expr = geom_expr(model, v, src_coords=rcords, wavelet=residual, fw=False)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    g_expr = grad_expr(gradm, u, v, model, w=w, freq=freq_list,
                       dft_sub=dft_sub, isic=isic)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + r_expr + g_expr, subs=subs, name="gradient"+name(model),
                  opt=opt_op(model))
    try:
        op.cfunction
    except:
        op = Operator(pde + r_expr + g_expr,
                      subs=subs, name="gradient"+name(model),
                      opt='advanced')
    return op
