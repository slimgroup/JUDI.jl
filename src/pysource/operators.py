import numpy as np

from devito.tools import memoized_func
from devito import Constant, Operator, Function

from models import EmptyModel
from kernels import wave_kernel
from geom_utils import geom_expr
from fields import wavefield
from fields_exprs import (otf_dft, extended_rec,
                          extented_src, save_subsampled, weighted_norm)
from sensitivity import grad_expr, lin_src
from utils import weight_fun, opt_op


def name(model):
    if model.is_tti:
        return "tti"
    elif model.is_viscoacoustic:
        return "viscoacoustic"
    else:
        return ""


@memoized_func
def forward_op(p_params, tti, visco, space_order, spacing, save, t_sub, fs, pt_src, pt_rec, dft, dft_sub, ws, full_q):
    # Some small dummy dims
    model = EmptyModel(tti, visco, spacing, fs, space_order, p_params)
    nt = 10
    ndim = len(spacing)
    scords = np.ones((1, ndim)) if pt_src else None
    rcords = np.ones((1, ndim)) if pt_rec else None
    wavelet = np.ones((nt, 1))
    freq_list = np.ones((2,)) if dft else None
    q = wavefield(model, space_order, save=True, nt=nt, name="qwf") if full_q else None
    src_weights = Function(name='src_weight', grid=model.grid, space_order=0) if ws else None

    # Setting forward wavefield
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub)

    # Expression for saving wavefield if time subsampling is used
    eq_save = save_subsampled(model, u, nt, t_sub)

    # Add extended source
    q = extented_src(model, src_weights, wavelet, q=q)

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
def adjoint_op(p_params, tti, visco, space_order, spacing, save, nv_weights, fs, pt_src, pt_rec, dft, dft_sub, ws, full_q):
    """
    Low level propagator, to be used through `interface.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Some small dummy dims
    model = EmptyModel(tti, visco, spacing, fs, space_order, p_params)
    nt = 10
    ndim = len(spacing)
    scords = np.ones((1, ndim)) if pt_src else None
    rcords = np.ones((1, ndim)) if pt_rec else None
    wavelet = np.ones((nt, 1))
    freq_list = np.ones((2,)) if dft else None
    q = wavefield(model, space_order, save=True, nt=nt, name="qwf") if full_q else None

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
    nv_t, nv_s = ([], [])
    if nv_weights:
        (nv_t, nv_s) = weighted_norm(v, weight=nv_weights)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + wsrc + nv_t + dft + g_expr + nv_s,
                  subs=subs, name="adjoint"+name(model),
                  opt=opt_op(model))
    op.cfunction
    return op