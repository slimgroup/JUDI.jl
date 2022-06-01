import numpy as np

from devito.tools import memoized_func
from devito import Constant, Operator, Function

from models import EmptyModel
from kernels import wave_kernel
from geom_utils import src_rec
from wave_utils import (wf_as_src, wavefield, otf_dft, extended_src_weights,
                        extented_src, wavefield_subsampled, weighted_norm)
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
    _, eq_save = wavefield_subsampled(model, u, nt, t_sub)

    # Add extended source
    q = q or wf_as_src(u, w=0)
    q = extented_src(model, src_weights, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, q=q, f0=Constant('fq'))

    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, u, src_coords=scords, nt=nt,
                                rec_coords=rcords, wavelet=wavelet)

    # On-the-fly Fourier
    dft, _ = otf_dft(u, freq_list, model.critical_dt, factor=dft_sub)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + dft + geom_expr + eq_save,
                  subs=subs, name="forward"+name(model),
                  opt=opt_op(model))
    op.cfunction
    return op
