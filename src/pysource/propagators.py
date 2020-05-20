from kernels import wave_kernel
from geom_utils import src_rec
from wave_utils import (wf_as_src, wavefield, otf_dft, extended_src_weights,
                        extented_src, wavefield_subsampled)
from sensitivity import grad_expr, lin_src

from devito import Operator, Function
from devito.tools import as_tuple


def name(model):
    return "tti" if model.is_tti else ""


# Forward propagation
def forward(model, src_coords, rcv_coords, wavelet, space_order=8, save=False,
            q=None, free_surface=False, return_op=False, freq_list=None, dft_sub=None,
            ws=None, t_sub=1):
    """
    Low level propagator, to be used through `interface.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Number of time steps
    nt = as_tuple(q)[0].shape[0] if wavelet is None else wavelet.shape[0]

    # Setting forward wavefield
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub)

    # Expression for saving wavefield if time subsampling is used
    u_save, eq_save = wavefield_subsampled(model, u, nt, t_sub)

    # Add extended source
    q = q or wf_as_src(u, w=0)
    q = extented_src(model, ws, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde, fs = wave_kernel(model, u, q=q, fs=free_surface)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, u, src_coords=src_coords, nt=nt,
                                rec_coords=rcv_coords, wavelet=wavelet)

    # On-the-fly Fourier
    dft, dft_modes = otf_dft(u, freq_list, model.critical_dt, factor=dft_sub)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(fs + pde + geom_expr + dft + eq_save,
                  subs=subs, name="forward"+name(model))

    if return_op:
        return op, u, rcv

    op()

    # Output
    return getattr(rcv, 'data', None), dft_modes or (u_save if t_sub > 1 else u)


def adjoint(model, y, src_coords, rcv_coords, space_order=8, q=0,
            save=False, free_surface=False, ws=None):
    """
    Low level propagator, to be used through `interface.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Number of time steps
    nt = as_tuple(q)[0].shape[0] if y is None else y.shape[0]

    # Setting adjoint wavefield
    v = wavefield(model, space_order, save=save, nt=nt, fw=False)

    # Set up PDE expression and rearrange
    pde, fs = wave_kernel(model, v, q=q, fw=False, fs=free_surface)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, v, src_coords=rcv_coords, nt=nt,
                                rec_coords=src_coords, wavelet=y, fw=False)

    # Extended source
    wsrc, ws_expr = extended_src_weights(model, ws, v)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + ws_expr + fs,
                  subs=subs, name="adjoint"+name(model))
    op()

    # Output
    if wsrc:
        return wsrc
    return getattr(rcv, 'data', None), v


def gradient(model, residual, rcv_coords, u, return_op=False, space_order=8, t_sub=1,
             w=None, free_surface=False, freq=None, dft_sub=None, isic=True):
    """
    Low level propagator, to be used through `interface.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Setting adjoint wavefieldgradient
    v = wavefield(model, space_order, fw=False)

    # Set up PDE expression and rearrange
    pde, fs = wave_kernel(model, v, fw=False, fs=free_surface)

    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, v, src_coords=rcv_coords,
                              wavelet=residual, fw=False)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    g_expr = grad_expr(gradm, u, v, model, w=w, freq=freq, dft_sub=dft_sub, isic=isic)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + g_expr + fs,
                  subs=subs, name="gradient"+name(model))

    if return_op:
        return op, gradm, v
    op()

    # Output
    return gradm.data


def born(model, src_coords, rcv_coords, wavelet, space_order=8,
         save=False, q=None, free_surface=False, isic=False, ws=None):
    """
    Low level propagator, to be used through `interface.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Setting adjoint wavefield
    u = wavefield(model, space_order, save=save, nt=wavelet.shape[0])
    ul = wavefield(model, space_order, name="l")

    # Extended source
    q = q or wf_as_src(u, w=0)
    q = extented_src(model, ws, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde, fsu = wave_kernel(model, u, fs=free_surface, q=q)
    pdel, fsul = wave_kernel(model, ul, q=lin_src(model, u, isic=isic), fs=free_surface)

    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, u, src_coords=src_coords, wavelet=wavelet)
    geom_exprl, _, rcvl = src_rec(model, ul, rec_coords=rcv_coords, nt=wavelet.shape[0])

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + fsu + geom_expr + pdel + geom_exprl + fsul,
                  subs=subs, name="born"+name(model))
    op()
    # Output
    return rcvl.data, u
