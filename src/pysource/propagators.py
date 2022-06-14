from kernels import wave_kernel
from geom_utils import src_rec
from fields import (fourier_modes, wavefield, lr_src_fields,
                    wavefield_subsampled)
from fields_exprs import extented_src
from sensitivity import grad_expr
from utils import weight_fun, opt_op, fields_kwargs
from operators import forward_op, adjoint_op, born_op, adjoint_born_op

from devito import Operator, Function, Constant
from devito.tools import as_tuple


# Forward propagation
def forward(model, src_coords, rcv_coords, wavelet, space_order=8, save=False,
            qwf=None, return_op=False, freq_list=None, dft_sub=None,
            ws=None, t_sub=1, f0=0.015, **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Number of time steps
    nt = as_tuple(qwf)[0].shape[0] if wavelet is None else wavelet.shape[0]

    # Setting forward wavefield
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub)

    # Setup source and receiver
    src, rcv = src_rec(model, u, src_coords, rcv_coords, wavelet, nt)

    # Create operator and run
    op = forward_op(model.physical_parameters, model.is_tti, model.is_viscoacoustic,
                    space_order, model.spacing, save, t_sub, model.fs,
                    src_coords is not None, rcv_coords is not None,
                    freq_list is not None, dft_sub, ws is not None, qwf is not None)

    if return_op:
        return op, u, rcv

    # Make kwargs
    kw = {'dt': model.critical_dt}
    f0q = Constant('f0', value=f0) if model.is_viscoacoustic else None

    # Expression for saving wavefield if time subsampling is used
    u_save = wavefield_subsampled(model, u, nt, t_sub)

    # On-the-fly Fourier
    dft_modes, fr = fourier_modes(u, freq_list)

    # Extended source
    ws, wt = lr_src_fields(model, ws, wavelet)

    # Update kwargs
    kw.update(fields_kwargs(u, src, rcv, u_save, dft_modes, fr, ws, wt, f0q))
    kw.update(model.physical_params())

    summary = op(**kw)

    # Output
    return rcv, dft_modes or (u_save if t_sub > 1 else u), summary


def adjoint(model, y, src_coords, rcv_coords, space_order=8, qwf=None, dft_sub=None,
            save=False, ws=None, norm_v=False, w_fun=None, freq_list=None, f0=0.015):
    """
    Low level propagator, to be used through `interface.py`
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Number of time steps
    nt = as_tuple(qwf)[0].shape[0] if y is None else y.shape[0]

    # Setting adjoint wavefield
    v = wavefield(model, space_order, save=save, nt=nt, fw=False)

    # Setup source and receiver
    src, rcv = src_rec(model, v, src_coords=rcv_coords, nt=nt,
                       rec_coords=src_coords, wavelet=y)
    # Wavefield norm
    nv_weights = weight_fun(w_fun, model, src_coords) if norm_v else None

    # Create operator and run
    op = adjoint_op(model.physical_parameters, model.is_tti, model.is_viscoacoustic,
                    space_order, model.spacing, save, nv_weights, model.fs,
                    src_coords is not None, rcv_coords is not None,
                    freq_list is not None, dft_sub, ws is not None, qwf is not None)

    # On-the-fly Fourier
    dft_modes, fr = fourier_modes(v, freq_list)

    # Extended source
    ws, wt = lr_src_fields(model, None, ws, empty_ws=True)

    # Update kwargs
    kw = {'dt': model.critical_dt}
    f0q = Constant('f0', value=f0) if model.is_viscoacoustic else None
    kw.update(fields_kwargs(v, src, rcv, dft_modes, fr, ws, wt, f0q))
    kw.update(model.physical_params())

    # Run op
    summary = op(**kw)

    # Output
    if ws:
        return ws, summary
    if norm_v:
        return rcv, dft_modes or v, norm_v.data[0], summary
    return rcv, v, summary


def gradient(model, residual, rcv_coords, u, return_op=False, space_order=8,
             w=None, freq=None, dft_sub=None, isic=False, f0=0.015):
    """
    Low level propagator, to be used through `interface.py`
    Compute the action of the adjoint Jacobian onto a residual J'* δ d.
    """
    # Setting adjoint wavefieldgradient
    v = wavefield(model, space_order, fw=False)
    t_sub = getattr(u.indices[0], '_factor', 1)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)

    # Setup source and receiver
    src, _ = src_rec(model, v, src_coords=rcv_coords, wavelet=residual)

    # Create operator and run
    op = adjoint_born_op(model.physical_parameters, model.is_tti, model.is_viscoacoustic,
                         space_order, model.spacing, rcv_coords is not None, model.fs, w,
                         t_sub, freq is not None, dft_sub, isic)
    if return_op:
        return op, gradm, v

    # Update kwargs
    kw = {'dt': model.critical_dt}
    f0q = Constant('f0', value=f0) if model.is_viscoacoustic else None
    kw.update(fields_kwargs(v, src, u, v, gradm, f0q))
    kw.update(model.physical_params())

    summary = op(**kw)

    # Output
    return gradm, summary


def born(model, src_coords, rcv_coords, wavelet, space_order=8, save=False,
         q=None, return_op=False, isic=False, freq_list=None, dft_sub=None,
         ws=None, t_sub=1, nlind=False, f0=0.015):
    """
    Low level propagator, to be used through `interface.py`
    Compute linearized wavefield U = J(m)* δ m
    and related quantities.
    """
    nt = wavelet.shape[0]
    # Setting wavefield
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub)
    ul = wavefield(model, space_order, name="l")

    # Setup source and receiver
    snl, rnl = src_rec(model, u, rec_coords=rcv_coords if nlind else None,
                       src_coords=src_coords, wavelet=wavelet)
    _, rcvl = src_rec(model, ul, rec_coords=rcv_coords, nt=nt)

    # Create operator and run
    op = born_op(model.physical_parameters, model.is_tti, model.is_viscoacoustic,
                 space_order, model.spacing, save,
                 src_coords is not None, rcv_coords is not None, model.fs, t_sub,
                 ws, freq_list is not None, dft_sub, isic, nlind)

    outrec = (rcvl, rnl) if nlind else rcvl
    if return_op:
        return op, u, outrec

    # Make kwargs
    kw = {'dt': model.critical_dt}
    f0q = Constant('f0', value=f0) if model.is_viscoacoustic else None
    # Expression for saving wavefield if time subsampling is used
    u_save = wavefield_subsampled(model, u, nt, t_sub)

    # On-the-fly Fourier
    dft_modes, fr = fourier_modes(u, freq_list)

    # Extended source
    ws, wt = lr_src_fields(model, ws, wavelet)

    # Update kwargs
    kw.update(fields_kwargs(u, ul, snl, rnl, rcvl, u_save, dft_modes, fr, ws, wt, f0q))
    kw.update(model.physical_params())

    summary = op(**kw)

    # Output
    return outrec, dft_modes or (u_save if t_sub > 1 else u), summary


# Forward propagation
def forward_grad(model, src_coords, rcv_coords, wavelet, v, space_order=8,
                 q=None, ws=None, isic=False, w=None, freq=None, f0=0.015, **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Number of time steps
    nt = as_tuple(q)[0].shape[0] if wavelet is None else wavelet.shape[0]

    # Setting forward wavefield
    u = wavefield(model, space_order, save=False)

    # Add extended source
    q = q or 0
    q = extented_src(model, ws, wavelet, q=q)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, q=q, f0=f0)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, u, src_coords=src_coords, nt=nt,
                                rec_coords=rcv_coords, wavelet=wavelet)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    g_expr = grad_expr(gradm, v, u, model, w=w, isic=isic, freq=freq)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + g_expr,
                  subs=subs, name="forward_grad",
                  opt=opt_op(model))

    summary = op()

    # Output
    return rcv, gradm, summary
