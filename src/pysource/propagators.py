from kernels import wave_kernel
from geom_utils import src_rec, geom_expr
from fields import (fourier_modes, wavefield, lr_src_fields, illumination,
                    wavefield_subsampled, norm_holder, frequencies)
from fields_exprs import extented_src
from sensitivity import grad_expr
from utils import weight_fun, opt_op, fields_kwargs, nfreq, base_kwargs
from operators import forward_op, born_op, adjoint_born_op

from devito import Operator, Function, Constant
from devito.tools import as_tuple


# Forward propagation
def forward(model, src_coords, rcv_coords, wavelet, space_order=8, save=False,
            qwf=None, return_op=False, freq_list=None, dft_sub=None,
            norm_wf=False, w_fun=None, ws=None, wr=None, t_sub=1, f0=0.015,
            illum=False, fw=True, **kwargs):
    """
    Low level propagator, to be used through `interface.py`
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Number of time steps
    nt = as_tuple(qwf)[0].shape[0] if wavelet is None else wavelet.shape[0]

    # Setting forward wavefield
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub, fw=fw)

    # Setup source and receiver
    src, rcv = src_rec(model, u, src_coords, rcv_coords, wavelet, nt)

    # Wavefield norm
    nv_weights = weight_fun(w_fun, model, src_coords) if norm_wf else None

    # Create operator and run
    op = forward_op(model.physical_parameters, model.is_tti, model.is_viscoacoustic,
                    model.is_elastic, space_order, fw, model.spacing, save,
                    t_sub, model.fs, src_coords is not None, rcv_coords is not None,
                    nfreq(freq_list), dft_sub, ws is not None,
                    wr is not None, qwf is not None, nv_weights, illum,
                    model.abc_type)

    # Make kwargs
    kw = base_kwargs(model.critical_dt)
    f0q = Constant('f0', value=f0) if model.is_viscoacoustic else None

    # Expression for saving wavefield if time subsampling is used
    u_save = wavefield_subsampled(model, u, nt, t_sub, space_order=space_order)

    # Illumination
    I = illumination(u, illum)

    # On-the-fly Fourier
    dft_modes, fr = fourier_modes(u, freq_list)

    # Extended source
    ws, wst = lr_src_fields(model, ws, wavelet)

    # Extended receiver
    wr, wrt = lr_src_fields(model, None, wr, empty_w=True, rec=True)

    # Norm v
    nv2, nvt2 = norm_holder(u) if norm_wf else (None, None)

    # Update kwargs
    fields = fields_kwargs(u, qwf, src, rcv, u_save, dft_modes, fr, ws, wst,
                           wr, wrt, nv2, nvt2, f0q, I)
    kw.update(fields)
    kw.update(model.physical_params())

    # Output
    rout = wr or rcv
    uout = dft_modes or (u_save if t_sub > 1 else u)

    if return_op:
        return op, uout, rout, kw

    summary = op(**kw)

    if norm_wf:
        return rout, uout, nv2.data[0], I, summary
    return rout, uout, I, summary


# legacy
def adjoint(*args, **kwargs):
    fw = not kwargs.pop('fw', True)
    return forward(*args, fw=fw, **kwargs)


def gradient(model, residual, rcv_coords, u, return_op=False, space_order=8, fw=True,
             w=None, freq=None, dft_sub=None, ic="as", f0=0.015, save=True, illum=False):
    """
    Low level propagator, to be used through `interface.py`
    Compute the action of the adjoint Jacobian onto a residual J'* δ d.
    """
    # Setting adjoint wavefieldgradient
    v = wavefield(model, space_order, fw=not fw)
    try:
        t_sub = as_tuple(u)[0].indices[0]._factor
    except AttributeError:
        t_sub = 1

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)

    # Setup source and receiver
    src, _ = src_rec(model, v, src_coords=rcv_coords, wavelet=residual)

    # Create operator and run
    op = adjoint_born_op(model.physical_parameters, model.is_tti, model.is_viscoacoustic,
                         model.is_elastic, space_order, fw, model.spacing,
                         rcv_coords is not None, model.fs, w, save, t_sub, nfreq(freq),
                         dft_sub, ic, illum, model.abc_type)

    # Update kwargs
    kw = base_kwargs(model.critical_dt)
    f, _ = frequencies(freq)
    f0q = Constant('f0', value=f0) if model.is_viscoacoustic else None

    # Illumination
    I = illumination(v, illum)

    kw.update(fields_kwargs(src, u, v, gradm, f0q, f, I))
    kw.update(model.physical_params())

    if return_op:
        return op, gradm, kw

    summary = op(**kw)

    # Output
    return gradm, I, summary


def born(model, src_coords, rcv_coords, wavelet, space_order=8, save=False,
         qwf=None, return_op=False, ic="as", freq_list=None, dft_sub=None,
         ws=None, t_sub=1, nlind=False, f0=0.015, illum=False, fw=True):
    """
    Low level propagator, to be used through `interface.py`
    Compute linearized wavefield U = J(m)* δ m
    and related quantities.
    """
    nt = wavelet.shape[0]

    # Wavefields
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub, fw=fw)
    ul = wavefield(model, space_order, name="l", fw=fw)

    # Illumination
    I = illumination(u, illum)

    # Setup source and receiver
    snl, rnl = src_rec(model, u, rec_coords=rcv_coords if nlind else None,
                       src_coords=src_coords, wavelet=wavelet)
    _, rcvl = src_rec(model, ul, rec_coords=rcv_coords, nt=nt)
    outrec = (rcvl, rnl) if nlind else rcvl

    # If the perturbation is zero, run only the forward and return zero data
    if getattr(model, 'dm', 0) == 0:
        op, u, _, kw = forward(model, src_coords, rcv_coords, wavelet,
                               space_order=space_order, save=save,
                               qwf=qwf, return_op=True, freq_list=freq_list,
                               dft_sub=dft_sub, ws=ws, t_sub=t_sub, f0=f0, illum=illum)

        kw.update(fields_kwargs(rnl, I))
        if return_op:
            return op, u, outrec, kw

        summary = op(**kw)
        return outrec, u, I, summary

    # Create operator and run
    op = born_op(model.physical_parameters, model.is_tti, model.is_viscoacoustic,
                 model.is_elastic, space_order, fw, model.spacing, save,
                 src_coords is not None, rcv_coords is not None, model.fs, t_sub,
                 ws is not None, nfreq(freq_list), dft_sub, ic, nlind, illum,
                 model.abc_type)

    # Make kwargs
    kw = base_kwargs(model.critical_dt)
    f0q = Constant('f0', value=f0) if model.is_viscoacoustic else None
    # Expression for saving wavefield if time subsampling is used
    u_save = wavefield_subsampled(model, u, nt, t_sub, space_order=space_order)

    # On-the-fly Fourier
    dft_m, fr = fourier_modes(u, freq_list)

    # Extended source
    ws, wt = lr_src_fields(model, ws, wavelet)

    # Update kwargs
    kw.update(fields_kwargs(u, ul, snl, rnl, rcvl, u_save, dft_m, fr, ws, wt, f0q, I))
    kw.update(model.physical_params(born=True))

    if return_op:
        return op, u, outrec, kw

    summary = op(**kw)

    # Output
    return outrec, dft_m or (u_save if t_sub > 1 else u), I, summary


# Forward propagation
def forward_grad(model, src_coords, rcv_coords, wavelet, v, space_order=8,
                 q=None, ws=None, ic="as", w=None, freq=None, f0=0.015, **kwargs):
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
    rexpr = geom_expr(model, u, src_coords=src_coords, nt=nt,
                      rec_coords=rcv_coords, wavelet=wavelet)
    _, rcv = src_rec(model, u, src_coords, rcv_coords, wavelet, nt)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    g_expr = grad_expr(gradm, v, u, model, w=w, ic=ic, freq=freq)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + rexpr + g_expr,
                  subs=subs, name="forward_grad",
                  opt=opt_op(model))

    kw = base_kwargs(model.critical_dt)
    summary = op(rcvu=rcv, **kw)

    # Output
    return rcv, gradm, summary
