import numpy as np

from devito import warning
from devito.tools import as_tuple
from pyrevolve import Revolver

from checkpoint import CheckpointOperator, DevitoCheckpoint
from propagators import forward, born, gradient, forward_grad
from sensitivity import Loss
from sources import Receiver
from utils import weight_fun, compute_optalpha
from fields import memory_field, src_wavefield
from fields_exprs import wf_as_src


# Forward wrappers Pr*F*Ps'*q
def forward_rec(model, src_coords, wavelet, rec_coords, f0=0.015,
                illum=False, fw=True):
    """
    Modeling of a point source with receivers Pr*F*Ps^T*q.

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, I, _ = forward(model, src_coords, rec_coords, wavelet, save=False,
                           f0=f0, illum=illum, fw=fw)
    return rec.data, getattr(I, "data", None)


#  Pr*F*Pw'*w
def forward_rec_w(model, weight, wavelet, rec_coords, f0=0.015,
                  illum=False, fw=True):
    """
    Forward modeling of an extended source with receivers  Pr*F*Pw^T*w

    Parameters
    ----------
    model: Model
        Physical model
    weights: Array
        Spatial distribution of the extended source.
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, I, _ = forward(model, None, rec_coords, wavelet, save=False, ws=weight,
                           f0=f0, illum=illum, fw=fw)
    return rec.data, getattr(I, "data", None)


# F*Ps'*q
def forward_no_rec(model, src_coords, wavelet, f0=0.015, illum=False,
                   fw=True):
    """
    Forward modeling of a point source without receiver.

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    wavelet: Array
        Source signature
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        Wavefield
    """
    _, u, I, _ = forward(model, src_coords, None, wavelet,
                         save=True, f0=f0, illum=illum, fw=fw)
    return u.data, getattr(I, "data", None)


# Pr*F*u
def forward_wf_src(model, u, rec_coords, f0=0.015, illum=False, fw=True):
    """
    Forward modeling of a full wavefield source Pr*F*u.

    Parameters
    ----------
    model: Model
        Physical model
    u: TimeFunction or Array
        Time-space dependent wavefield
    rec_coords: Array
        Coordiantes of the receiver(s)
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        Shot record
    """
    wsrc = src_wavefield(model, u, fw=True)
    rec, _, I, _ = forward(model, None, rec_coords, None,
                           qwf=wsrc, illum=illum, f0=f0, fw=fw)
    return rec.data, getattr(I, "data", None)


# F*u
def forward_wf_src_norec(model, u, f0=0.015, illum=False, fw=True):
    """
    Forward modeling of a full wavefield source without receiver F*u.

    Parameters
    ----------
    model: Model
        Physical model
    u: TimeFunction or Array
        Time-space dependent wavefield
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        Wavefield
    """
    wf_src = src_wavefield(model, u, fw=True)
    _, u, I, _ = forward(model, None, None, None, save=True,
                         qwf=wf_src, f0=f0, illum=illum, fw=fw)
    return u.data, getattr(I, "data", None)


# Pw*F'*Pr'*d_obs
def adjoint_w(model, rec_coords, data, wavelet, f0=0.015, illum=False,
              fw=True):
    """
    Adjoint/backward modeling of a shot record (receivers as source) for an
    extended source setup Pw*F^T*Pr^T*d_obs.

    Parameters
    ----------
    model: Model
        Physical model
    rec_coords: Array
        Coordiantes of the receiver(s)
    data: Array
        Shot gather
    wavelet: Array
        Time signature of the forward source for stacking along time
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        spatial distribution
    """
    w, _, I, _ = forward(model, rec_coords, None, data, wr=wavelet,
                         f0=f0, illum=illum, fw=fw)
    return w.data, getattr(I, "data", None)


# Linearized modeling ∂/∂m (Pr*F*Ps'*q)
def born_rec(model, src_coords, wavelet, rec_coords,
             ic="as", f0=0.015, illum=False, fw=True):
    """
    Linearized (Born) modeling of a point source for a model perturbation
    (square slowness) dm.

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, I, _ = born(model, src_coords, rec_coords, wavelet, save=False,
                        ic=ic, f0=f0, illum=illum, fw=fw)
    return rec.data, getattr(I, "data", None)


# ∂/∂m (Pr*F*Pw'*w)
def born_rec_w(model, weight, wavelet, rec_coords,
               ic="as", f0=0.015, illum=False, fw=True):
    """
    Linearized (Born) modeling of an extended source for a model
    perturbation (square slowness) dm with an extended source

    Parameters
    ----------
    model: Model
        Physical model
    weight: Array
        Spatial distriubtion of the extended source
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, I, _ = born(model, None, rec_coords, wavelet, save=False, ws=weight,
                        ic=ic, f0=f0, illum=illum, fw=fw)
    return rec.data, getattr(I, "data", None)


def J_adjoint(model, src_coords, wavelet, rec_coords, recin,
              is_residual=False, checkpointing=False, n_checkpoints=None, t_sub=1,
              return_obj=False, freq_list=[], dft_sub=None, ic="as", illum=False,
              ws=None, f0=0.015, born_fwd=False, nlind=False, misfit=None, fw=True):
    """
    Jacobian (adjoint fo born modeling operator) operator on a shot record
    as a source (i.e data residual). Supports three modes:
    * Checkpinting
    * Frequency compression (on-the-fly DFT)
    * Standard zero lag cross correlation over time

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    recin: Array
        Receiver data
    checkpointing: Bool
        Whether or not to use checkpointing
    n_checkpoints: Int
        Number of checkpoints for checkpointing
    maxmem: Float
        Maximum memory to use for checkpointing
    freq_list: List
        List of frequencies for on-the-fly DFT
    dft_sub: Int
        Subsampling factor for on-the-fly DFT
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    ws : Array
        Extended source spatial distribution
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation as base propagator

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    if checkpointing:
        return J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin,
                                       is_residual=is_residual, ws=ws,
                                       n_checkpoints=n_checkpoints, ic=ic, f0=f0,
                                       nlind=nlind, return_obj=return_obj, illum=illum,
                                       born_fwd=born_fwd, misfit=misfit, fw=fw)
    elif freq_list is not None:
        return J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin, ws=ws,
                              dft_sub=dft_sub, f0=f0, ic=ic,
                              freq_list=freq_list, is_residual=is_residual, nlind=nlind,
                              return_obj=return_obj, misfit=misfit, born_fwd=born_fwd,
                              illum=illum, fw=fw)
    else:
        return J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin,
                                  is_residual=is_residual, ic=ic, ws=ws, t_sub=t_sub,
                                  return_obj=return_obj,
                                  born_fwd=born_fwd, f0=f0, nlind=nlind,
                                  illum=illum, misfit=misfit, fw=fw)


def J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin,
                   freq_list=[], is_residual=False, return_obj=False, nlind=False,
                   dft_sub=None, ic="as", ws=None, born_fwd=False, f0=0.015,
                   misfit=None, illum=False, fw=True):
    """
    Jacobian (adjoint fo born modeling operator) operator on a shot record
    as a source (i.e data residual). Outputs the gradient with Frequency
    compression (on-the-fly DFT).

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    recin: Array
        Receiver data
    freq_list: List
        List of frequencies for on-the-fly DFT
    dft_sub: Int
        Subsampling factor for on-the-fly DFT
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    ws : Array
        Extended source spatial distribution
    is_residual: Bool
        Whether to treat the input as the residual or as the observed data
    born_fwd: Bool
        Whether to use the forward or linearized forward modeling operator
    nlind: Bool
        Whether to remove the non linear data from the input data. This option is
        only available in combination with `born_fwd`
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation as base propagator

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    ffunc = op_fwd_J[born_fwd]
    rec, u, Iu, _ = ffunc(model, src_coords, rec_coords, wavelet, save=False,
                          freq_list=freq_list, ic=ic, ws=ws,
                          dft_sub=dft_sub, nlind=nlind, illum=illum, f0=f0, fw=fw)
    # Residual and gradient
    f, residual = Loss(rec, recin, model.critical_dt,
                       is_residual=is_residual, misfit=misfit)

    g, Iv, _ = gradient(model, residual, rec_coords, u, ic=ic,
                        freq=freq_list, dft_sub=dft_sub, f0=f0, illum=illum, fw=fw)
    if return_obj:
        return f, g.data, getattr(Iu, "data", None), getattr(Iv, "data", None)
    return g.data, getattr(Iu, "data", None), getattr(Iv, "data", None)


def J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin,
                       is_residual=False, return_obj=False, born_fwd=False, illum=False,
                       ic="as", ws=None, t_sub=1, nlind=False, f0=0.015, misfit=None,
                       fw=True):
    """
    Adjoint Jacobian (adjoint fo born modeling operator) operator on a shot record
    as a source (i.e data residual). Outputs the gradient with standard
    zero lag cross correlation over time.

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    recin: Array
        Receiver data
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    ws : Array
        Extended source spatial distribution
    is_residual: Bool
        Whether to treat the input as the residual or as the observed data
    born_fwd: Bool
        Whether to use the forward or linearized forward modeling operator
    nlind: Bool
        Whether to remove the non linear data from the input data. This option is
        only available in combination with `born_fwd`
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation as base propagator

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    ffunc = op_fwd_J[born_fwd]
    rec, u, Iu, _ = ffunc(model, src_coords, rec_coords, wavelet, save=True, nlind=nlind,
                          f0=f0, ws=ws, illum=illum, ic=ic,
                          t_sub=t_sub, fw=fw)

    # Residual and gradient
    f, residual = Loss(rec, recin, model.critical_dt,
                       is_residual=is_residual, misfit=misfit)

    g, Iv, _ = gradient(model, residual, rec_coords, u, ic=ic,
                        f0=f0, illum=illum, fw=fw)

    if return_obj:
        return f, g.data, getattr(Iu, "data", None), getattr(Iv, "data", None)
    return g.data, getattr(Iu, "data", None), getattr(Iv, "data", None)


def J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin,
                            is_residual=False, n_checkpoints=None, born_fwd=False,
                            return_obj=False, ic="as", ws=None, nlind=False, f0=0.015,
                            misfit=None, illum=False, fw=True):
    """
    Jacobian (adjoint fo born modeling operator) operator on a shot record
    as a source (i.e data residual). Outputs the gradient with Checkpointing.

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    wavelet: Array
        Source signature
    rec_coords: Array
        Coordiantes of the receiver(s)
    recin: Array
        Receiver data
    checkpointing: Bool
        Whether or not to use checkpointing
    n_checkpoints: Int
        Number of checkpoints for checkpointing
    maxmem: Float
        Maximum memory to use for checkpointing
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    ws : Array
        Extended source spatial distribution
    is_residual: Bool
        Whether to treat the input as the residual or as the observed data
    born_fwd: Bool
        Whether to use the forward or linearized forward modeling operator
    nlind: Bool
        Whether to remove the non linear data from the input data. This option is
        only available in combination with `born_fwd`
    f0: float
        peak frequency
    illum: bool
        Whether to compute illumination during propagation
    fw: bool
        Whether it is forward or adjoint propagation as base propagator

    Returns
    ----------
     Array
        Adjoint jacobian on the input data (gradient)
    """
    ffunc = op_fwd_J[born_fwd]
    # Optimal checkpointing
    op_f, u, rec_g, kwu = ffunc(model, src_coords, rec_coords, wavelet, fw=fw,
                                save=False, return_op=True,
                                ic=ic, nlind=nlind, ws=ws, f0=f0, illum=illum)
    op, g, kwg = gradient(model, recin, rec_coords, u,
                          return_op=True, ic=ic, f0=f0, save=False, illum=illum,
                          fw=fw)

    nt = wavelet.shape[0]
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    kwg['srcv1' if model.is_tti else 'srcv'] = rec

    # Wavefields to checkpoint
    cpwf = [uu for uu in as_tuple(u)]
    if model.is_viscoacoustic:
        r = memory_field(u)
        cpwf.append(r)
        kwu.update({r.name: r})
    cp = DevitoCheckpoint(cpwf)

    # Wrapped ops
    wrap_fw = CheckpointOperator(op_f, **kwu)
    wrap_rev = CheckpointOperator(op, **kwg)

    # Run forward
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)
    wrp.apply_forward()

    # Residual and gradient
    f, _ = Loss(rec_g, recin, model.critical_dt, is_residual=is_residual,
                misfit=misfit)
    rec.data[:] = as_tuple(rec_g)[0].data[:]

    wrp.apply_reverse()

    Iu = getattr(kwu.get("Iu", None), "data", None)
    Iv = getattr(kwg.get("Iv", None), "data", None)
    if return_obj:
        return f, g.data, Iu, Iv
    return g.data, Iu, Iv


op_fwd_J = {False: forward, True: born}


def wri_func(model, src_coords, wavelet, rec_coords, recin, yin,
             ic="as", ws=None, t_sub=1, grad="m", grad_corr=False,
             alpha_op=False, w_fun=None, eps=0, freq_list=[], wfilt=None, f0=0.015):
    """
    Time domain wavefield reconstruction inversion wrapper
    """
    if freq_list is not None:
        if grad_corr or grad in ["all", "y"]:
            warning("On-the-fly DFT is not supported with gradient correction")
        dft = True
    else:
        dft = False
        freq_list = None
        wfilt = wavelet

    # F(m0) * q if y is not an input and compute y = r(m0)
    if yin is None or grad_corr:
        y, u0, _, _ = forward(model, src_coords, rec_coords, wavelet, save=grad_corr,
                              ws=ws, f0=f0)
        ydat = recin[:] - y.data[:]
    else:
        ydat = yin

    # Compute wavefield vy = adjoint(F(m0))*y and norm on the fly
    srca, v, norm_v, _, _ = forward(model, rec_coords, src_coords, ydat,
                                    norm_wf=True, w_fun=w_fun, freq_list=freq_list,
                                    save=not (grad is None or dft), f0=f0, fw=False)
    c1 = 1 / (recin.shape[1])
    c2 = np.log(np.prod(model.shape))
    # <PTy, d-F(m)*f> = <PTy, d>-<adjoint(F(m))*PTy, f>
    ndt = np.sqrt(model.critical_dt)
    PTy_dot_r = ndt**2 * (np.dot(ydat.reshape(-1), recin.reshape(-1)) -
                          np.dot(srca.data.reshape(-1), wavelet.reshape(-1)))
    norm_y = ndt * np.linalg.norm(ydat)

    # alpha
    α = compute_optalpha(c2*norm_y, c1*norm_v, eps, comp_alpha=alpha_op)

    # Lagrangian evaluation
    fun = -.5 * c1 * α**2 * norm_v + c2 * α * PTy_dot_r - eps * np.abs(α) * norm_y

    gradm = grady = None
    if grad is not None:
        w = weight_fun(w_fun, model, src_coords)
        w = c1*α/w**2 if w is not None else c1*α
        Q = wf_as_src(v, w=w, freq_list=freq_list)
        rcv, gradm, _ = forward_grad(model, src_coords, rec_coords, c2*wfilt,
                                     freq=freq_list, q=Q, v=v, f0=f0)

        # Compute gradient wrt y
        if grad_corr or grad in ["all", "y"]:
            grady = c2 * recin - rcv.data[:]
            if norm_y != 0:
                grady -= np.abs(eps) * ydat / norm_y

        # Correcting for reduced gradient
        if not grad_corr:
            gradm = gradm.data
        else:
            gradm_corr, _, _ = gradient(model, grady, rec_coords, u0, f0=f0)
            # Reduced gradient post-processing
            gradm = gradm.data + gradm_corr.data

    return fun, gradm if gradm is None else α * gradm, grady
