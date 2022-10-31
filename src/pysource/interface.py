import numpy as np

from devito import warning
from devito.tools import as_tuple
from pyrevolve import Revolver

from checkpoint import CheckpointOperator, DevitoCheckpoint
from propagators import forward, adjoint, born, gradient, forward_grad
from sensitivity import Loss
from sources import Receiver
from utils import weight_fun, compute_optalpha
from fields import memory_field, src_wavefield
from fields_exprs import wf_as_src


# Forward wrappers Pr*F*Ps'*q
def forward_rec(model, src_coords, wavelet, rec_coords, space_order=8, f0=0.015):
    """
    Forward modeling of a point source with receivers Pr*F*Ps^T*q.

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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, _ = forward(model, src_coords, rec_coords, wavelet, save=False,
                        space_order=space_order, f0=f0)
    return rec.data


#  Pr*F*Pw'*w
def forward_rec_w(model, weight, wavelet, rec_coords, space_order=8, f0=0.015):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, _ = forward(model, None, rec_coords, wavelet, save=False, ws=weight,
                        space_order=space_order, f0=f0)
    return rec.data


# F*Ps'*q
def forward_no_rec(model, src_coords, wavelet, space_order=8, f0=0.015):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Wavefield
    """
    _, u, _ = forward(model, src_coords, None, wavelet, space_order=space_order,
                      save=True, f0=f0)
    return u.data


# Pr*F*u
def forward_wf_src(model, u, rec_coords, space_order=8, f0=0.015):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Shot record
    """
    wsrc = src_wavefield(model, u, fw=True)
    rec, _, _ = forward(model, None, rec_coords, None, space_order=space_order, qwf=wsrc)
    return rec.data


# F*u
def forward_wf_src_norec(model, u, space_order=8, f0=0.015):
    """
    Forward modeling of a full wavefield source without receiver F*u.

    Parameters
    ----------
    model: Model
        Physical model
    u: TimeFunction or Array
        Time-space dependent wavefield
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Wavefield
    """
    wf_src = src_wavefield(model, u, fw=True)
    _, u, _ = forward(model, None, None, None, space_order=space_order, save=True,
                      qwf=wf_src, f0=f0)
    return u.data


# Adjoint wrappers Ps*F'*Pr'*d_obs
def adjoint_rec(model, src_coords, rec_coords, data,
                space_order=8, f0=0.015):
    """
    Adjoint/backward modeling of a shot record (receivers as source) Ps*F^T*Pr^T*d.

    Parameters
    ----------
    model: Model
        Physical model
    src_coords: Array
        Coordiantes of the source(s)
    rec_coords: Array
        Coordiantes of the receiver(s)
    data: Array
        Shot gather
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Shot record (adjoint wavefield at source position(s))
    """
    rec, _, _ = adjoint(model, data, src_coords, rec_coords, space_order=space_order,
                        f0=f0)
    return rec.data


# Pw*F'*Pr'*d_obs
def adjoint_w(model, rec_coords, data, wavelet, space_order=8, f0=0.015):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        spatial distribution
    """
    w, _ = adjoint(model, data, None, rec_coords, ws=wavelet, space_order=space_order,
                   f0=f0)
    return w.data


# F'*Pr'*d_obs
def adjoint_no_rec(model, rec_coords, data, space_order=8, f0=0.015):
    """
    Adjoint/backward modeling of a shot record (receivers as source)
    without source sampling F^T*Pr^T*d_obs.

    Parameters
    ----------
    model: Model
        Physical model
    rec_coords: Array
        Coordiantes of the receiver(s)
    data: Array
        Shot gather
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Adjoint wavefield
    """
    _, v, _ = adjoint(model, data, None, rec_coords, space_order=space_order, save=True,
                      f0=f0)
    return v.data


# Ps*F'*u
def adjoint_wf_src(model, u, src_coords, space_order=8, f0=0.015):
    """
    Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source)
    Ps*F^T*u.

    Parameters
    ----------
    model: Model
        Physical model
    u: Array or TimeFunction
        Time-space dependent source
    src_coords: Array
        Source coordinates
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Shot record (sampled at source position(s))
    """
    wsrc = src_wavefield(model, u, fw=False)
    rec, _, _ = adjoint(model, None, src_coords, None, space_order=space_order, qwf=wsrc)
    return rec.data


# F'*u
def adjoint_wf_src_norec(model, u, space_order=8, f0=0.015):
    """
    Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source)
    F^T*u.

    Parameters
    ----------
    model: Model
        Physical model
    u: Array or TimeFunction
        Time-space dependent source
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        Adjoint wavefield
    """
    wf_src = src_wavefield(model, u, fw=False)
    _, v, _ = adjoint(model, None, None, None, space_order=space_order,
                      save=True, qwf=wf_src, f0=f0)
    return v.data


# Linearized modeling ∂/∂m (Pr*F*Ps'*q)
def born_rec(model, src_coords, wavelet, rec_coords,
             space_order=8, ic="as", f0=0.015):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    f0: peak frequency

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, _ = born(model, src_coords, rec_coords, wavelet, save=False,
                     space_order=space_order, ic=ic, f0=f0)
    return rec.data


# ∂/∂m (Pr*F*Pw'*w)
def born_rec_w(model, weight, wavelet, rec_coords,
               space_order=8, ic="as", f0=0.015):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    ic: String
        Imaging conditions ("as", "isic" or "fwi"), defaults to "as"
    f0: peak frequency

    Returns
    ----------
    Array
        Shot record
    """
    rec, _, _ = born(model, None, rec_coords, wavelet, save=False, ws=weight,
                     space_order=space_order, ic=ic, f0=f0)
    return rec.data


# Gradient wrappers
def grad_fwi(model, recin, rec_coords, u, space_order=8, f0=0.015):
    """
    FWI gradient, i.e adjoint Jacobian on a data residual.

    Parameters
    ----------
    model: Model
        Physical model
    recin: Array
        Data residual
    rec_coords: Array
        Receivers coordinates
    u: TimeFunction
        Forward wavefield
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
    f0: peak frequency

    Returns
    ----------
    Array
        FWI gradient
    """
    g, _ = gradient(model, recin, rec_coords, u, space_order=space_order, f0=f0)
    return g.data


def J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=8,
              is_residual=False, checkpointing=False, n_checkpoints=None, t_sub=1,
              return_obj=False, freq_list=[], dft_sub=None, ic="as",
              ws=None, f0=0.015, born_fwd=False, nlind=False, misfit=None):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
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
    f0: peak frequency

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    if checkpointing:
        return J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin,
                                       space_order=8, is_residual=is_residual, ws=ws,
                                       n_checkpoints=n_checkpoints, ic=ic, f0=f0,
                                       nlind=nlind, return_obj=return_obj,
                                       born_fwd=born_fwd, misfit=misfit)
    elif freq_list is not None:
        return J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin, ws=ws,
                              space_order=space_order, dft_sub=dft_sub, f0=f0, ic=ic,
                              freq_list=freq_list, is_residual=is_residual, nlind=nlind,
                              return_obj=return_obj, misfit=misfit, born_fwd=born_fwd)
    else:
        return J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin,
                                  is_residual=is_residual, ic=ic, ws=ws, t_sub=t_sub,
                                  return_obj=return_obj, space_order=space_order,
                                  born_fwd=born_fwd, f0=f0, nlind=nlind, misfit=misfit)


def J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin, space_order=8,
                   freq_list=[], is_residual=False, return_obj=False, nlind=False,
                   dft_sub=None, ic="as", ws=None, born_fwd=False, f0=0.015,
                   misfit=None):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
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
    f0: peak frequency

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    rec, u, _ = op_fwd_J[born_fwd](model, src_coords, rec_coords, wavelet, save=False,
                                   space_order=space_order, freq_list=freq_list,
                                   ic=ic, ws=ws, dft_sub=dft_sub, nlind=nlind, f0=f0)
    # Residual and gradient
    f, residual = Loss(rec, recin, model.critical_dt,
                       is_residual=is_residual, misfit=misfit)

    g, _ = gradient(model, residual, rec_coords, u, space_order=space_order, ic=ic,
                    freq=freq_list, dft_sub=dft_sub, f0=f0)
    if return_obj:
        return f, g.data
    return g.data


def J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin, space_order=8,
                       is_residual=False, return_obj=False, born_fwd=False,
                       ic="as", ws=None, t_sub=1, nlind=False, f0=0.015, misfit=None):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
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
    f0: peak frequency

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    rec, u, _ = op_fwd_J[born_fwd](model, src_coords, rec_coords, wavelet, save=True,
                                   ws=ws, space_order=space_order,
                                   ic=ic, t_sub=t_sub, nlind=nlind, f0=f0)

    # Residual and gradient
    f, residual = Loss(rec, recin, model.critical_dt,
                       is_residual=is_residual, misfit=misfit)

    g, _ = gradient(model, residual, rec_coords, u, space_order=space_order, ic=ic,
                    f0=f0)
    if return_obj:
        return f, g.data
    return g.data


def J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin, space_order=8,
                            is_residual=False, n_checkpoints=None, born_fwd=False,
                            return_obj=False, ic="as", ws=None, nlind=False, f0=0.015,
                            misfit=None):
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
    space_order: Int (optional)
        Spatial discretization order, defaults to 8
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
    f0: peak frequency

    Returns
    ----------
     Array
        Adjoint jacobian on the input data (gradient)
    """
    # Optimal checkpointing
    op_f, u, rec_g, kwu = op_fwd_J[born_fwd](model, src_coords, rec_coords, wavelet,
                                             save=False, space_order=space_order,
                                             return_op=True, ic=ic, nlind=nlind,
                                             ws=ws, f0=f0)
    op, g, kwg = gradient(model, recin, rec_coords, u, space_order=space_order,
                          return_op=True, ic=ic, f0=f0, save=False)

    nt = wavelet.shape[0]
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    kwg['srcv'] = rec
    # Wavefields to checkpoint
    cpwf = [uu for uu in as_tuple(u)]
    if model.is_viscoacoustic:
        cpwf += [memory_field(u)]
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

    if return_obj:
        return f, g.data
    return g.data


op_fwd_J = {False: forward, True: born}


def wri_func(model, src_coords, wavelet, rec_coords, recin, yin, space_order=8,
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
        y, u0, _ = forward(model, src_coords, rec_coords, wavelet, save=grad_corr,
                           space_order=space_order, ws=ws, f0=f0)
        ydat = recin[:] - y.data[:]
    else:
        ydat = yin

    # Compute wavefield vy = adjoint(F(m0))*y and norm on the fly
    srca, v, norm_v, _ = adjoint(model, ydat, src_coords, rec_coords,
                                 norm_v=True, w_fun=w_fun, freq_list=freq_list,
                                 save=not (grad is None or dft), f0=f0)
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
            gradm_corr, _ = gradient(model, grady, rec_coords, u0, f0=f0)
            # Reduced gradient post-processing
            gradm = gradm.data + gradm_corr.data

    return fun, gradm if gradm is None else α * gradm, grady
