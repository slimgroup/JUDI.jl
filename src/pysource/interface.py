import numpy as np

from devito import TimeFunction, norm
from pyrevolve import Revolver

from checkpoint import CheckpointOperator, DevitoCheckpoint
from propagators import forward, adjoint, born, gradient
from sources import Receiver


# Forward wrappers Pr*F*Ps'*q
def forward_rec(model, src_coords, wavelet, rec_coords, space_order=8,
                free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Shot record
    """
    rec, _ = forward(model, src_coords, rec_coords, wavelet, save=False,
                     space_order=space_order, free_surface=free_surface)
    return rec.data


#  Pr*F*Pw'*w
def forward_rec_w(model, weight, wavelet, rec_coords, space_order=8,
                  free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Shot record
    """
    rec, _ = forward(model, None, rec_coords, wavelet, save=False, ws=weight,
                     space_order=space_order, free_surface=free_surface)
    return rec.data


# Pr*F*Ps'*q
def forward_rec_wf(model, src_coords, wavelet, rec_coords, t_sub=1,
                   space_order=8, free_surface=False):
    """
    Forward modeling of a point source Pr*F*Ps^T*q and return wavefield.

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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Shot record
    TimeFunction
        Wavefield
    """
    rec, u = forward(model, src_coords, rec_coords, wavelet, save=True, t_sub=t_sub,
                     space_order=space_order, free_surface=free_surface)
    return rec.data, u


# F*Ps'*q
def forward_no_rec(model, src_coords, wavelet, space_order=8, free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Wavefield
    """
    _, u = forward(model, src_coords, None, wavelet, space_order=space_order,
                   save=True, free_surface=free_surface)
    return u.data


# Pr*F*u
def forward_wf_src(model, u, rec_coords, space_order=8, free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Shot record
    """
    wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2,
                          space_order=space_order, save=u.shape[0])
    if isinstance(u, TimeFunction):
        wf_src._data = u._data
    else:
        wf_src.data[:] = u[:]
    rec, _ = forward(model, None, rec_coords, None, space_order=space_order,
                     free_surface=free_surface, q=wf_src)
    return rec.data


# F*u
def forward_wf_src_norec(model, u, space_order=8, free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Wavefield
    """
    wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2,
                          space_order=space_order, save=u.shape[0])
    if isinstance(u, TimeFunction):
        wf_src._data = u._data
    else:
        wf_src.data[:] = u[:]
    _, u = forward(model, None, None, None, space_order=space_order,
                   save=True, free_surface=free_surface, q=wf_src)
    return u.data


# Adjoint wrappers Ps*F'*Pr'*d_obs
def adjoint_rec(model, src_coords, rec_coords, data,
                space_order=8, free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Shot record (adjoint wavefield at source position(s))
    """
    rec, _ = adjoint(model, data, src_coords, rec_coords,
                     space_order=space_order, free_surface=free_surface)
    return rec.data


# Pw*F'*Pr'*d_obs
def adjoint_w(model, rec_coords, data, wavelet, space_order=8,
              free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        spatial distribution
    """
    w = adjoint(model, data, None, rec_coords, ws=wavelet,
                space_order=space_order, free_surface=free_surface)
    slices = [slice(model.nbl, -model.nbl) for _ in range(model.dim)]
    return w.data[slices]


# F'*Pr'*d_obs
def adjoint_no_rec(model, rec_coords, data, space_order=8,
                   free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Adjoint wavefield
    """
    _, v = adjoint(model, data, None, rec_coords, space_order=space_order,
                   save=True, free_surface=free_surface)
    return v.data


# Ps*F'*u
def adjoint_wf_src(model, u, src_coords, space_order=8, free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Shot record (sampled at source position(s))
    """
    wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2,
                          space_order=space_order, save=u.shape[0])
    if isinstance(u, TimeFunction):
        wf_src._data = u._data
    else:
        wf_src.data[:] = u[:]
    rec, _ = adjoint(model, None, src_coords, None, space_order=space_order,
                     free_surface=free_surface, q=wf_src)
    return rec.data


# F'*u
def adjoint_wf_src_norec(model, u, space_order=8, free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        Adjoint wavefield
    """
    wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2,
                          space_order=space_order, save=u.shape[0])
    if isinstance(u, TimeFunction):
        wf_src._data = u._data
    else:
        wf_src.data[:] = u[:]
    _, v = adjoint(model, None, None, None, space_order=space_order,
                   save=True, free_surface=free_surface, q=wf_src)
    return v.data


# Linearized modeling ∂/∂m (Pr*F*Ps'*q)
def born_rec(model, src_coords, wavelet, rec_coords,
             space_order=8, free_surface=False, isic=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface
    isic : Bool
        Whether or not to use ISIC imaging condition

    Returns
    ----------
    Array
        Shot record
    """
    rec, _ = born(model, src_coords, rec_coords, wavelet, save=False,
                  space_order=space_order, free_surface=free_surface, isic=isic)
    return rec.data


# ∂/∂m (Pr*F*Pw'*w)
def born_rec_w(model, weight, wavelet, rec_coords,
               space_order=8, free_surface=False, isic=False):
    """
    Linearized (Born) modeling of a point source for a model
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
    free_surface: Bool (optional)
        Whether or not to use a free surface
    isic : Bool
        Whether or not to use ISIC imaging condition

    Returns
    ----------
    Array
        Shot record
    """
    rec, _ = born(model, None, rec_coords, wavelet, save=False, ws=weight,
                  space_order=space_order, free_surface=free_surface, isic=isic)
    return rec.data


# Gradient wrappers
def grad_fwi(model, recin, rec_coords, u, space_order=8, free_surface=False):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface

    Returns
    ----------
    Array
        FWI gradient
    """
    g = gradient(model, recin, rec_coords, u, space_order=space_order,
                 free_surface=free_surface)
    return g.data


def J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=8,
              checkpointing=False, free_surface=False, n_checkpoints=None,
              maxmem=None, freq_list=[], dft_sub=None, isic=False, ws=None,
              t_sub=1):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface
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
    isic : Bool
        Whether or not to use ISIC imaging condition
    ws : Array
        Extended source spatial distribution

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    if checkpointing:
        grad = J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords,
                                       recin, space_order=8, free_surface=False,
                                       n_checkpoints=n_checkpoints, is_residual=True,
                                       maxmem=maxmem, isic=isic, ws=ws, t_sub=t_sub)
    elif len(freq_list) > 0:
        grad = J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin,
                              space_order=space_order, is_residual=True, dft_sub=dft_sub,
                              free_surface=free_surface, freq_list=freq_list,
                              isic=isic, ws=ws, t_sub=t_sub)
    else:
        grad = J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin,
                                  is_residual=True, isic=isic, ws=ws, t_sub=t_sub,
                                  space_order=space_order, free_surface=free_surface)

    return grad


def J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin, space_order=8,
                   free_surface=False, freq_list=[], is_residual=False, return_obj=False,
                   dft_sub=None, isic=False, ws=None, t_sub=1):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface
    freq_list: List
        List of frequencies for on-the-fly DFT
    dft_sub: Int
        Subsampling factor for on-the-fly DFT
    isic : Bool
        Whether or not to use ISIC imaging condition
    ws : Array
        Extended source spatial distribution
    is_residual: Bool
        Whether to treat the input as the residual or as the observed data

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    rec, u = forward(model, src_coords, rec_coords, wavelet, save=False,
                     space_order=space_order, free_surface=free_surface,
                     freq_list=freq_list, dft_sub=dft_sub, ws=ws)
    # Residual and gradient
    if not is_residual:
        recin[:] = rec.data[:] - recin[:]   # input is observed data

    g = gradient(model, recin, rec_coords, u, space_order=space_order, isic=isic,
                 free_surface=free_surface, freq=freq_list, dft_sub=dft_sub)
    if return_obj:
        return .5*model.critical_dt*np.linalg.norm(recin)**2, g.data
    return g.data


def J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin, space_order=8,
                       free_surface=False, is_residual=False, return_obj=False,
                       isic=False, ws=None, t_sub=1):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface
    isic : Bool
        Whether or not to use ISIC imaging condition
    ws : Array
        Extended source spatial distribution
    is_residual: Bool
        Whether to treat the input as the residual or as the observed data

    Returns
    ----------
    Array
        Adjoint jacobian on the input data (gradient)
    """
    rec, u = forward(model, src_coords, rec_coords, wavelet, save=True, ws=ws,
                     space_order=space_order, free_surface=free_surface, t_sub=t_sub)
    # Residual and gradient
    if not is_residual:
        recin[:] = rec.data[:] - recin[:]   # input is observed data

    g = gradient(model, recin, rec_coords, u, space_order=space_order,
                 free_surface=free_surface, isic=isic)
    if return_obj:
        return .5*model.critical_dt*np.linalg.norm(recin)**2, g.data
    return g.data


def J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin, space_order=8,
                            free_surface=False, is_residual=False, n_checkpoints=None,
                            maxmem=None, return_obj=False, isic=False, ws=None,
                            t_sub=1):
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
    free_surface: Bool (optional)
        Whether or not to use a free surface
    checkpointing: Bool
        Whether or not to use checkpointing
    n_checkpoints: Int
        Number of checkpoints for checkpointing
    maxmem: Float
        Maximum memory to use for checkpointing
    isic : Bool
        Whether or not to use ISIC imaging condition
    ws : Array
        Extended source spatial distribution
    is_residual: Bool
        Whether to treat the input as the residual or as the observed data

    Returns
    ----------
     Array
        Adjoint jacobian on the input data (gradient)
    """
    # Optimal checkpointing
    op_f, u, rec_g = forward(model, src_coords, rec_coords, wavelet,
                             space_order=space_order, return_op=True,
                             free_surface=free_surface, ws=ws)
    op, g, v = gradient(model, recin, rec_coords, u, space_order=space_order,
                        return_op=True, free_surface=free_surface, isic=isic)

    nt = wavelet.shape[0]
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    cp = DevitoCheckpoint([u])
    if maxmem is not None:
        memsize = (cp.size * u.data.itemsize)
        n_checkpoints = int(np.floor(maxmem * 10**6 / memsize))
    wrap_fw = CheckpointOperator(op_f, u=u, vp=model.vp, rcv=rec_g)
    wrap_rev = CheckpointOperator(op, u=u, v=v, vp=model.vp, src=rec)

    # Run forward
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)
    wrp.apply_forward()

    # Residual and gradient
    if is_residual is True:  # input data is already the residual
        rec.data[:] = recin[:]
    else:
        rec.data[:] = rec.data[:] - recin[:]   # input is observed data

    wrp.apply_reverse()

    if return_obj:
        return .5*model.critical_dt*norm(rec)**2, g.data
    return g.data
