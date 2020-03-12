import numpy as np

from devito import TimeFunction
from propagators import *


# Forward wrappers
def forward_rec(model, src_coords, wavelet, rec_coords,
                space_order=8, free_surface=False):
    """
    Forward modeling of a point source.
    Outputs the shot record.
    """
    rec, _ = forward(model, src_coords, rec_coords, wavelet, save=False,
                     space_order=space_order, free_surface=free_surface)
    return rec.data


def forward_no_rec(model, src_coords, wav, space_order=8, free_surface=False):
    """
    Forward modeling of a point source without receiver.
    Outputs the full wavefield.
    """
    _, u = forward(model, src_coords, None, wav, space_order=space_order,
                   save=True, free_surface=free_surface)
    return u.data


def forward_wf_src(model, u, rec_coords, space_order=8, free_surface=False):
    """
    Forward modeling of a full wavefield source.
    Outputs the shot record.
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


def forward_wf_src_norec(model, u, space_order=8, free_surface=False):
    """
    Forward modeling of a full wavefield source without receiver.
    Outputs the full wavefield
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


# Adjoint wrappers
def adjoint_rec(model, src_coords, rec_coords, data,
                space_order=8, free_surface=False):
    """
    Adjoint/backward modeling of a shot record (receivers as source).
    Outputs the adjoint wavefield sampled at the source location.
    """
    rec, v = adjoint(model, data, src_coords, rec_coords,
                     space_order=space_order, free_surface=free_surface)
    return rec.data


def adjoint_no_rec(model, rec_coords, data, space_order=8, free_surface=False):
    """
    Adjoint/backward modeling of a shot record (receivers as source).
    Outputs the full adjoint wavefield.
    """
    _, v = adjoint(model, data, None, rec_coords, space_order=space_order,
                   save=True, free_surface=free_surface)
    return v.data


def adjoint_wf_src(model, u, src_coords, space_order=8, free_surface=False):
    """
    Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source).
    Outputs the adjoint wavefield sampled at the source location.
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


def adjoint_wf_src_norec(model, u, src_coords,
                         space_order=8, free_surface=False):
    """
    Adjoint/backward modeling of a full wavefield (full wavefield as adjoint source).
    Outputs the full adjoint wavefield.
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


# Linearized modeling
def born_rec(model, src_coords, wavelet, rec_coords,
                space_order=8, free_surface=False):
    """
    Linearized (Born) modeling of a point source for a model perturbation (square slowness) dm.
    Output the linearized data.
    """
    rec, _ = born(model, src_coords, rec_coords, wavelet, save=False,
                  space_order=space_order, free_surface=free_surface)
    return rec.data


# Gradient wrappers
def J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=8,
              checkpointing=False, free_surface=False, n_checkpoints=None,
              maxmem=None, freq_list=[]):
    """
    Jacobian (adjoint fo born modeling operator) iperator on a shot record as a source (i.e data residual).
    Outputs the gradient.
    Supports three modes:
    * Checkpinting
    * Frequency compression (on-the-fly DFT)
    * Standard zero lag cross correlation over time
    """
    if checkpointing:
        grad = J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords,
                                       recin, space_order=8, free_surface=False,
                                       n_checkpoints=n_checkpoints, is_residual=True,
                                       maxmem=maxmem)
    elif len(freq_list) > 0:
        grad = J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin,
                              space_order=space_order, is_residual=True,
                              free_surface=free_surface, freq_list=freq_list)
    else:
        grad = J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin,
                                  is_residual=True,
                                  space_order=space_order, free_surface=free_surface)

    return grad


def J_adjoint_freq(model, src_coords, wavelet, rec_coords, recin,
                   space_order=8, free_surface=False, freq_list=[],
                   is_residual=False, return_obj=False):
    """
    Gradient (appication of Jacobian to a shot record) computed with on-the-fly
    Fourier transform.
    Outputs gradient, and objective function (least-square) if requested.
    """
    rec, u = forward(model, src_coords, None, wavelet, save=True,
                   space_order=space_order, free_surface=free_surface,
                   freq_list=freq_list)
    # Residual and gradient
    if is_residual is not True:  # input data is already the residual
        recin[:] = rec.data[:] - recin[:]   # input is observed data

    g = gradient(model, recin, rec_coords, u, space_order=space_order,
                 free_surface=free_surface, freq=freq_list)
    if return_obj:
        return .5*np.linalg.norm(recin)**2, g.data
    return g.data


def J_adjoint_standard(model, src_coords, wavelet, rec_coords, recin,
                       space_order=8, free_surface=False,
                       is_residual=False, return_obj=False):
    """
    Gradient (appication of Jacobian to a shot record) computed with the standard sum over time.
    Outputs gradient, and objective function (least-square) if requested.
    """
    rec, u = forward(model, src_coords, None, wavelet, save=True,
                     space_order=space_order, free_surface=free_surface)
    # Residual and gradient
    if is_residual is not True:  # input data is already the residual
        recin[:] = rec.data[:] - recin[:]   # input is observed data

    g = gradient(model, recin, rec_coords, u, space_order=space_order,
                 free_surface=free_surface)
    if return_obj:
        return .5*np.linalg.norm(recin)**2, g.data
    return g.data


def J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin,
                            space_order=8, free_surface=False, is_residual=False,
                            n_checkpoints=None, maxmem=None, return_obj=False):
    """
    Gradient (appication of Jacobian to a shot record) computed with (optimal?) checkpointing.
    Outputs gradient, and objective function (least-square) if requested.
    """
    # Optimal checkpointing
    op_f, u, rec = forward(model, src_coords, rec_coords, wavelet,
                           space_order=space_order, return_op=True,
                           free_surface=free_surface)
    op, g = gradient(model, rec_coords, recin, space_order=space_order,
                     return_op=True, free_surface=free_surface)
    cp = DevitoCheckpoint([u])
    if maxmem is not None:
        memsize = (cp.size * u.data.itemsize)
        n_checkpoints = int(np.floor(maxmem * 10**6 / memsize))
    wrap_fw = CheckpointOperator(op_f, u=u, m=model.m, rec=rec)
    wrap_rev = CheckpointOperator(op, u=u, v=v, m=model.m, src=rec)

    # Run forward
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)
    wrp.apply_forward(**op_kwargs(model, fs=free_surface))

    # Residual and gradient
    if is_residual is True:  # input data is already the residual
        rec.data[:] = recin[:]
    else:
        rec.data[:] = rec.data[:] - recin[:]   # input is observed data

    wrp.apply_reverse(**op_kwargs(model, fs=free_surface))

    if return_obj:
        return .5*norm(rec)**2, g.data
    return g.data
