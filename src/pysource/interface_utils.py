from devito import TimeFunction
from propagators import *

### Forward wrappers

def forward_no_rec(model, src_coords, wavelet, space_order=8, free_surface=False):
    _, u = forward(model, src_coords, None, wavelet, space_order=space_order,
                   save=True, free_surface=free_surface)
    return u.data

def forward_wf_src(model, u, rec_coords, space_order=8, free_surface=False):
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
    wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2,
                          space_order=space_order, save=u.shape[0])
    if isinstance(u, TimeFunction):
        wf_src._data = u._data
    else:
        wf_src.data[:] = u[:]
    _, u = forward(model, None, None, None, space_order=space_order,
                   save=True, free_surface=free_surface, q=wf_src)
    return u.data

### Adjoint wrappers

def adjoint_no_rec(model, rec_coords, data, space_order=8, free_surface=False):
    _, v = adjoint(model, data, None, rcv_coords, space_order=space_order, save=True,
                   free_surface=False)
    return v.data


def adjoint_wf_src(model, u, src_coords, space_order=8, free_surface=False):
    wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2,
                          space_order=space_order, save=u.shape[0])
    if isinstance(u, TimeFunction):
        wf_src._data = u._data
    else:
        wf_src.data[:] = u[:]
    rec, _ = adjoint(model, None, src_coords, None, space_order=space_order,
                     free_surface=free_surface, q=wf_src)
    return rec.data

def adjoint_wf_src_norec(model, u, src_coords, space_order=8, free_surface=False):
    wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2,
                          space_order=space_order, save=u.shape[0])
    if isinstance(u, TimeFunction):
        wf_src._data = u._data
    else:
        wf_src.data[:] = u[:]
    rec, _ = adjoint(model, None, None, None, space_order=space_order,
                     save=True, free_surface=free_surface, q=wf_src)
    return rec.data


#### Gradient wrappers


def J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=8, checkpointing=False, free_surface=False,
              n_checkpoints=None, maxmem=None, freq_list=None):
    if checkpointing:
        grad = J_adjoint_checkpointing(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False,
                                       n_checkpoints=n_checkpoints, maxmem=maxmem)
    elif freq_list is not None:
        _, u = forward(model, src_coords, wavelet, None, save=True, space_order=space_order, free_surface=free_surface,
                      freq_list=freq_list)
        grad = gradient(model, recin, rec_coords, space_order=space_order, free_surface=freq_list)
    else:
        _, u = forwad(model, src_coords, wavelet, None, save=True, space_order=space_order, free_surface=free_surface)
        grad = gradient(model, recin, rec_coords, space_order=space_order, free_surface=free_surface)

    return grad



def fwi_gradient_checkpointing(model, src_coords, wavelet, rec_coords, recin, space_order=8, free_surface=False,
                               n_checkpoints=n_checkpoints, maxmem=maxmem):
    # Optimal checkpointing
    op_f, u, rec = forward(model, src_coords, wavelet, rec_coords, space_order=space_order, return_op=True, free_surface=free_surface)
    op, g = gradient(model, rec_coords, recin, space_order=space_order, return_op=True, free_surface=free_surface)
    cp = DevitoCheckpoint([u])
    if maxmem is not None:
        n_checkpoints = int(np.floor(maxmem * 10**6 / (cp.size * u.data.itemsize)))
    wrap_fw = CheckpointOperator(op_f, u=u, m=model.m, rec=rec)
    wrap_rev = CheckpointOperator(op, u=u, v=v, m=model.m, src=src)

    # Run forward
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)
    wrp.apply_forward(**op_kwargs(model, fs=free_surface))

    # Residual and gradient
    if is_residual is True:  # input data is already the residual
        rec_g.data[:] = rec_data[:]
    else:
        rec_g.data[:] = rec.data[:] - rec_data[:]   # input is observed data
        fval = .5*np.dot(rec_g.data[:].flatten(), rec_g.data[:].flatten()) * dt
    wrp.apply_reverse(**op_kwargs(model, fs=free_surface))

    return g.data
