################################################################################
#
# FWI functional/gradient computation routines (python implementation using devito)
#
################################################################################

# Module loading
import numpy.linalg as npla

from utils import applyfilt
from propagators import forward, gradient

def fwi_gradient_checkpointing():
    # Optimal checkpointing
    op_f, u, rec = forward()
    op, g = gradient(op_return=True)
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
