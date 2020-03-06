################################################################################
#
# FWI functional/gradient computation routines (python implementation using devito)
#
################################################################################

# Module loading
import numpy.linalg as npla

from utils import applyfilt
from propagators import forward, gradient


# Objective functional
def obj_fwi(model, src_coords, rcv_coords, src, data, Filter=None,
            mode="eval", space_order=8):
    """
    Evaluate FWI objective functional/gradients for current m
    """
    dt = model.critical_dt
    # Normalization constant
    data_filtered = applyfilt(data, Filter)
    eta = dt * npla.norm(data_filtered.reshape(-1))**2

    # Computing residual
    dmod, u = forward(model, src_coords, rcv_coords, src, space_order=space_order,
                      save=(mode == "grad"))
    Pres = applyfilt(data - dmod, Filter)

    # ||P*r||^2
    norm_Pr2 = dt * npla.norm(Pres.reshape(-1))**2

    # Functional evaluation
    fun = norm_Pr2 / eta

    # Gradient computation
    if mode == "grad":
        gradm = gradient(model, Pres, rcv_coords, u, space_order=space_order, w=2*dt/eta)

    # Return output
    if mode == "eval":
        return fun
    elif mode == "grad":
        return fun, gradm
