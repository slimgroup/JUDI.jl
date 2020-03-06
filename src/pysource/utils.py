import numpy as np
from numpy.fft import fft, ifft

from sympy import sqrt
# Utility functions such as weighting and filtering


# Weighting
def weight_fun(weight_fun_pars, model, src_coords):
    if weight_fun_pars is None:
        return None
    if weight_fun_pars[0] == "srcfocus":
        return weight_srcfocus(model, src_coords, delta=np.float32(weight_fun_pars[1]))
    elif weight_fun_pars[0] == "depth":
        return weight_depth(model, src_coords, delta=np.float32(weight_fun_pars[1]))


def weight_srcfocus(model, src_coords, delta=np.float32(0.01)):
    """
    w(x) = sqrt((||x-xsrc||^2+delta^2)/delta^2)
    """

    ix, iz = model.grid.dimensions
    isrc = (np.float32(model.nbpml) + src_coords[0, 0] / model.spacing[0],
            np.float32(model.nbpml) + src_coords[0, 1] / model.spacing[1])
    h = np.sqrt(model.spacing[0]*model.spacing[1])
    return sqrt((ix-isrc[0])**2+(iz-isrc[1])**2+(delta/h)**np.float32(2))/(delta/h)


def weight_depth(model, src_coords, delta=np.float32(0.01)):
    """
    w(x) = sqrt((||z-zsrc||^2+delta^2)/delta^2)
    """

    _, iz = model.grid.dimensions
    isrc = (np.float32(model.nbpml)+src_coords[0, 0]/model.spacing[0],
            np.float32(model.nbpml)+src_coords[0, 1]/model.spacing[1])
    h = np.sqrt(model.spacing[0]*model.spacing[1])
    return sqrt((iz-isrc[1])**2+(delta/h)**np.float32(2))/(delta/h)


# Data filtering
def applyfilt(dat, Filter=None):
    if Filter is None:
        return dat
    else:
        pad = max(dat.shape[0], Filter.size)
        filtered = ifft(fft(dat, n=pad, axis=0)*Filter.reshape(-1, 1), axis=0)
        return np.real(filtered[:dat.shape[0], :])


def applyfilt_transp(dat, Filter=None):

    if Filter is None:
        return dat
    else:
        pad = max(dat.shape[0], Filter.size)
        filtered = ifft(fft(dat, n=pad, axis=0)*np.conj(Filter).reshape(-1, 1), axis=0)
        return np.real(filtered[:dat.shape[0], :])


# Alpha for wri

def compute_optalpha(v1, v2, v3, comp_alpha=True):

    if comp_alpha:
        if v3 < np.abs(v2):
            a = np.sign(v2)*(np.abs(v2)-v3)/(np.float32(2)*v1)
            if np.isinf(a) or np.isnan(a):
                return np.float32(0)
            else:
                return a
        else:
            return np.float32(0)
    else:
        return np.float32(1)
