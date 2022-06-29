import numpy as np
from sympy import sqrt
from devito import configuration

try:
    from devito.core.cpu import Cpu64OperatorMixin as cpo
except ImportError:
    from devito.core.cpu import CPU64NoopOperator as cpo

from devito.exceptions import InvalidOperator
from devito.tools import as_tuple


# Weighting
def weight_fun(w_fun, model, src_coords):
    """
    Symbolic weighting function

    Parameters
    ----------
    w_fun: Tuple(String, Float)
        Weighting fucntion and weight.
    model: Model
        Model structure
    src_coords: Array
        Source coordinates.
    """
    if w_fun is None:
        return 1
    else:
        return weight_srcfocus(model, src_coords, delta=w_fun[1],
                               full=(w_fun[0] == "srcfocus"))


def weight_srcfocus(model, src_coords, delta=.01, full=True):
    """
    Source focusing weighting function
    w(x) = sqrt((||x-xsrc||^2+delta^2)/delta^2)

    Parameters
    ----------
    model: Model
        Model structure
    src_coords: Array
        Source coordinates
    delta: Float
        Reference distance for weights
    """
    w_dim = as_tuple(model.grid.dimensions if full else model.grid.dimensions[-1])
    isrc = tuple(np.float32(model.padsizes[i][0]) + src_coords[0, i] / model.spacing[i]
                 for i in range(model.dim))
    h = np.prod(model.spacing)**(1/model.dim)
    radius = sum((d - isrc[i])**2 for i, d in enumerate(w_dim))
    return sqrt(radius + (delta / h)**2) / (delta/h)


def compute_optalpha(norm_r, norm_Fty, epsilon, comp_alpha=True):
    """
    Compute optimal alpha for WRI

    Parameters
    ----------
    norm_r: Float
        Norm of residual
    norm_Fty: Float
        Norm of adjoint wavefield squared
    epsilon: Float
        Noise level
    comp_alpha: Bool
        Whether to compute the optimal alpha or just return 1
    """
    if comp_alpha:
        if norm_r > epsilon and norm_Fty > 0:
            return norm_r * (norm_r - epsilon) / norm_Fty
        else:
            return 0
    else:
        return 1


def opt_op(model):
    """
    Setup the compiler options for the operator. Dependeing on the devito
    version more or less options can be used for performance, mostly impacting TTI.

    Parameters
    ----------
    model: Model
        Model structure to know if we are in a TTI model
    """
    if configuration['platform'].name in ['nvidiaX', 'amdgpuX']:
        opts = {'openmp': True if configuration['language'] == 'openmp' else None,
                'mpi': configuration['mpi']}
    else:
        opts = {'openmp': True, 'par-collapse-ncores': 2, 'mpi': configuration['mpi']}
    # Minimal size temporaries
    if not model.fs and not (configuration['platform'].name in ['nvidiaX', 'amdgpuX']):
        try:
            opts['min-storage'] = True
            'min-storage' in cpo._normalize_kwargs(options=dict(opts))['options']
        except InvalidOperator:
            opts.pop('min-storage')
    # Cire rotate for tti
    if model.is_tti and model.dim > 2:
        try:
            opts['cire-rotate'] = True
            'cire-rotate' in cpo._normalize_kwargs(options=dict(opts))['options']
        except InvalidOperator:
            opts.pop('cire-rotate')
        try:
            opts['cire-repeats-sops'] = 9
            'cire-repeats-sops' in cpo._normalize_kwargs(options=dict(opts))['options']
        except InvalidOperator:
            opts.pop('cire-repeats-sops')
    return ('advanced', opts)


def fields_kwargs(*args):
    """
    Creates a dictionary of {f.name: f} for any field argument that is not None
    """
    kw = {}
    for field in args:
        if field is not None:
            # In some case could be a tuple of fields, such as dft modes
            try:
                kw.update({f.name: f for f in as_tuple(field)})
            except AttributeError:
                for f in field:
                    kw.update({ff.name: ff for ff in f})

    return kw
