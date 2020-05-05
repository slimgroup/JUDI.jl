import numpy as np
from sympy import cos, sin

from devito import (TimeFunction, Function, Inc, DefaultDimension,
                    Eq, ConditionalDimension)
from devito.tools import as_tuple


def wavefield(model, space_order, save=False, nt=None, fw=True, name=''):
    """
    Create the wavefield for the wave equation

    Parameters
    ----------

    model : Model
        Physical model
    space_order: int
        Spatial discretization order
    save : Bool
        Whether or not to save the time history
    nt : int (optional)
        Number of time steps if the wavefield is saved
    fw : Bool
        Forward or backward (for naming)
    name: string
        Custom name attached to default (u+name)
    """
    name = "u"+name if fw else "v"+name
    if model.is_tti:
        u = TimeFunction(name="%s1" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None if not save else nt)
        v = TimeFunction(name="%s2" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None)
        return (u, v)
    else:
        return TimeFunction(name=name, grid=model.grid, time_order=2,
                            space_order=space_order, save=None if not save else nt)


def wf_as_src(v, w=1):
    """
    Weighted source as a time-space wavefield

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    w: Float or Expr (optional)
        Weight for the source expression (default=1)
    """
    if type(v) is tuple:
        return (w * v[0], 0)
    return w * v


def extented_src(model, weight, wavelet, q=0):
    """
    Extended source for modelling where the source is the outer product of
    a spatially varying weight and a time-dependent wavelet i.e.:
    u.dt2 - u.laplace = w(x)*q(t)
    This function returns the extended source w(x)*q(t)

    Parameters
    ----------
    model: Model
        Physical model structure
    weight: Array
        Array of weight for the spatial Function
    wavelet: Array
        Time-serie for the time-varying source
    q: Symbol or Expr (optional)
        Previously existing source to be added to (source will be q +  w(x)*q(t))
    """
    if weight is None:
        return 0
    time = model.grid.time_dim
    nt = wavelet.shape[0]
    wavelett = Function(name='wf_src', dimensions=(time,), shape=(nt,))
    wavelett.data[:] = wavelet[:, 0]
    source_weight = Function(name='src_weight', grid=model.grid)
    slices = tuple(slice(model.nbl, -model.nbl, 1) for _ in range(model.grid.dim))
    source_weight.data[slices] = weight
    if model.is_tti:
        return (source_weight*wavelett, source_weight*wavelett)
    return source_weight*wavelett


def extended_src_weights(model, wavelet, v):
    """
    Adjoint of extended source. This function returns the expression to obtain
    the spatially varrying weights from the wavefield and time-dependent wavelet

    Parameters
    ----------
    model: Model
        Physical model structure
    wavelet: Array
        Time-serie for the time-varying source
    v: TimeFunction
        Wavefield to get the weights from
    """
    if wavelet is None:
        return None, []
    nt = wavelet.shape[0]
    # Data is sampled everywhere as a sum over time and weighted by wavelet
    w_out = Function(name='src_weight', grid=model.grid)
    time = model.grid.time_dim
    wavelett = Function(name='wf_src', dimensions=(time,), shape=(nt,))
    wavelett.data[:] = wavelet[:, 0]
    wf = v[0] + v[1] if model.is_tti else v
    return w_out, [Eq(w_out, w_out + wf*wavelett)]


def freesurface(field, npml, forward=True):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation

    Parameters
    ----------
    field: TimeFunction or Tuple
        Field for which to add a free surface
    npml: int
        Number of ABC points
    forward: Bool
        Whether it is forward or backward propagation (in time)
    """
    size = as_tuple(field)[0].space_order // 2
    fs = DefaultDimension(name="fs", default_value=size)
    fs_eq = []
    for f in as_tuple(field):
        f_m = f.forward if forward else f.backward
        lhs = f_m.subs({f.indices[-1]: npml - fs - 1})
        rhs = -f_m.subs({f.indices[-1]: npml + fs + 1})
        fs_eq += [Eq(lhs, rhs), Eq(f_m.subs({f.indices[-1]: npml}), 0)]
    return fs_eq


def otf_dft(u, freq, dt, factor=None):
    """
    On the fly DFT wavefield (frequency slices) and expression

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield
    freq: Array
        Array of frequencies for on-the-fly DFT
    factor: int
        Subsampling factor for DFT
    """
    if freq is None:
        return [], None

    # init
    dft = []
    dft_modes = []

    # Subsampled dft time axis
    time = as_tuple(u)[0].grid.time_dim
    tsave, factor = sub_time(time, factor, dt=dt, freq=freq)

    # Frequencies
    nfreq = freq.shape[0]
    freq_dim = DefaultDimension(name='freq_dim', default_value=nfreq)
    f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
    f.data[:] = freq[:]

    # Pulsation
    omega_t = 2*np.pi*f*tsave*factor*dt
    for wf in as_tuple(u):
        ufr = Function(name='ufr%s' % wf.name, dimensions=(freq_dim,) + wf.indices[1:],
                       grid=wf.grid, shape=(nfreq,) + wf.shape[1:])
        ufi = Function(name='ufi%s' % wf.name, dimensions=(freq_dim,) + wf.indices[1:],
                       grid=wf.grid, shape=(nfreq,) + wf.shape[1:])
        dft += [Inc(ufr, factor * cos(omega_t) * wf)]
        dft += [Inc(ufi, -factor * sin(omega_t) * wf)]
        dft_modes += [(ufr, ufi)]
    return dft, dft_modes


def sub_time(time, factor, dt=1, freq=None):
    """
    Subsampled  time axis

    Parameters
    ----------
    time: Dimension
        time Dimension
    factor: int
        Subsampling factor
    """
    if factor == 1:
        return time, factor
    elif freq is not None:
        factor = factor or int(1 / (dt*4*np.max(freq)))
        return ConditionalDimension(name='tsave', parent=time, factor=factor), factor
    elif factor is not None:
        return ConditionalDimension(name='tsave', parent=time, factor=factor), factor
    else:
        return time, 1
