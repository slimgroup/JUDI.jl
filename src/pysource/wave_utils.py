import numpy as np
from sympy import cos, sin

from devito import TimeFunction, Function, Inc, Dimension, DefaultDimension, Eq, ConditionalDimension
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
        ufr = Function(name='ufr%s'%wf.name, dimensions=(freq_dim,) + wf.indices[1:],
                       grid=wf.grid, shape=(nfreq,) + wf.shape[1:])
        ufi = Function(name='ufi%s'%wf.name, dimensions=(freq_dim,) + wf.indices[1:],
                       grid=wf.grid,  shape=(nfreq,) + wf.shape[1:])
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