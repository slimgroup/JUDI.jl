import numpy as np

from devito import TimeFunction, ConditionalDimension, Function, DefaultDimension
from devito.data.allocators import ExternalAllocator
from devito.tools import as_tuple


def wavefield(model, space_order, save=False, nt=None, fw=True, name='', t_sub=1):
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
    save = False if t_sub > 1 else save
    if model.is_tti:
        u = TimeFunction(name="%s1" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None if not save else nt)
        v = TimeFunction(name="%s2" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None if not save else nt)
        return (u, v)
    else:
        return TimeFunction(name=name, grid=model.grid, time_order=2,
                            space_order=space_order, save=None if not save else nt)


def forward_wavefield(model, space_order, nt=10, dft=False, t_sub=1):
    """
    Return the wavefield to be used in the gradient calculations depending on the options.

    Parameters
    ----------

    model : Model
        Physical model
    space_order : int
        Spatial discretization order
    nt : int
        Number of time steps on original time axis
    t_sub : int
        Factor for time-subsampling
    dft: Bool
        Whether to use on the fly dft
    """
    u = wavefield(model, space_order, save=True, nt=nt, t_sub=t_sub)
    if dft:
        return fourier_modes(u, np.ones((10,)))[0]
    elif t_sub > 1:
        return wavefield_subsampled(model, u, nt, t_sub, space_order)
    else:
        return u


def memory_field(p):
    """
    Memory variable for viscosity modeling.

    Parameters
    ----------

    p : TimeFunction
        Forward wavefield
    """
    return TimeFunction(name='r%s' % p.name, grid=p.grid, time_order=2,
                        space_order=p.space_order, save=None)


def wavefield_subsampled(model, u, nt, t_sub, space_order=8):
    """
    Create a subsampled wavefield

    Parameters
    ----------

    model : Model
        Physical model
    u : TimeFunction
        Forward wavefield for modeling
    nt : int
        Number of time steps on original time axis
    t_sub : int
        Factor for time-subsampling
    space_order: int
        Spatial discretization order
    """
    if t_sub > 1:
        time_subsampled = ConditionalDimension(name='t_sub', parent=model.grid.time_dim,
                                               factor=t_sub)
        nsave = (nt-1)//t_sub + 2
    else:
        return None
    wf_s = []
    for wf in as_tuple(u):
        usave = TimeFunction(name='us_%s' % wf.name, grid=model.grid, time_order=2,
                             space_order=space_order, time_dim=time_subsampled,
                             save=nsave)
        wf_s.append(usave)
    return wf_s


def lr_src_fields(model, weight, wavelet, empty_ws=False):
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
    if (weight is None and not empty_ws) or wavelet is None:
        return None, None
    time = model.grid.time_dim
    nt = wavelet.shape[0]
    wavelett = Function(name='wf_src', dimensions=(time,), shape=(nt,))
    wavelett.data[:] = np.array(wavelet)[:, 0]
    if empty_ws:
        source_weight = Function(name='src_weight', grid=model.grid, space_order=0)
    else:
        source_weight = Function(name='src_weight', grid=model.grid, space_order=0,
                                 allocator=ExternalAllocator(weight),
                                 initializer=lambda x: None)
    return source_weight, wavelett


def fourier_modes(u, freq):
    """
    On the fly DFT wavefield (frequency slices) and expression

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield
    freq: Array
        Array of frequencies for on-the-fly DFT
    """
    if freq is None:
        return None, None

    # Frequencies
    nfreq = np.shape(freq)[0]
    freq_dim = DefaultDimension(name='freq_dim', default_value=nfreq)
    f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
    f.data[:] = np.array(freq[:])

    dft_modes = []
    for wf in as_tuple(u):
        ufr = Function(name='ufr%s' % wf.name, dimensions=(freq_dim,) + wf.indices[1:],
                       grid=wf.grid, shape=(nfreq,) + wf.shape[1:])
        ufi = Function(name='ufi%s' % wf.name, dimensions=(freq_dim,) + wf.indices[1:],
                       grid=wf.grid, shape=(nfreq,) + wf.shape[1:])
        dft_modes += [(ufr, ufi)]
    return dft_modes, f
