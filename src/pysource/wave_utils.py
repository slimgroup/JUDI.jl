import numpy as np
from sympy import cos, sin, sign

from devito import (TimeFunction, Function, Inc, DefaultDimension,
                    Eq, ConditionalDimension, Dimension)
from devito.tools import as_tuple
from devito.symbolics import retrieve_functions, INT


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
    wf_s = []
    eq_save = []
    if t_sub > 1:
        time_subsampled = ConditionalDimension(name='t_sub', parent=model.grid.time_dim,
                                               factor=t_sub)
        nsave = (nt-1)//t_sub + 2
    else:
        return None, []
    for wf in as_tuple(u):
        usave = TimeFunction(name='us_%s' % wf.name, grid=model.grid, time_order=2,
                             space_order=space_order, time_dim=time_subsampled,
                             save=nsave)
        wf_s.append(usave)
        eq_save.append(Eq(usave, wf))
    return wf_s, eq_save


def wf_as_src(v, w=1, freq_list=None):
    """
    Weighted source as a time-space wavefield

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield (tuple of fields for TTI or dft)
    w: Float or Expr (optional)
        Weight for the source expression (default=1)
    """
    v = idft(v, freq=freq_list) if freq_list is not None else as_tuple(v)
    if len(v) == 2:
        return (w * v[0], w * v[1])
    return w * v[0]


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
        return q
    time = model.grid.time_dim
    nt = wavelet.shape[0]
    wavelett = Function(name='wf_src', dimensions=(time,), shape=(nt,))
    wavelett.data[:] = wavelet[:, 0]
    source_weight = Function(name='src_weight', grid=model.grid, space_order=0)
    source_weight.data[:] = weight
    if model.is_tti:
        return (q[0] + source_weight * wavelett, q[1] + source_weight * wavelett)
    return q + source_weight * wavelett


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
    w_out = Function(name='src_weight', grid=model.grid, space_order=0)
    time = model.grid.time_dim
    wavelett = Function(name='wf_src', dimensions=(time,), shape=(nt,))
    wavelett.data[:] = wavelet[:, 0]
    wf = v[0] + v[1] if model.is_tti else v
    return w_out, [Eq(w_out, w_out + time.spacing * wf * wavelett)]


def freesurface(model, eq):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation

    Parameters
    ----------
    model: Model
        Physical model
    eq: Eq or List of Eq
        Equation to apply mirror to
    """
    fs_eq = []
    for eq_i in eq:
        for p in eq_i._flatten:
            lhs, rhs = p.evaluate.args
            # Add modulo replacements to to rhs
            zfs = model.grid.subdomains['fsdomain'].dimensions[-1]
            z = zfs.parent

            funcs = retrieve_functions(rhs.evaluate)
            mapper = {}
            for f in funcs:
                zind = f.indices[-1]
                if (zind - z).as_coeff_Mul()[0] < 0:
                    s = sign((zind - z.symbolic_min).subs({z: zfs, z.spacing: 1}))
                    mapper.update({f: s * f.subs({zind: INT(abs(zind))})})
            fs_eq.append(Eq(lhs, sign(lhs.indices[-1]-z.symbolic_min) * rhs.subs(mapper),
                            subdomain=model.grid.subdomains['fsdomain']))

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
    nfreq = np.shape(freq)[0]
    freq_dim = DefaultDimension(name='freq_dim', default_value=nfreq)
    f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
    f.data[:] = np.array(freq[:])

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


def idft(v, freq=None):
    """
    Symbolic inverse dft of v

    Parameters
    ----------
    v: TimeFunction or Tuple
        Wavefield to take inverse DFT of
    freq: Array
        Array of frequencies for on-the-fly DFT
    """
    # Subsampled dft time axis
    idft = []
    for vv in v:
        ufr, ufi = vv
        # Frequencies
        time = ufr.grid.time_dim
        dt = time.spacing
        omega_t = lambda f: 2*np.pi*f*time*dt
        idftloc = sum([ufr.subs({ufr.indices[0]: i})*cos(omega_t(f)) -
                       ufi.subs({ufi.indices[0]: i})*sin(omega_t(f))
                       for i, f in enumerate(freq)])
        idft.append(idftloc)
    return tuple(idft)


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


def weighted_norm(u, weight=None):
    """
    Space-time norm of a wavefield, split into norm in time first then in space to avoid
    breaking loops

    Parameters
    ----------
    u: TimeFunction or Tuple of TimeFunction
        Wavefield to take the norm of
    weight: String
        Spacial weight to apply
    """
    grid = as_tuple(u)[0].grid
    expr = grid.time_dim.spacing * sum(uu**2 for uu in as_tuple(u))
    # Norm in time
    norm_vy2_t = Function(name="nvy2t", grid=grid, space_order=0)
    n_t = [Eq(norm_vy2_t, norm_vy2_t + expr)]
    # Then norm in space
    i = Dimension(name="i",)
    norm_vy2 = Function(name="nvy2", shape=(1,), dimensions=(i,), grid=grid)
    w = weight or 1
    n_s = [Inc(norm_vy2[0], norm_vy2_t / w**2)]
    # Return norm object and expr
    return norm_vy2, (n_t, n_s)
