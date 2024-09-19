import numpy as np

from devito import Inc, Eq, ConditionalDimension, cos, sin
from devito.tools import as_tuple

from fields import (wavefield_subsampled, lr_src_fields, fourier_modes, norm_holder,
                    illumination)


def save_subsampled(model, u, nt, t_sub, space_order=8):
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
    wf_s = wavefield_subsampled(model, u, nt, t_sub, space_order)
    if wf_s is None:
        return []
    eq_save = []
    for (wfs, wf) in zip(wf_s, as_tuple(u)):
        eq_save.append(Eq(wfs, wf, subdomain=model.physical))
    return eq_save


def wf_as_src(v, w=1, freq_list=None):
    """
    Weighted source as a time-space wavefield

    Parameters
    ----------
    model: Model
        Physical model structure
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
    ws, wt = lr_src_fields(model, weight, wavelet)
    if model.is_tti:
        q = (0, 0) if q == 0 else q
        return (q[0] + ws * wt, q[1] + ws * wt)
    return q + ws * wt


def extended_rec(model, wavelet, v):
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
        return []
    ws, wt = lr_src_fields(model, None, wavelet, empty_w=True, rec=True)
    wf = v[0] + v[1] if model.is_tti else v
    return [Inc(ws, model.grid.time_dim.spacing * wf * wt)]


def freesurface(model, fields, r_coeff=-1):
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
    eqs = []

    z = model.grid.dimensions[-1]
    zfs = model.fs_dim

    for u in as_tuple(fields):
        if u == 0:
            continue

        sh = 1 if z in as_tuple(u.staggered) else 0
        eqs.extend([Eq(u._subs(z, - zfs), r_coeff * u._subs(z, zfs - sh))])
        if z not in as_tuple(u.staggered):
            eqs.append(Eq(u._subs(z, 0), 0))

    return eqs


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
        return []

    # init
    dft_modes, f = fourier_modes(u, freq)

    # Subsampled dft time axis
    time = as_tuple(u)[0].grid.time_dim
    tsave, factor = sub_time(time, factor, dt=dt, freq=freq)

    # Pulsation
    omega_t = 2*np.pi*f*tsave*factor*dt
    # DFT
    dft = []
    for ((ufr, ufi), wf) in zip(dft_modes, as_tuple(u)):
        dft += [Inc(ufr, factor * cos(omega_t) * wf)]
        dft += [Inc(ufi, -factor * sin(omega_t) * wf)]
    return dft


def idft(v, freq=None):
    """
    Symbolic inverse dft of v
dft_modes
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
        w = 1/time.symbolic_max
        idftloc = sum([w*(ufr.subs({ufr.indices[0]: i})*cos(omega_t(f)) -
                          ufi.subs({ufi.indices[0]: i})*sin(omega_t(f)))
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
    nv, nvt = norm_holder(u)
    n_t = [Eq(nvt, nvt + expr)]
    # Then norm in space
    w = weight or 1
    n_s = [Inc(nv[0], nvt / w**2)]
    # Return norm object and expr
    return (n_t, n_s)


def illumexpr(u, illum):
    """
    Illumination of the wavefield `u` as the 2 norm squared of `u` over time.
    Parameters
    ----------
    u: TimeFunction or Tuple of TimeFunction
        Wavefield to take energy along time of
    illum: bool
        Flag on whether or not illumination is computed
    """
    if not illum:
        return []
    u0 = as_tuple(u)
    I = illumination(u, illum)
    expr = sum([ui**2 for ui in u0])
    return [Inc(I, u0[0].grid.time_dim.spacing * expr)]
