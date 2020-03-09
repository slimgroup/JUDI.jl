from devito import TimeFunction, Function, Inc, Dimension, DefaultDimension, Eq
from devito.tools import as_tuple


def wavefield(model, space_order, save=False, nt=None, fw=True, name=''):
    name = "u"+name if fw else "v"+name
    if model.is_tti:
        u = TimeFunction(name="%s1" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None if not save else nt)
        v = TimeFunction(name="%s2" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=None if not save else nt)
        return (u, v)
    else:
        return TimeFunction(name=name, grid=model.grid, time_order=2,
                            space_order=space_order, save=None if not save else nt)


def weighted_norm(u, weight=None):
    """
    Space-time nor of a wavefield, split into norm in time first then in space to avoid
    breaking loops
    """
    if type(u) is tuple:
        expr = u[0]**2 + u[1]**2
        grid = u[0].grid
    else:
        expr = u**2
        grid = u.grid
    # Norm in time
    norm_vy2_t = Function(name="nvy2t", grid=grid)
    n_v = [Inc(norm_vy2_t, expr)]
    # Then norm in space
    i = Dimension(name="i", )
    norm_vy2 = Function(name="nvy2", shape=(1,), dimensions=(i, ), grid=grid)
    if weight is None:
        n_v += [Inc(norm_vy2[0], norm_vy2_t)]
    else:
        n_v += [Inc(norm_vy2[0], norm_vy2_t / weight**2)]
    return norm_vy2, n_v


def corr_fields(u, v, freq=False):
    if freq:
        ufr, ufi = u
        tsave = u.grid.time_dim
        expr = (ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v
    else:
        expr = - v * u.dt2
    return expr


def grad_expr(gradm, v, u, w=1, freq=False):
    expr = 0
    for wfu, wfv in zip(as_tuple(u), as_tuple(v)):
        expr += w * corr_fields(wfu, wfv)
    return [Inc(gradm, expr)]


def wf_as_src(v, w=1):
    if type(v) is tuple:
        return (w * v[0], w * v[1])
    return w * v

def lin_src(model, v):
    w = - model.dm * model.irho
    if type(v) is tuple:
        return (w * v[0].dt2, w * v[1].dt2)
    return w * v.dt2


def freesurface(field, npml, forward=True):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation
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


def otf_dft(u, freq, factor=None):
    if freq is None:
        return [], None
    # init
    dft = []
    dft_modes = []
    freq_dim = Dimension(name='freq_dim')
    nfreq = freq.shape[0]
    f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
    f.data[:] = freq[:]
    for wf in as_tuple(u):
        tsave = wf.grid.time_dim
        ufr = Function(name='ufr%s'%wf.name, dimensions=(freq_dim,) + wf.indices[1:], shape=(nfreq,) + model.shape_domain)
        ufi = Function(name='ufi%s'%wf.name, dimensions=(freq_dim,) + wf.indices[1:], shape=(nfreq,) + model.shape_domain)
        dft += [Inc(ufr, wf*cos(2*np.pi*f*tsave*factor*dt))]
        dft += [Inc(ufi, -dwf*sin(2*np.pi*f*tsave*factor*dt))]
        dft_modes += [(ufr, ufi)]
    return dft, dft_modes
