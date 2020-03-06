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


def grad_expr(gradm, vPTy, u, w=1):
    if type(vPTy) is tuple:
        return [Inc(gradm, -w * (vPTy[0] * u[0].dt2 + vPTy[1] * u[1].dt2))]
    return [Inc(gradm, -w * vPTy * u.dt2)]


def wf_as_src(v, w=1):
    if type(v) is tuple:
        return (w * v[0], w * v[1])
    return w * v

def lin_src(model, v):
    w = -model.dm * model.grid.time_dim.spacing * model.irho
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
