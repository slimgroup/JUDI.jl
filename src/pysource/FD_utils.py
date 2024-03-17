from sympy import rot_axis2, rot_axis3

from devito import TensorFunction, VectorFunction, Differentiable


def grads(func, so_fact=1, side=1):
    """
    Gradient shifted by half a grid point, only to be used in combination
    with divs.
    """
    so = func.space_order // so_fact
    comps = [getattr(func, 'd%s' % d.name)(x0=d + side * d.spacing/2, fd_order=so)
             for d in func.dimensions if d.is_Space]
    st = tuple([None]*func.grid.dim)
    return VectorFunction(name='grad_%s' % func.name, space_order=func.space_order,
                          components=comps, grid=func.grid, staggered=st)


def divs(func, so_fact=1, side=-1):
    """
    GrDivergenceadient shifted by half a grid point, only to be used in combination
    with grads.
    """
    res = 0
    zfd = lambda *ar, **kw: 0
    for i, d in enumerate(func.space_dimensions):
        so = getattr(func[i], 'space_order', 0) // so_fact
        res += getattr(func[i], 'd%s' % d.name, zfd)(x0=d+side*d.spacing/2, fd_order=so)

    return res


def laplacian(v, irho):
    """
    Laplacian with density div( 1/rho grad) (u)
    """
    if irho is None or irho == 1:
        Lap = v.laplace
    else:
        if isinstance(irho, Differentiable):
            so = irho.space_order // 2
            Lap = sum([getattr(irho *
                               getattr(v, 'd%s' % d.name)(x0=d + d.spacing/2,
                                                          fd_order=so),
                               'd%s' % d.name)(x0=d - d.spacing/2, fd_order=so)
                       for d in irho.dimensions])
        else:
            Lap = irho * v.laplace

    return Lap


def R_mat(model):
    """
    Rotation matrix according to tilt and asymut.

    Parameters
    ----------
    model: Model
        Model structure
    """
    # Rotation matrix
    try:
        Rt = rot_axis2(model.theta)
    except AttributeError:
        Rt = rot_axis2(0)
    if model.dim == 3:
        try:
            Rt *= rot_axis3(model.phi)
        except AttributeError:
            Rt *= rot_axis3(0)
    else:
        Rt = Rt[[0, 2], [0, 2]]
    R = TensorFunction(name="R", grid=model.grid, components=Rt, symmetric=False)
    try:
        R.name == "R"
        return R
    except AttributeError:
        return Rt


def thomsen_mat(model):
    """
    Diagonal Matrices with Thomsen parameters for vectorial temporaries
    computation.

    Parameters
    ----------
    model: Model
        Model structure
    """
    # Diagonal matrices
    b = model.irho
    eps, delt = model.epsilon, model.delta
    a_ii = [[b * delt, 0, 0],
            [0, b * delt, 0],
            [0, 0, b]]
    b_ii = [[b * ((eps - delt) * delt)**.5, 0, 0],
            [0, b * ((eps - delt) * delt)**.5, 0],
            [0, 0, 0]]
    c_ii = [[b * (eps - delt), 0, 0],
            [0, b * (eps - delt), 0],
            [0, 0, 0]]

    if model.dim == 2:
        s = slice(0, 3, 2)
        a_ii = [a_ii[i][s] for i in range(0, 3, 2)]
        b_ii = [b_ii[i][s] for i in range(0, 3, 2)]
        c_ii = [c_ii[i][s] for i in range(0, 3, 2)]
    A = TensorFunction(name="A", grid=model.grid, components=a_ii, diagonal=True)
    B = TensorFunction(name="B", grid=model.grid, components=b_ii, diagonal=True)
    C = TensorFunction(name="C", grid=model.grid, components=c_ii, diagonal=True)
    return A, B, C


def sa_tti(u, v, model):
    """
    Tensor factorized SSA TTI wave equation spatial derivatives.

    Parameters
    ----------
    u : TimeFunction
        first TTI field
    v : TimeFunction
        second TTI field
    model: Model
        Model structure
    """
    # MAtrix of Thomsen params
    A, B, C = thomsen_mat(model)
    # Rotation Matrix
    R = R_mat(model)

    PI = R.T * (A * R * grads(u, so_fact=2) + B * R * grads(v, so_fact=2))
    MI = R.T * (B * R * grads(u, so_fact=2) + C * R * grads(v, so_fact=2))

    return divs(PI, so_fact=2), divs(MI, so_fact=2)
