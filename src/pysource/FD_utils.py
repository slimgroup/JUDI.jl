from sympy import sqrt, rot_axis2, rot_axis3

from devito import TensorFunction, VectorFunction, Eq, Differentiable


def grads(func):
    """
    Gradient shifted by half a grid point, only to be used in combination
    with divs.
    """
    comps = [getattr(func, 'd%s' % d.name)(x0=d + d.spacing/2)
             for d in func.dimensions if d.is_Space]
    st = tuple([None]*func.grid.dim)
    return VectorFunction(name='grad_%s' % func.name, space_order=func.space_order,
                          components=comps, grid=func.grid, staggered=st)


def divs(func):
    """
    GrDivergenceadient shifted by half a grid point, only to be used in combination
    with grads.
    """
    return sum([getattr(func[i], 'd%s' % d.name)(x0=d - d.spacing/2)
                for i, d in enumerate(func.space_dimensions)])


def laplacian(v, irho):
    """
    Laplacian with density div( 1/rho grad) (u)
    """
    if irho is None or irho == 1:
        Lap = v.laplace
    else:
        if isinstance(irho, Differentiable):
            so = irho.space_order // 2
            Lap = sum([getattr(irho._subs(d, d + d.spacing/2) *
                               getattr(v, 'd%s' % d.name)(x0=d + d.spacing/2,
                                                          fd_order=so),
                               'd%s' % d.name)(x0=d - d.spacing/2, fd_order=so)
                       for d in irho.dimensions])
        else:
            Lap = irho * v.laplace

    return Lap


def ssa_tti(u, v, model):
    """
    TTI finite difference kernel.

    Parameters
    ----------
    u : TimeFunction
        first TTI field
    v : TimeFunction
        second TTI field
    model: Model
        Model structure
    """

    return tensor_fact(u, v, model)


def R_mat(model):
    """
    Rotation matrix according to tilt and asymut.

    Parameters
    ----------
    model: Model
        Model structure
    """
    # Rotation matrix
    Rt = rot_axis2(model.theta)
    if model.dim == 3:
        Rt *= rot_axis3(model.phi)
    else:
        Rt = Rt[[0, 2], [0, 2]]
    return TensorFunction(name="R", grid=model.grid, components=Rt, symmetric=False)


def P_M(model, u, v):
    """
    Vectorial temporaries for TTI.

    Parameters
    ----------
    model: Model
        Model structure
    so: Int
        Space order for discretization
    """
    # Vector for gradients
    st = tuple([None]*model.dim)
    P_I = VectorFunction(name="P_I%s" % u.name, grid=model.grid,
                         space_order=u.space_order, staggered=st)
    M_I = VectorFunction(name="M_I%s" % v.name, grid=model.grid,
                         space_order=v.space_order, staggered=st)
    return P_I, M_I


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
    b_ii = [[b * sqrt((eps - delt) * delt), 0, 0],
            [0, b * sqrt((eps - delt) * delt), 0],
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


def tensor_fact(u, v, model):
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
    # Tensor temps
    P_I, M_I = P_M(model, u, v)
    eq_PI = Eq(P_I, R.T * (A * R * grads(u) + B * R * grads(v)))
    eq_MI = Eq(M_I, R.T * (B * R * grads(u) + C * R * grads(v)))

    return divs(P_I), divs(M_I), eq_PI, eq_MI
