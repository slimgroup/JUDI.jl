from sympy import rot_axis2, rot_axis3

from devito import TensorFunction, Differentiable, div, grad


def laplacian(v, irho):
    """
    Laplacian with density div( 1/rho grad) (u)
    """
    irho = irho or 1
    if isinstance(irho, Differentiable):
        so = irho.space_order // 2
        Lap = div(irho * grad(v, shift=.5, order=so), shift=-.5, order=so)
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
    # Space order
    so = u.space_order // 2
    # Matrix of Thomsen params
    A, B, C = thomsen_mat(model)
    # Rotation Matrix
    R = R_mat(model)

    PI = R.T * (A * R * grad(u, shift=.5, order=so) +
                B * R * grad(v, shift=.5, order=so))
    MI = R.T * (B * R * grad(u, shift=.5, order=so) +
                C * R * grad(v, shift=.5, order=so))

    return div(PI, shift=-.5, order=so), div(MI, shift=-.5, order=so)
