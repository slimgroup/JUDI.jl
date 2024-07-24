from sympy import rot_axis2, rot_axis3
from devito import TensorFunction, Differentiable, div, grad, cos, sin


trig_mapper = {cos.__sympy_class__: cos, sin.__sympy_class__: sin}

r2 = lambda x: rot_axis2(x).applyfunc(lambda i: trig_mapper.get(i.func, i.func)(*i.args))
r3 = lambda x: rot_axis3(x).applyfunc(lambda i: trig_mapper.get(i.func, i.func)(*i.args))


def laplacian(v, irho):
    """
    Laplacian with density div( 1/rho grad) (u)
    """
    irho = irho or 1
    if isinstance(irho, Differentiable):
        Lap = div(irho * grad(v, shift=.5), shift=-.5)
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
        Rt = r2(model.theta)
    except AttributeError:
        Rt = r2(0)
    if model.dim == 3:
        try:
            Rt *= r3(model.phi)
        except AttributeError:
            Rt *= r3(0)
    else:
        Rt = Rt[[0, 2], [0, 2]]
    # Rebuild sin/cos

    return TensorFunction(name="R", grid=model.grid, components=Rt, symmetric=False)


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
    # Matrix of Thomsen params
    A, B, C = thomsen_mat(model)
    # Rotation Matrix
    R = R_mat(model)

    PI = R.T * (A * R * grad(u, shift=.5) + B * R * grad(v, shift=.5))
    MI = R.T * (B * R * grad(u, shift=.5) + C * R * grad(v, shift=.5))

    return div(PI, shift=-.5), div(MI, shift=-.5)
