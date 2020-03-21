from sympy import sqrt, cos, sin
from devito import grad, first_derivative, centered, transpose, Function, div, left, right


def laplacian(v, irho):
    """
    Laplacian with density div( 1/rho grad) (u)
    """
    if irho is None or irho == 1:
        Lap = v.laplace
    else:
        if isinstance(irho, Function):
            Lap = grad(irho).T * grad(v) + irho * v.laplace
        else:
            Lap = irho * v.laplace

    return Lap


def ssa_tti(u, v, model):
    """
    TTI finite difference kernel. The equation we solve is:
    see ref
    :param u: first TTI field
    :param v: second TTI field
    :param  model: Model structure
    :return: RHS of PDE for each component
    """

    return ssa_1(u, v, model), ssa_2(u, v, model)

def ssa_1(u, v, model):
    """
    First row of
    gx_t(A * gx(P)) + gy_t( A1 * gy(P)) + gz_T( A2 * gz(P))
    """
    delta, epsilon, irho = model.delta, model.epsilon, model.irho
    a11 = irho * delta
    a12 = irho * sqrt( (epsilon - delta) * delta)
    b11 = irho
    b12 = 0

    g1 = gx_T(a11 * gx(u, model) + a12 * gx(v, model), model)
    if model.dim == 3:
        g1 += gy_T(a11 * gy(u, model) + a12 * gy(v, model), model)
    g1 +=  gz_T(b11 * gz(u, model) + b12 * gz(v, model), model)
    return g1


def ssa_2(u, v, model):
    """
    Second row of
    gx_t(A * gx(P)) + gy_t( A1 * gy(P)) + gz_T( A2 * gz(P))
    """
    delta, epsilon, irho = model.delta, model.epsilon, model.irho
    a21 = irho * sqrt( (epsilon - delta) * delta)
    a22 = irho * (epsilon - delta)
    b21 = 0
    b22 = 0

    g2 = gx_T(a21 * gx(u, model) + a22 * gx(v, model), model)
    if model.dim == 3:
        g2 += gy_T(a21 * gy(u, model) + a22 * gy(v, model), model)
    g2 += gz_T(b21 * gz(u, model) + b22 * gz(v, model), model)
    return g2


def angles_to_trig(model):

    return cos(model.theta), sin(model.theta), cos(model.phi), sin(model.phi)


def gx(field, model):
    """
    Rotated first derivative in x
    :param u: TTI field
    :param  model: Model structure
    :return: du/dx in rotated coordinates
    """
    costheta, sintheta, cosphi, sinphi =  angles_to_trig(model)
    dims = field.dimensions[1:model.dim+1]
    order1 = field.space_order // 2

    Dx = (costheta * cosphi * first_derivative(field, dim=dims[0], side=left, fd_order=order1) -
          sintheta * first_derivative(field, dim=dims[-1], side=left, fd_order=order1))

    if len(dims) == 3:
        Dx += costheta * sinphi * first_derivative(field, dim=dims[1], side=left, fd_order=order1)
    return Dx


def gy(field, model):
    """
    Rotated first derivative in y
    :param u: TTI field
    :param  model: Model structure
    :return: du/dy in rotated coordinates
    """
    costheta, sintheta, cosphi, sinphi =  angles_to_trig(model)
    dims = field.dimensions[1:model.dim+1]
    order1 = field.space_order // 2

    Dy = (-sinphi * first_derivative(field, dim=dims[0], side=centered, fd_order=order1) +
          cosphi * first_derivative(field, dim=dims[1],side=centered, fd_order=order1))

    return Dy


def gz(field, model):
    """
    Rotated first derivative in z
    :param u: TI field
    :param  model: Model structure
    :return: du/dz in rotated coordinates
    """
    costheta, sintheta, cosphi, sinphi =  angles_to_trig(model)
    dims = field.dimensions[1:model.dim+1]
    order1 = field.space_order // 2

    Dz = (sintheta * cosphi * first_derivative(field, dim=dims[0], side=right, fd_order=order1) +
          costheta * first_derivative(field, dim=dims[-1], side=right, fd_order=order1))

    if len(dims) == 3:
        Dz += sintheta * sinphi * first_derivative(field, dim=dims[1], side=right, fd_order=order1)
    return Dz


def gx_T(field, model):
    """
    Rotated first derivative in x
    :param u: TTI field
    :param  model: Model structure
    :return: du/dx in rotated coordinates
    """
    if field == 0:
        return 0

    costheta, sintheta, cosphi, sinphi =  angles_to_trig(model)
    dims = field.dimensions[1:model.dim+1]
    order1 = field.space_order // 2

    Dx = -(first_derivative(costheta * cosphi * field, dim=dims[0],
                           side=left, fd_order=order1, matvec=transpose) -
          first_derivative(sintheta * field, dim=dims[-1],
                           side=left, fd_order=order1, matvec=transpose))

    if len(dims) == 3:
        Dx += first_derivative(costheta * sinphi * field, dim=dims[1],
                               side=left, fd_order=order1, matvec=transpose)
    return Dx


def gy_T(field, model):
    """
    Rotated first derivative in y
    :param u: TTI field
    :param  model: Model structure
    :return: du/dy in rotated coordinates
    """
    if field == 0:
        return 0

    costheta, sintheta, cosphi, sinphi =  angles_to_trig(model)
    dims = field.dimensions[1:model.dim+1]
    order1 = field.space_order // 2

    Dy = (first_derivative(-sinphi * field, dim=dims[0], matvec=transpose,
                           side=centered, fd_order=order1) +
          first_derivative(cosphi * field, dim=dims[1], matvec=transpose,
                           side=centered, fd_order=order1))

    return Dy


def gz_T(field, model):
    """
    Rotated first derivative in z
    :param u: TI field
    :param  model: Model structure
    :return: du/dz in rotated coordinates
    """
    if field == 0:
        return 0

    costheta, sintheta, cosphi, sinphi =  angles_to_trig(model)
    dims = field.dimensions[1:model.dim+1]
    order1 = field.space_order // 2

    Dz = -(first_derivative(sintheta * cosphi * field, dim=dims[0],
                            side=right, fd_order=order1, matvec=transpose) +
           first_derivative(costheta * field, dim=dims[-1],
                            side=right, fd_order=order1, matvec=transpose))

    if len(dims) == 3:
        Dz += first_derivative(sintheta * sinphi * field, dim=dims[1],
                               side=right, fd_order=order1, matvec=transpose)
    return Dz
