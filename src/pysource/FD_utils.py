from devito import grad, first_derivative, centered, transpose, Function, div, left, right
from devito.symbolics import retrieve_functions


def laplacian(v, irho):
    """
    Laplacian with density div( 1/rho grad) (u)
    """
    func = list(retrieve_functions(v))[0]
    dims = func.space_dimensions
    if irho is None or irho == 1:
        Lap = v.laplace
    else:
        if isinstance(irho, Function):
            Lap = grad(irho).T * grad(v) + irho * v.laplace
        else:
            Lap = irho * v.laplace

    return Lap


def rotated_weighted_lap(u, v, costheta, sintheta, cosphi, sinphi,
                         epsilon, delta, irho, fw=True):
    """
    TTI finite difference kernel. The equation we solve is:
    u.dt2 = (1+2 *epsilon) (Gxx(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)) +  Gzz(v)
    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)
    :param u: first TTI field
    :param v: second TTI field
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle, has to be 0 in 2D
    :param sinphi: sine of the azymuth angle, has to be 0 in 2D
    :param space_order: discretization order
    :return: u and v component of the rotated Laplacian in 2D
    """
    if fw:
        Gxx = Gxxyy(u, costheta, sintheta, cosphi, sinphi, irho)
        Gzzr = Gzz(v, costheta, sintheta, cosphi, sinphi, irho)
        return (epsilon * Gxx + delta * Gzzr, delta * Gxx + Gzzr)
    else:
        a = epsilon * u + delta * v
        b = delta * u + v
        H0 = Gxxyy(a, costheta, sintheta, cosphi, sinphi, irho)
        H1 = Gzz(b, costheta, sintheta, cosphi, sinphi, irho)
        return H0, H1


def Gzz(field, costheta, sintheta, cosphi, sinphi, irho):
    """
    3D rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    if field.grid.dim == 2:
        return Gzz2d(field, costheta, sintheta, irho)

    order1 = field.space_order // 2
    func = list(retrieve_functions(field))[0]
    x, y, z = func.space_dimensions
    Gz = -(sintheta * cosphi * first_derivative(field, dim=x, side=centered,
                                                fd_order=order1) +
           sintheta * sinphi * first_derivative(field, dim=y, side=centered,
                                                fd_order=order1) +
           costheta * first_derivative(field, dim=z, side=centered,
                                       fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta * cosphi * irho,
                            dim=x, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * sintheta * sinphi * irho,
                            dim=y, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta * irho,
                            dim=z, side=centered, fd_order=order1,
                            matvec=transpose))
    return Gzz


def Gzz2d(field, costheta, sintheta, irho):
    """
    3D rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt ztranspose
    """
    order1 = field.space_order // 2
    func = list(retrieve_functions(field))[0]
    x, z = func.space_dimensions
    Gz = -(sintheta * first_derivative(field, dim=x, side=centered, fd_order=order1) +
           costheta * first_derivative(field, dim=z, side=centered, fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta * irho, dim=x, side=centered,
                            fd_order=order1, matvec=transpose) +
           first_derivative(Gz * costheta * irho, dim=z, side=centered,
                            fd_order=order1, matvec=transpose))
    return Gzz


# Centered case produces directly Gxx + Gyy
def Gxxyy(field, costheta, sintheta, cosphi, sinphi, irho):
    """
    Sum of the 3D rotated second order derivative in the direction x and y.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx + Gyy = field.laplace - Gzz
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: Sum of the 3D rotated second order derivative in the direction x and y
    """
    lap = laplacian(field, irho)
    func = list(retrieve_functions(field))[0]
    if func.grid.dim == 2:
        Gzzr = Gzz2d(field, costheta, sintheta, irho)
    else:
        Gzzr = Gzz(field, costheta, sintheta, cosphi, sinphi, irho)
    return lap - Gzzr
