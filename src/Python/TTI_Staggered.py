# Acoustic wave equations with Devito
# Forward/adjoint nonlinear and Born modeling
# Authors: Mathias Louboutin, Philipp Witte
# Date: November 2017
#

# Import modules
from __future__ import print_function
import numpy as np

from functools import reduce
from operator import mul

from sympy import cos, sin, finite_diff_weights
from devito.logger import set_log_level, error

from devito import Eq, Function, TimeFunction, Operator, clear_cache, Grid, Inc, ConditionalDimension
from devito.finite_difference import (centered, right, left)
from devito.symbolics import retrieve_functions

from PySource import PointSource, Receiver
from PyModel import Model

# And gradient
# grad = sum_t ph.dt2 * ph_a + pv.dt2 * pv_a


def J_transpose(model, src_coords, wavelet, rec_coords, recin, space_order=12, nb=40,
                t_sub_factor=20, h_sub_factor=2, op_return=False, dt=None, isic=False):
    rec, u0, v0 = forward_modeling(model, src_coords, wavelet, rec_coords, save=True, space_order=space_order, nb=nb,
                                   t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor, dt=dt)
    grad = adjoint_born(model, rec_coords, recin, u=u0, v=v0, space_order=space_order, nb=nb, isiciso=isic, dt=dt)
    return grad

def sub_ind(model):
    """
    Dimensions of the inner part of the domain without ABC layers
    """
    sub_dim =[]
    for dim in model.grid.dimensions:
        sub_dim += [ConditionalDimension(dim.name + '_in', parent=dim, factor=2)]

    return tuple(sub_dim)

def custom_FD(args, dim, indx, x0):
    # f.diff(dim).as_finite_difference(indx, x0=x0)
    deriv = 0
    coeffs = finite_diff_weights(1, indx, x0)
    coeffs = coeffs[-1][-1]
    # Loop through positions
    for i in range(0, len(indx)):
        var = [a.subs({dim: indx[i]}) for a in args]
        deriv += coeffs[i] * reduce(mul, var, 1)
    return deriv

def src_rec(model, fields, src_coords, rec_coords, src_data, backward=False):
    nt = src_data.shape[0]
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = src_data[:]
    inv = fields[0].backward if backward else fields[0].forward
    inh = fields[1].backward if backward else fields[1].forward
    src_term = src.inject(field=inv, offset=model.nbpml, expr=src * model.rho * s / model.m)
    src_term += src.inject(field=inh, offset=model.nbpml, expr=src * model.rho * s / model.m)

    # Data is sampled at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=fields[0] + fields[1], offset=model.nbpml)

    return rec, rec_term, src_term

def forward_stencil(model, space_order, save=None, q=(0, 0), name=''):
    """
    Forward wave equation
    # Solves the TTI elastic system:
    rho * vx.dt = - ph.dx
    rho * vy.dt = - ph.dx
    rho * vz.dt = - pv.dx
    m / rho * ph.dt = - sqrt(1 + 2 delta) (vx.dx + vy.dy) - vz.dz + Fh
    m / rho * phv.dt = - sqrt(1 + 2 epsilon) (vx.dx + vy.dy) - sqrt(1 + 2 delta) vz.dz + Fv

    """
    m, epsilon, delta, theta, phi, rho = model.m, model.epsilon, model.delta, model.theta, model.phi, model.rho
    damp = model.damp
    s = model.grid.stepping_dim.spacing

    ndim = model.grid.dim

    if ndim == 3:
        stagg_x = (0, 1, 0, 0)
        stagg_z = (0, 0, 0, 1)
        stagg_y = (0, 0, 1, 0)
        x, y, z = model.grid.dimensions
    else:
        stagg_x = (0, 1, 0)
        stagg_z = (0, 0, 1)
        x, z = model.grid.dimensions
    # Create symbols for forward wavefield, source and receivers
    vx = TimeFunction(name='vx'+name, grid=model.grid, staggered=stagg_x,
                      time_order=1, space_order=space_order)
    vz = TimeFunction(name='vz'+name, grid=model.grid, staggered=stagg_z,
                      time_order=1, space_order=space_order)

    if model.grid.dim == 3:
        vy = TimeFunction(name='vy'+name, grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)

    pv = TimeFunction(name='pv'+name, grid=model.grid, save=save,
                      time_order=1, space_order=space_order)
    ph = TimeFunction(name='ph'+name, grid=model.grid, save=save,
                      time_order=1, space_order=space_order)
    # Stencils
    u_vx = [Eq(vx.forward, damp * vx - damp *s/rho*staggered_diff(ph, dim=x, order=space_order, stagger=left, theta=theta, phi=phi))]
    u_vz = [Eq(vz.forward, damp * vz - damp *s/rho*staggered_diff(pv, dim=z, order=space_order, stagger=left, theta=theta, phi=phi))]

    dvx = staggered_diff(vx.forward, dim=x, order=space_order, stagger=right, theta=theta, phi=phi)
    dvz = staggered_diff(vz.forward, dim=z, order=space_order, stagger=right, theta=theta, phi=phi)

    u_vy = []
    dvy = 0
    if ndim == 3:
        u_vy = [Eq(vy.forward, damp * vy - damp *s/rho*staggered_diff(ph, dim=y, order=space_order, stagger=left, theta=theta, phi=phi))]
        dvy = staggered_diff(vy.forward, dim=y, order=space_order, stagger=right, theta=theta, phi=phi)


    pv_eq = Eq(pv.forward, damp * pv - damp *s * rho / m * (delta*(dvx + dvy) + dvz + q[0]))

    ph_eq = Eq(ph.forward, damp * ph - damp *s * rho / m * (epsilon*(dvx + dvy) + delta * dvz + q[1]))

    vel_expr = u_vx + u_vy + u_vz
    pressure_expr = [ph_eq, pv_eq]
    return vel_expr, pressure_expr, (ph, pv)


def adjoint_stencil(model, space_order):
    """
    Adjoint wave equation stencil
    rho * vxa.dt = - d/dx ( sqrt(1 + 2 delta) pva + (1 + 2 epsilon) pha  )
    rho * vya.dt = - d/dy ( sqrt(1 + 2 delta) pva + (1 + 2 epsilon) pha  )
    rho * vza.dt = - d/dz ( pva +  sqrt(1 + 2 delta) pha  )
    m / rho * pha.dt = - vza.dz + Fha
    m / rho * pva.dt = - vxa.dx - vya.dy + Fva
    """
    m, epsilon, delta, theta, phi, rho = model.m, model.epsilon, model.delta, model.theta, model.phi, model.rho
    damp = model.damp
    s = model.grid.stepping_dim.spacing

    ndim = model.grid.dim

    if ndim == 3:
        stagg_x = (0, 1, 0, 0)
        stagg_z = (0, 0, 0, 1)
        stagg_y = (0, 0, 1, 0)
        x, y, z = model.grid.dimensions
    else:
        stagg_x = (0, 1, 0)
        stagg_z = (0, 0, 1)
        x, z = model.grid.dimensions
    # Create symbols for forward wavefield, source and receivers
    vx = TimeFunction(name='vxa', grid=model.grid, staggered=stagg_x,
                      time_order=1, space_order=space_order)
    vz = TimeFunction(name='vza', grid=model.grid, staggered=stagg_z,
                      time_order=1, space_order=space_order)

    if model.grid.dim == 3:
        vy = TimeFunction(name='vya', grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)

    pv = TimeFunction(name='pva', grid=model.grid,
                      time_order=1, space_order=space_order)
    ph = TimeFunction(name='pha', grid=model.grid,
                      time_order=1, space_order=space_order)
    # Stencils
    u_vx = [Eq(vx.backward, damp * vx - damp * s / rho * staggered_diff(delta*pv + epsilon*ph, dim=x, order=space_order, stagger=left, theta=theta, phi=phi))]
    u_vz = [Eq(vz.backward, damp * vz - damp * s / rho * staggered_diff(pv + delta*ph, dim=z, order=space_order, stagger=left, theta=theta, phi=phi))]

    dvx = staggered_diff(vx.backward, dim=x, order=space_order, stagger=right, theta=theta, phi=phi)
    dvz = staggered_diff(vz.backward, dim=z, order=space_order, stagger=right, theta=theta, phi=phi)

    u_vy = []
    dvy = 0
    if ndim == 3:
        u_vy = [Eq(vy.backward, damp * vy - damp * s / rho * staggered_diff(delta*pv + epsilon*ph, dim=y, order=space_order, stagger=left, theta=theta, phi=phi))]
        dvy = staggered_diff(vy.backward, dim=y, order=space_order, stagger=right, theta=theta, phi=phi)


    pv_eq = Eq(pv.backward, damp * pv - damp * s * rho / m * dvz)

    ph_eq = Eq(ph.backward, damp * ph - damp * s * rho / m * (dvx + dvy))

    vel_expr = u_vx + u_vy + u_vz
    pressure_expr = [ph_eq, pv_eq]
    return vel_expr, pressure_expr, (ph, pv)

def forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=16, nb=40,
                      t_sub_factor=1, h_sub_factor=1, op_return=False, dt=None, freesurface=False):
    """
    Constructor method for the forward modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    save_p = wavelet.shape[0] if save else None

    vel_expr, p_expr, fields = forward_stencil(model, space_order, save=save_p)
    # Source and receivers
    rec, rec_term, src_term = src_rec(model, fields, src_coords, rec_coords, wavelet)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + src_term, subs=model.spacing_map,
                  dse='advanced', dle='advanced')

    op()
    return rec.data, fields[0], fields[1]


def adjoint_modeling(model, src_coords, rec_coords, rec_data, space_order=12, nb=40, dt=None):
    """
    Constructor method for the adjoint modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    vel_expr, p_expr, fields = adjoint_stencil(model, space_order)
    # Adjoint source is injected at receiver locations
    rec, rec_term, src_term = src_rec(model, fields, rec_coords, src_coords, rec_data, backward=True)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + src_term, subs=model.spacing_map,
                  dse='aggressive', dle='advanced')
    op()
    return rec.data, fields[0], fields[1]

def forward_born(model, src_coords, wavelet, rec_coords, space_order=12, nb=40, isic=False, dt=None, save=False, isiciso=False,
                 h_sub_factor=1):
    """
    Constructor method for the born modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    save_p = source.nt if save else None

    vel_expr, p_expr, fields = forward_stencil(model, space_order, save=save_p)
    _, _, src_term = src_rec(model, fields, src_coords, rec_coords, src_data=wavelet)

    lin_src = (model.dm * fields[0].dt / model.rho, model.dm * fields[1].dt / model.rho)
    vel_exprl, p_exprl, fieldsl = forward_stencil(model, space_order, save=save_p, q=lin_src, name='lin')
    # Source and receivers
    rec, rec_term, _ = src_rec(model, fieldsl, src_coords, rec_coords, src_data=wavelet)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + vel_exprl + src_term + p_exprl, subs=model.spacing_map,
                  dse='advanced', dle='advanced')

    op()
    return rec.data, fields[0], fields[1]


def adjoint_born(model, rec_coords, rec_data, u=None, v=None, op_forward=None, is_residual=False,
                 space_order=12, nb=40, isic=False, isiciso=False, isicnothom=False, dt=None):
    """
    Constructor method for the adjoint born modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    vel_expr, p_expr, fields = adjoint_stencil(model, space_order)

    grad = Function(name="grad", grid=model.grid)
    gradient = [Inc(grad, grad - model.grid.time_dim.spacing *(u.dt * fields[0] - v.dt * fields[1]))]
    # Adjoint source is injected at receiver locations
    rec, _, src_term = src_rec(model, fields, rec_coords, np.zeros((1,1)), rec_data, backward=True)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr  + p_expr + src_term + gradient , subs=model.spacing_map,
                  dse='aggressive', dle='advanced')
    op()
    return grad.data


def resample_grad(grad, model, factor):
    from scipy import interpolate
    x = [i*factor for i in range(grad.data.shape[0])]
    xnew = [i for i in range(model.shape_pml[0])]
    y = [i*factor for i in range(grad.data.shape[1])]
    ynew = [i for i in range(model.shape_pml[1])]
    if model.grid.dim > 2:
        z = [i*factor for i in range(grad.data.shape[2])]
        znew = [i for i in range(model.shape_pml[2])]
        interpolator = interpolate.RegularGridInterpolator((x, y, z), grad.data, bounds_INFO=False,fill_value=0.)
        gridnew = np.ix_(xnew, ynew, znew)
    else:
        interpolator = interpolate.RegularGridInterpolator((x, y), grad.data, bounds_INFO=False, fill_value=0.)
        gridnew = np.ix_(xnew, ynew)
    return interpolator(gridnew)


def staggered_diff(*args, dim, order, stagger=centered, theta=0, phi=0):
    """
    Utility function to generate staggered derivatives
    """
    func = list(retrieve_functions(*args))[0]
    ndim = func.grid.dim
    off = dict([(d, 0) for d in func.grid.dimensions])
    if stagger == left:
        off[dim] = -.5
    elif stagger == right:
        off[dim] = .5
    else:
        off[dim] = 0

    if theta == 0 and phi == 0:
        diff = dim.spacing
        idx = [(dim + int(i+.5+off[dim])*diff) for i in range(-int(order / 2), int(order / 2))]
        return custom_FD(args, dim, idx, dim + off[dim]*dim.spacing)
    else:
        x = func.grid.dimensions[0]
        z = func.grid.dimensions[-1]
        idxx = list(set([(x + int(i+.5+off[x])*x.spacing) for i in range(-int(order / 2), int(order / 2))]))
        dx = custom_FD(args, x, idxx, x + off[x]*x.spacing)

        idxz = list(set([(z + int(i+.5+off[z])*z.spacing) for i in range(-int(order / 2), int(order / 2))]))
        dz = custom_FD(args, z, idxz, z + off[z]*z.spacing)

        dy = 0
        is_y = False

        if ndim == 3:
            y = func.grid.dimensions[1]
            idxy = list(set([(y + int(i+.5+off[y])*y.spacing) for i in range(-int(order / 2), int(order / 2))]))
            dy = custom_FD(args, y, idxy, y + off[y]*y.spacing)
            is_y = (dim == y)

        if dim == x:
            return cos(theta) * cos(phi) * dx + sin(phi) * cos(theta) * dy - sin(theta) * dz
        elif dim == z:
            return sin(theta) * cos(phi) * dx + sin(phi) * sin(theta) * dy + cos(theta) * dz
        elif is_y:
            return -sin(phi) * dx + cos(phi) *  dy
        else:
            return 0
