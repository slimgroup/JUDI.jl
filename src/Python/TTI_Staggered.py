# Acoustic wave equations with Devito
# Forward/adjoint nonlinear and Born modeling
# Authors: Mathias Louboutin, Philipp Witte
# Date: November 2017
#

# Import modules
from __future__ import print_function
import numpy as np
from scipy import interpolate

from functools import reduce
from operator import mul

from sympy import cos, sin, finite_diff_weights
from devito.logger import error

from devito import Eq, Function, TimeFunction, Operator, clear_cache, Inc, ConditionalDimension, Grid
from devito.finite_difference import (centered, right, left)
from devito.symbolics import retrieve_functions

from PySource import PointSource, Receiver

def J_transpose(model, src_coords, wavelet, rec_coords, recin, space_order=12,
                t_sub_factor=20, h_sub_factor=2, isic='noop'):
    rec, ph, pv = forward_modeling(model, src_coords, wavelet, rec_coords, save=True,
                                   space_order=space_order, h_sub_factor=h_sub_factor,
                                   t_sub_factor=t_sub_factor)
    grad = adjoint_born(model, rec_coords, recin, ph=ph, pv=pv,
                        space_order=space_order, isic=isic)
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
    """
    Generalized staggered finite  difference for expression instead of Functions
    f.diff(dim).as_finite_difference(indx, x0=x0)
    where f is an expression
    """
    deriv = 0
    coeffs = finite_diff_weights(1, indx, x0)
    coeffs = coeffs[-1][-1]
    # Loop through positions
    for i in range(0, len(indx)):
        var = [a.subs({dim: indx[i]}) for a in args]
        deriv += coeffs[i] * reduce(mul, var, 1)
    return deriv


def src_rec(model, fields, src_coords, rec_coords, src_data, backward=False):
    """
    Source and receiver setup
    """
    nt = src_data.shape[0]
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = src_data[:]
    inv = fields[0].backward if backward else fields[0].forward
    inh = fields[1].backward if backward else fields[1].forward
    src_term = src.inject(field=inv, offset=model.nbpml,
                          expr=src * model.rho * s / model.m)
    src_term += src.inject(field=inh, offset=model.nbpml,
                           expr=src * model.rho * s / model.m)

    # Data is sampled at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=fields[0] + fields[1], offset=model.nbpml)

    return rec, rec_term, src_term

def pressure_fields(model, space_order, save=0, t_sub_factor=1, h_sub_factor=1, name=''):
    """
    Initialize pressure fields with time hisytory and space/time subsampling
    :param model: physical model
    :param space_order: FD spatial order
    :param save: time histroy size or 0 if no save
    :param t_sub_factor: time subsampling factor
    :param h_sub_factor: space subsampling factor
    :param name: name extension forthe fields
    """
    # Create the forward wavefield
    phsave, pvsave = None, None
    if save>1 and (t_sub_factor>1 or h_sub_factor>1):
        pv = TimeFunction(name='pv'+name, grid=model.grid, time_order=1, space_order=space_order)
        ph = TimeFunction(name='ph'+name, grid=model.grid, time_order=1, space_order=space_order)
        if t_sub_factor > 1:
            time_subsampled = ConditionalDimension('t_sub', parent=ph.grid.time_dim, factor=t_sub_factor)
            nsave = (save-1)//t_sub_factor + 2
        else:
            time_subsampled = model.grid.time_dim
            nsave = save

        if h_sub_factor > 1:
            grid2 = Grid(shape=tuple([i//2 for i in ph.data.shape[1:]]),
                         extent=ph.grid.extent, dimensions=sub_ind(model))
        else:
            grid2 = model.grid

        phsave = TimeFunction(name='phs'+name, grid=grid2, time_order=1, space_order=space_order,
                              time_dim=time_subsampled, save=nsave)
        pvsave = TimeFunction(name='pvs'+name, grid=grid2, time_order=1, space_order=space_order,
                              time_dim=time_subsampled, save=nsave)
        eqsave = [Eq(phsave, ph), Eq(pvsave, pv)]
    elif save>1 and t_sub_factor==1 and h_sub_factor==1:
        pv = TimeFunction(name='pv'+name, grid=model.grid, time_order=1, space_order=space_order,
                          save=save)
        ph = TimeFunction(name='ph'+name, grid=model.grid, time_order=1, space_order=space_order,
                          save=save)
        eqsave = []
    else:
        pv = TimeFunction(name='pv'+name, grid=model.grid, time_order=1, space_order=space_order)
        ph = TimeFunction(name='ph'+name, grid=model.grid, time_order=1, space_order=space_order)
        eqsave = []

    return (pv, ph), (pvsave, phsave), eqsave


def forward_stencil(model, space_order, save=0, q=(0, 0, 0, 0, 0), name='', h_sub_factor=1,
                    t_sub_factor=1):
    """
    Forward wave equation
    # Solves the TTI elastic system:
    rho * vx.dt = - ph.dx
    rho * vy.dt = - ph.dy
    rho * vz.dt = - pv.dz
    m / rho * ph.dt = - sqrt(1 + 2 delta) (vx.dx + vy.dy) - vz.dz + Fh
    m / rho * phv.dt = - (1 + 2 epsilon) (vx.dx + vy.dy) - sqrt(1 + 2 delta) vz.dz + Fv

    """
    m, epsilon, delta, theta, phi, rho = (model.m, model.epsilon, model.delta,
                                          model.theta, model.phi, model.rho)
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

    vy = 0
    if model.grid.dim == 3:
        vy = TimeFunction(name='vy'+name, grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)

    (pv, ph), saved_fields, eqsave = pressure_fields(model, space_order, save=save,
                                                     t_sub_factor=t_sub_factor,
                                                     h_sub_factor=h_sub_factor,
                                                     name=name)
    # Stencils
    u_vx = [Eq(vx.forward, damp * vx - damp *s/rho*(staggered_diff(ph, dim=x, order=space_order, stagger=left, theta=theta, phi=phi) + q[0]))]
    u_vz = [Eq(vz.forward, damp * vz - damp *s/rho*(staggered_diff(pv, dim=z, order=space_order, stagger=left, theta=theta, phi=phi) + q[2]))]

    dvx = staggered_diff(vx.forward, dim=x, order=space_order, stagger=right, theta=theta, phi=phi)
    dvz = staggered_diff(vz.forward, dim=z, order=space_order, stagger=right, theta=theta, phi=phi)

    u_vy = []
    dvy = 0
    if ndim == 3:
        u_vy = [Eq(vy.forward, damp * vy - damp *s/rho*(staggered_diff(ph, dim=y, order=space_order, stagger=left, theta=theta, phi=phi) + q[1]))]
        dvy = staggered_diff(vy.forward, dim=y, order=space_order, stagger=right, theta=theta, phi=phi)


    pv_eq = Eq(pv.forward, damp * pv - damp *s * rho / m * (delta*(dvx + dvy) + dvz + q[3]))

    ph_eq = Eq(ph.forward, damp * ph - damp *s * rho / m * (epsilon*(dvx + dvy) + delta * dvz + q[4]))

    vel_expr = u_vx + u_vy + u_vz
    pressure_expr = [ph_eq, pv_eq] + eqsave
    if save>1 and (t_sub_factor>1 or h_sub_factor>1):
        out_fields = saved_fields
    else:
        out_fields = (ph, pv)
    return vel_expr, pressure_expr, (ph, pv),  (vx, vy, vz), out_fields


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

    vy = 0
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
    return vel_expr, pressure_expr, (ph, pv), (vx, vy, vz)


def imaging_condition(model, ph, pv, fields, vel_fields, isic='noop'):
    """
    FWI or RTM imaging conditiongradient
    # grad = .5 * sum_t ph.dt * ph_a + pv.dt * pv_a
    with the .5 gradient to guaranty we have the same as acoustic when ph=pv
    :param ph: forward ph
    :param pv: forward pv
    :param fields: adjoint wavefields
    :param isic: typy of isic imaging condition to choose from
    `None(defauly, FWI)``, `isotrpopic`, `rotated`
    """
    space_order = ph.space_order

    m, rho = model.m, model.rho

    grad = Function(name="grad", grid=ph.grid, space_order=ph.space_order)
    inds = model.grid.dimensions

    phadt = -fields[0].dt.subs({model.grid.stepping_dim: model.grid.stepping_dim - model.grid.stepping_dim.spacing})
    pvadt = -fields[1].dt.subs({model.grid.stepping_dim: model.grid.stepping_dim - model.grid.stepping_dim.spacing})

    factor = model.grid.time_dim.spacing
    if ph.indices[0].is_Conditional:
        factor *= ph.indices[0].factor

    if isic == 'noop':
        gradh = Function(name="gradh", grid=ph.grid, space_order=ph.space_order)
        gradv = Function(name="gradv", grid=ph.grid, space_order=ph.space_order)
        grad_expr = [Eq(gradh, gradh - .5 * factor * ph * fields[0] / rho)]
        grad_expr += [Eq(gradv, gradv - .5 * factor * pv * fields[1] / rho)]
        if model.grid.dim == 2:
            theta, phi = model.theta, model.phi
            dgx = staggered_diff(gradh, dim=inds[0], order=space_order, stagger=left, theta=theta, phi=phi)
            dgx2 = staggered_diff(dgx, dim=inds[0], order=space_order, stagger=left, theta=theta, phi=phi)

            dgz = staggered_diff(gradv, dim=inds[-1], order=space_order, stagger=left, theta=theta, phi=phi)
            dgz2 = staggered_diff(dgz, dim=inds[-1], order=space_order, stagger=left, theta=theta, phi=phi)
            grad_expr += [Eq(grad, model.epsilon*dgx2 + model.delta*dgz2)]
        else:
            grad_expr += [Eq(grad, model.epsilon*(gradh.dx2 + gradh.dy2) + model.delta*gradv.dz2)]
    else:
        if isic == 'isotropic':
            theta, phi = 0, 0
        elif isic == 'rotated':
            theta, phi = model.theta, model.phi
        else:
            error('Unrecognized imaging condition %s' % isic)
        divs = (staggered_diff(ph, dim=inds[0], order=space_order, stagger=left, theta=theta, phi=phi) * vel_fields[0])
                # staggered_diff(vel_fields[0], dim=inds[0], order=space_order, stagger=left, theta=theta, phi=phi))
        if model.grid.dim == 3:
            divs += staggered_diff(ph, dim=inds[1], order=space_order, stagger=left, theta=theta, phi=phi) * vel_fields[1]
        divs += (staggered_diff(pv, dim=inds[-1], order=space_order, stagger=left, theta=theta, phi=phi) * vel_fields[-1])
                 # staggered_diff(vel_fields[-1], dim=inds[-1], order=space_order, stagger=left, theta=theta, phi=phi))
        grad_expr = [Inc(grad, grad + .5 * factor * (m * ph * phadt /rho + m * pv * pvadt / rho - divs) / rho)]

    return grad, grad_expr


def linearized_source(model, fields, part_vel, isic='noop'):
    """
    FWI or RTM imaging condition
    :param ph: forward ph
    :param pv: forward pv
    :param isic: typy of isic imaging condition to choose from
    `None(defauly, FWI)``, `isotrpopic`, `rotated`
    """
    space_order = fields[0].space_order
    m, dm, rho = model.m, model.dm, model.rho
    gradient = Function(name="grad", grid=model.grid)
    inds = model.grid.dimensions
    if isic == 'noop':
        lin_src = (0, 0, 0, dm * fields[0] / rho, dm * fields[1] / rho)
    else:
        if isic == 'isotropic':
            theta, phi = 0, 0
        elif isic == 'rotated':
            theta, phi = model.theta, model.phi
        else:
            error('Unrecognized imaging condition %s' % isic)
        dvx = .5 * staggered_diff(fields[0], dim=inds[0], order=space_order, stagger=right, theta=theta, phi=phi)
        dvy = 0
        if model.grid.dim == 3:
            dvy = .5 * staggered_diff(fields[0], dim=inds[1], order=space_order, stagger=right, theta=theta, phi=phi)
        dvz = .5 *  staggered_diff(fields[-1], dim=inds[-1], order=space_order, stagger=right, theta=theta, phi=phi)
        dph = .5 * m * fields[0].dt
        dpv = .5 * m * fields[1].dt

        lin_src = (dm * dvx / rho, dm * dvy / rho, dm * dvz / rho, dm * dph / rho, dm * dpv / rho)

    return lin_src


def forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=16,
                     h_sub_factor=1, t_sub_factor=1):
    """
    Constructor method for the forward modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    save_p = wavelet.shape[0] if save else 0
    vel_expr, p_expr, fields, _, saved_fields = forward_stencil(model, space_order, save=save_p,
                                                                h_sub_factor=h_sub_factor,
                                                                t_sub_factor=t_sub_factor)
    # Source and receivers
    rec, rec_term, src_term = src_rec(model, fields, src_coords, rec_coords, wavelet)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + src_term, subs=model.spacing_map,
                  dse='advanced', dle='advanced')

    op()
    return rec.data, saved_fields[0], saved_fields[1]


def adjoint_modeling(model, src_coords, rec_coords, rec_data, space_order=16):
    """
    Constructor method for the adjoint modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    vel_expr, p_expr, fields, _ = adjoint_stencil(model, space_order)
    # Adjoint source is injected at receiver locations
    rec, rec_term, src_term = src_rec(model, fields, rec_coords, src_coords, rec_data,
                                      backward=True)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + src_term, subs=model.spacing_map,
                  dse='advanced', dle='advanced')
    op()
    return rec.data, fields[0], fields[1]


def forward_born(model, src_coords, wavelet, rec_coords, space_order=16, isic='noop', save=False,
                 h_sub_factor=1, t_sub_factor=1):
    """
    Constructor method for the born modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    save_p = wavelet.shape[0] if save else 0
    vel_expr, p_expr, fields, part_vel, saved_fields = forward_stencil(model, space_order, save=save_p,
                                                                       h_sub_factor=h_sub_factor,
                                                                       t_sub_factor=t_sub_factor)
    _, _, src_term = src_rec(model, fields, src_coords, rec_coords, src_data=wavelet)

    lin_src =  linearized_source(model, fields, part_vel, isic=isic)
    vel_exprl, p_exprl, fieldsl, _, _ = forward_stencil(model, space_order, q=lin_src, name='lin')
    # Source and receivers
    rec, rec_term, _ = src_rec(model, fieldsl, src_coords, rec_coords, src_data=wavelet)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + vel_exprl + src_term + p_exprl, subs=model.spacing_map,
                  dse='advanced', dle='advanced')

    op()
    return rec.data, saved_fields[0], saved_fields[1]


def adjoint_born(model, rec_coords, rec_data, ph=None, pv=None, space_order=16, isic='noop'):
    """
    Constructor method for the adjoint born modelling operator in an acoustic media
    :param model: :class:`Model` object containing the physical parameters
    :param source: :class:`PointData` object containing the source geometry
    :param receiver: :class:`PointData` object containing the acquisition geometry
    :param space_order: Space discretization order
    :param save: Saving flag, True saves all time steps, False only the three
    """
    clear_cache()
    vel_expr, p_expr, fields, vel_fields = adjoint_stencil(model, space_order)

    resample = False
    if ph.indices[1].is_Conditional:
        resample = True
    grad, grad_expr = imaging_condition(model, ph, pv, fields, vel_fields, isic=isic)

    # Adjoint source is injected at receiver locations
    _, _, src_term = src_rec(model, fields, rec_coords, np.zeros((1, 1)),
                               rec_data, backward=True)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + p_expr + src_term + grad_expr, subs=model.spacing_map,
                  dse='advanced', dle='advanced')
    op()
    if resample:
        grad = resample_grad(grad, model, ph.indices[1].factor)
        return grad
    else:
        return grad.data


def resample_grad(grad, model, factor):
    x = [i*factor for i in range(grad.data.shape[0])]
    xnew = [i for i in range(model.shape_pml[0])]
    y = [i*factor for i in range(grad.data.shape[1])]
    ynew = [i for i in range(model.shape_pml[1])]
    if model.grid.dim > 2:
        z = [i*factor for i in range(grad.data.shape[2])]
        znew = [i for i in range(model.shape_pml[2])]
        interpolator = interpolate.RegularGridInterpolator((x, y, z), grad.data,
                                                           bounds_error=False,
                                                           fill_value=0.)
        gridnew = np.ix_(xnew, ynew, znew)
    else:
        interpolator = interpolate.RegularGridInterpolator((x, y), grad.data,
                                                           bounds_error=False,
                                                           fill_value=0.)
        gridnew = np.ix_(xnew, ynew)
    return interpolator(gridnew)


def staggered_diff(*args, dim, order, stagger=centered, theta=0, phi=0):
    """
    Utility function to generate staggered derivatives
    """
    func = list(retrieve_functions(*args))
    for i in func:
        if isinstance(i, Function):
            dims = i.space_dimensions
            ndim = i.grid.dim
            break
    off = dict([(d, 0) for d in dims])
    if stagger == left:
        off[dim] = -.5
    elif stagger == right:
        off[dim] = .5
    else:
        off[dim] = 0

    if theta == 0 and phi == 0:
        diff = dim.spacing
        idx = [(dim + int(i+.5+off[dim])*diff)
               for i in range(-int(order / 2), int(order / 2))]
        return custom_FD(args, dim, idx, dim + off[dim]*dim.spacing)
    else:
        x = dims[0]
        z = dims[-1]
        idxx = list(set([(x + int(i+.5+off[x])*x.spacing)
                         for i in range(-int(order / 2), int(order / 2))]))
        dx = custom_FD(args, x, idxx, x + off[x]*x.spacing)

        idxz = list(set([(z + int(i+.5+off[z])*z.spacing)
                         for i in range(-int(order / 2), int(order / 2))]))
        dz = custom_FD(args, z, idxz, z + off[z]*z.spacing)

        dy = 0
        is_y = False

        if ndim == 3:
            y = dims[1]
            idxy = list(set([(y + int(i+.5+off[y])*y.spacing)
                             for i in range(-int(order / 2), int(order / 2))]))
            dy = custom_FD(args, y, idxy, y + off[y]*y.spacing)
            is_y = (dim == y)

        if dim == x:
            return cos(theta) * cos(phi) * dx + sin(phi) * cos(theta) * dy - sin(theta) * dz
        elif dim == z:
            return sin(theta) * cos(phi) * dx + sin(phi) * sin(theta) * dy + cos(theta) * dz
        elif is_y:
            return -sin(phi) * dx + cos(phi) * dy
        else:
            return 0
