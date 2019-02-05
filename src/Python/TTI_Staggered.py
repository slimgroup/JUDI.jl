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
from devito.finite_differences import (centered, right, left)
from devito.symbolics import retrieve_functions

from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
from PySource import PointSource, Receiver


def J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=12,
                t_sub_factor=20, h_sub_factor=2, checkpointing=False, free_surface=False,
                n_checkpoints=None, maxmem=None, dt=None, isic=False):
    if checkpointing:
        F = forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=space_order,
                             t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor,op_return=True,
                             free_surface=free_surface)
        grad = adjoint_born(model, rec_coords, recin, op_forward=F, space_order=space_order,
                            is_residual=True, isic=isic, n_checkpoints=n_checkpoints, maxmem=maxmem,
                            free_surface=free_surface)
    else:
        ph, pv = forward_modeling(model, src_coords, wavelet, None, save=True, space_order=space_order,
                                  t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor, free_surface=free_surface)
        grad = adjoint_born(model, rec_coords, recin, ph=ph, pv=pv, space_order=space_order, isic=isic, free_surface=free_surface,
                            t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor)

    return grad


def sub_ind(model):
    """
    Dimensions of the inner part of the domain without ABC layers
    """
    sub_dim =[]
    for dim in model.grid.dimensions:
        sub_dim += [ConditionalDimension(dim.name + '_in', parent=dim, factor=2)]

    return tuple(sub_dim)

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
    src_term = src.inject(field=inv, expr=src * model.rho * s / model.m)
    src_term += src.inject(field=inh, expr=src * model.rho * s / model.m)

    # Data is sampled at receiver locations
    if rec_coords is not None:
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        rec_term = rec.interpolate(expr=fields[0] + fields[1])

        return rec, rec_term, src_term, src
    else:
        return [], [], src_term, src


def particle_velocity(model, space_order, name=''):
    """
    Initialize particle velocity fields vx, vy, vz
    """
    ndim = model.grid.dim

    if ndim == 3:
        x, y, z = model.grid.dimensions
        stagg_x = x
        stagg_z = z
        stagg_y = y
    else:
        x, z = model.grid.dimensions
        stagg_x = x
        stagg_z = z
    # Create symbols for forward wavefield, source and receivers
    vx = TimeFunction(name='vx'+name, grid=model.grid, staggered=stagg_x,
                      time_order=1, space_order=space_order)
    vz = TimeFunction(name='vz'+name, grid=model.grid, staggered=stagg_z,
                      time_order=1, space_order=space_order)

    vy = 0
    if model.grid.dim == 3:
        vy = TimeFunction(name='vy'+name, grid=model.grid, staggered=stagg_y,
                          time_order=1, space_order=space_order)

    return vx, vy, vz

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
        pv = TimeFunction(name='pv'+name, grid=model.grid, time_order=1, space_order=space_order, staggered='NODE')
        ph = TimeFunction(name='ph'+name, grid=model.grid, time_order=1, space_order=space_order, staggered='NODE')
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
                          save=save, staggered='NODE')
        ph = TimeFunction(name='ph'+name, grid=model.grid, time_order=1, space_order=space_order,
                          save=save, staggered='NODE')
        eqsave = []
    else:
        pv = TimeFunction(name='pv'+name, grid=model.grid, time_order=1, space_order=space_order, staggered='NODE')
        ph = TimeFunction(name='ph'+name, grid=model.grid, time_order=1, space_order=space_order, staggered='NODE')
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
    x = model.grid.dimensions[0]
    y = model.grid.dimensions[1]
    z = model.grid.dimensions[-1]

    vx, vy, vz = particle_velocity(model, space_order, name)
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
    x = model.grid.dimensions[0]
    y = model.grid.dimensions[1]
    z = model.grid.dimensions[-1]

    vx, vy, vz = particle_velocity(model, space_order, name='a')

    pv = TimeFunction(name='pva', grid=model.grid, stagger='NODE',
                      time_order=1, space_order=space_order)
    ph = TimeFunction(name='pha', grid=model.grid, stagger='NODE',
                      time_order=1, space_order=space_order)
    # Stencils
    u_vx = [Eq(vx.backward, damp * vx - damp * s / rho * staggered_diff(delta*pv + epsilon*ph, dim=x,
                                                                        order=space_order, stagger=left,
                                                                        theta=theta, phi=phi, adjoint=True))]
    u_vz = [Eq(vz.backward, damp * vz - damp * s / rho * staggered_diff(pv + delta*ph, dim=z,
                                                                        order=space_order, stagger=left,
                                                                        theta=theta, phi=phi, adjoint=True))]

    dvx = staggered_diff(vx.backward, dim=x, order=space_order, stagger=right, theta=theta, phi=phi, adjoint=True)
    dvz = staggered_diff(vz.backward, dim=z, order=space_order, stagger=right, theta=theta, phi=phi, adjoint=True)

    u_vy = []
    dvy = 0
    if ndim == 3:
        u_vy = [Eq(vy.backward, damp * vy - damp * s / rho * staggered_diff(delta*pv + epsilon*ph, dim=y,
                                                                            order=space_order, stagger=left,
                                                                            theta=theta, phi=phi, adjoint=True))]
        dvy = staggered_diff(vy.backward, dim=y, order=space_order, stagger=right, theta=theta, phi=phi)


    pv_eq = Eq(pv.backward, damp * pv - damp * s * rho / m * dvz)

    ph_eq = Eq(ph.backward, damp * ph - damp * s * rho / m * (dvx + dvy))

    vel_expr = u_vx + u_vy + u_vz
    pressure_expr = [ph_eq, pv_eq]

    return vel_expr, pressure_expr, (ph, pv), (vx, vy, vz)


def imaging_condition(model, ph, pv, fields, vel_fields, isic=False):
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
    space_order = fields[0].space_order

    m, rho = model.m, model.rho

    grad = Function(name="grad", grid=fields[0].grid, space_order=space_order)
    inds = model.grid.dimensions

    phadt = -fields[0].dt.subs({model.grid.stepping_dim: model.grid.stepping_dim - model.grid.stepping_dim.spacing})
    pvadt = -fields[1].dt.subs({model.grid.stepping_dim: model.grid.stepping_dim - model.grid.stepping_dim.spacing})

    factor = model.grid.time_dim.spacing
    if ph.indices[0].is_Conditional:
        factor *= ph.indices[0].factor

    if isic:
        theta, phi = model.theta, model.phi
        divs = (staggered_diff(ph, dim=inds[0], order=space_order, stagger=left, theta=theta, phi=phi) * vel_fields[0])
        if model.grid.dim == 3:
            divs += staggered_diff(ph, dim=inds[1], order=space_order, stagger=left, theta=theta, phi=phi) * vel_fields[1]
        divs += (staggered_diff(pv, dim=inds[-1], order=space_order, stagger=left, theta=theta, phi=phi) * vel_fields[-1])
        grad_expr = [Inc(grad, .5 * factor * (m * ph * phadt /rho + m * pv * pvadt / rho - divs) / rho)]
    else:
        grad_expr = [Inc(grad, .5 * factor * (ph * phadt /rho + pv * pvadt / rho))]

    return grad, grad_expr


def linearized_source(model, fields, part_vel, isic=False):
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

    if isic:
        theta, phi = model.theta, model.phi
        dvx = .5 * staggered_diff(fields[0], dim=inds[0], order=space_order, stagger=right, theta=theta, phi=phi)
        dvy = 0
        if model.grid.dim == 3:
            dvy = .5 * staggered_diff(fields[0], dim=inds[1], order=space_order, stagger=right, theta=theta, phi=phi)
        dvz = .5 *  staggered_diff(fields[-1], dim=inds[-1], order=space_order, stagger=right, theta=theta, phi=phi)
        dph = .5 * m * fields[0].dt
        dpv = .5 * m * fields[1].dt

        lin_src = (dm * dvx / rho, dm * dvy / rho, dm * dvz / rho, dm * dph / rho, dm * dpv / rho)
    else:
        lin_src = (0, 0, 0, dm * fields[0] / rho, dm * fields[1] / rho)

    return lin_src


def forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=16,
                     h_sub_factor=1, t_sub_factor=1,op_return=False,
                     free_surface=False):
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
    rec, rec_term, src_term, _ = src_rec(model, fields, src_coords, rec_coords, wavelet)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + src_term, subs=model.spacing_map,
                  dse='advanced', dle='advanced')

    if op_return:
        return op
    else:
        op()
        if rec_coords is not None:
            return rec.data, saved_fields[0], saved_fields[1]
        else:
            return saved_fields[0], saved_fields[1]


def adjoint_modeling(model, src_coords, rec_coords, rec_data, space_order=16, free_surface=False):
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
    rec, rec_term, src_term, _ = src_rec(model, fields, rec_coords, src_coords, rec_data,
                                         backward=True)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + src_term, subs=model.spacing_map,
                  dse='advanced', dle='advanced')
    op()
    return rec.data


def forward_born(model, src_coords, wavelet, rec_coords, space_order=16, isic=False, save=False,
                 h_sub_factor=1, t_sub_factor=1, free_surface=False):
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
    _, _, src_term, _ = src_rec(model, fields, src_coords, rec_coords, src_data=wavelet)

    lin_src =  linearized_source(model, fields, part_vel, isic=isic)
    vel_exprl, p_exprl, fieldsl, _, _ = forward_stencil(model, space_order, q=lin_src, name='lin')
    # Source and receivers
    rec, rec_term, _, _ = src_rec(model, fieldsl, src_coords, rec_coords, src_data=wavelet)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + rec_term + p_expr + vel_exprl + src_term + p_exprl, subs=model.spacing_map,
                  dse='advanced', dle='advanced')

    op()
    return rec.data

def adjoint_born(model, rec_coords, rec_data, ph=None, pv=None, space_order=16, isic=False,
                 n_checkpoints=None, maxmem=None, op_forward=None, is_residual=False,
                 free_surface=False, t_sub_factor=1, h_sub_factor=1):
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
    adj_fields = {f.name: f for f in fields+vel_fields if hasattr(f, 'name')}
    resample = False
    if h_sub_factor>1:
        resample = True

    if op_forward is not None:
        pv = TimeFunction(name='pv', grid=model.grid,
                          time_order=1, space_order=space_order)
        ph = TimeFunction(name='ph', grid=model.grid,
                          time_order=1, space_order=space_order)
    grad, grad_expr = imaging_condition(model, ph, pv, fields, vel_fields, isic=isic)

    # Adjoint source is injected at receiver locations
    _, _, src_term, rec_g = src_rec(model, fields, rec_coords, np.zeros((1, 1)),
                             rec_data, backward=True)

    # Substitute spacing terms to reduce flops
    op = Operator(vel_expr + p_expr + src_term + grad_expr, subs=model.spacing_map,
                  dse='advanced', dle='advanced')

    if op_forward is not None:
        nt = rec_data.shape[0]
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        vx, vy, vz = particle_velocity(model, space_order)
        fwd_fields = {f.name: f for f in [vx, vy, vz, ph, pv] if hasattr(f, 'name')}
        cp = DevitoCheckpoint([v for k, v in fwd_fields.items()])
        if maxmem is not None:
            n_checkpoints = int(np.floor(maxmem * 10**6 / (cp.size * u.data.itemsize)))
        wrap_fw = CheckpointOperator(op_forward,  m=model.m, epsilon=model.epsilon,
                                     delta=model.delta, theta=model.theta, rec=rec,
                                     **fwd_fields)
        wrap_rev = CheckpointOperator(op, ph=ph, pv=pv, m=model.m, epsilon=model.epsilon,
                                      delta=model.delta, theta=model.theta, src=rec_g,
                                      **adj_fields)

        # Run forward
        wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)
        wrp.apply_forward()

        # Residual and gradient
        if is_residual is True:  # input data is already the residual
            rec_g.data[:] = rec_data[:]
        else:
            rec_g.data[:] = rec.data[:] - rec_data[:]   # input is observed data
            fval = .5*np.linalg.norm(rec_g.data[:])**2
        wrp.apply_reverse()
    else:
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

def staggered_diff(args, dim, order, stagger=centered, theta=0, phi=0, adjoint=False):
    """
    Utility function to generate staggered derivatives
    """
    func = list(retrieve_functions(args))
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
        fd = custom_FD(args, dim, idx, dim + off[dim]*dim.spacing)
        if adjoint:
            fd = fd.subs({diff: - diff})
    else:
        # rotated FD in y if 3D
        dy = 0
        is_y = False
        x = dims[0]
        z = dims[-1]

        if ndim == 3:
            y = dims[1]
            idxy = list(set([(y + int(i+.5+off[y])*y.spacing)
                             for i in range(-int(order / 2), int(order / 2))]))
            is_y = (dim == y)

            if adjoint:
                if dim == x:
                    ina = sin(phi) * cos(theta) * args
                elif dim == z:
                    ina = sin(phi) * sin(theta) * args
                elif is_y:
                    ina = cos(phi) * args
                else:
                    ina = 0
                dy = custom_FD(ina, y, idxy, y + off[y]*y.spacing)
                dy = dy.subs({y.spacing: -y.spacing})
            else:
                dy = custom_FD(args, y, idxy, y + off[y]*y.spacing)

        # Rotated FD in x
        idxx = list(set([(x + int(i+.5+off[x])*x.spacing)
                         for i in range(-int(order / 2), int(order / 2))]))
        dx = custom_FD(args, x, idxx, x + off[x]*x.spacing)

        if adjoint:
            if dim == x:
                ina = cos(theta) * cos(phi) * args
            elif dim == z:
                ina = sin(theta) * cos(phi) * args
            elif is_y:
                ina = -sin(phi) * args
            else:
                ina = 0
            dx = custom_FD(ina, x, idxx, x + off[x]*x.spacing)
            dx = dx.subs({x.spacing: -x.spacing})
        else:
            dx = custom_FD(args, x, idxx, x + off[x]*x.spacing)

        # Rotated FD in z
        idxz = list(set([(z + int(i+.5+off[z])*z.spacing)
                         for i in range(-int(order / 2), int(order / 2))]))

        if adjoint:
            if dim == x:
                ina = - sin(theta) * args
            elif dim == z:
                ina =  cos(theta) * args
            elif is_y:
                ina = 0
            else:
                ina = 0
            dz = custom_FD(ina, z, idxz, z + off[z]*z.spacing)
            dz = dz.subs({z.spacing: -z.spacing})
        else:
            dz = custom_FD(args, z, idxz, z + off[z]*z.spacing)

        if adjoint:
            if is_y:
                fd = dx + dy
            else:
                fd = dx + dy + dz
        else:
            if dim == x:
                fd = cos(theta) * cos(phi) * dx + sin(phi) * cos(theta) * dy - sin(theta) * dz
            elif dim == z:
                fd = sin(theta) * cos(phi) * dx + sin(phi) * sin(theta) * dy + cos(theta) * dz
            elif is_y:
                fd = -sin(phi) * dx + cos(phi) * dy
            else:
                fd = 0
    return fd

def custom_FD(expr, dim, indx, x0):
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
        var = expr.subs({dim: indx[i]})
        deriv += coeffs[i] * var
    return deriv
