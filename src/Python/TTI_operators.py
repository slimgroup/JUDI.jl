# Acoustic wave equations with Devito
# Forward/adjoint nonlinear and Born modeling
# Authors: Mathias Louboutin, Philipp Witte
# Date: November 2017
#

# Import modules
from __future__ import print_function
import numpy as np
import gc, os, psutil
from numpy.random import randint
from sympy import solve, cos, sin, expand, symbols
from sympy import Function as fint
from devito.logger import set_log_level
from devito import Eq, Function, TimeFunction, Dimension, Operator, clear_cache, ConditionalDimension, Grid, Inc, DefaultDimension
from devito.finite_differences import (centered, first_derivative, right, transpose,
                                      second_derivative, left)
from devito.symbolics import retrieve_functions

from devito.symbolics import retrieve_functions
from PySource import PointSource, Receiver
from PyModel import Model
from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
from utils import freesurface

def J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=12, nb=40,
                t_sub_factor=20, h_sub_factor=2, checkpointing=False, free_surface=False,
                n_checkpoints=None, maxmem=None, dt=None, isic=False):
    if checkpointing:
        F = forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=space_order, nb=nb,
                             t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor, dt=dt, op_return=True,
                             free_surface=free_surface)
        grad = adjoint_born(model, rec_coords, recin, op_forward=F, space_order=space_order,
                            nb=nb, is_residual=True, isic=isic, n_checkpoints=n_checkpoints, maxmem=maxmem,
                            free_surface=free_surface)
    else:
        u0, v0 = forward_modeling(model, src_coords, wavelet, None, save=True, space_order=space_order, nb=nb,
                                  t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor, dt=dt, free_surface=free_surface)
        grad = adjoint_born(model, rec_coords, recin, u=u0, space_order=space_order, nb=nb, isic=isic, dt=dt, free_surface=free_surface,
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


def forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=16, nb=40,
                     t_sub_factor=1, h_sub_factor=1, op_return=False, dt=None, free_surface=False):
    clear_cache()

    # Parameters
    nt = wavelet.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, damp, epsilon, delta, theta, phi = (model.m, model.damp, model.epsilon,
                                           model.delta, model.theta, model.phi)

    # Tilt and azymuth setup
    ang0 = cos(theta)
    ang1 = sin(theta)
    ang2 = 1
    ang3 = 0
    if len(model.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)

    # Create the forward wavefield
    if save and (t_sub_factor>1 or h_sub_factor>1):
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
        if t_sub_factor > 1:
            time_subsampled = ConditionalDimension('t_sub', parent=u.grid.time_dim, factor=t_sub_factor)
            nsave = (nt-1)//t_sub_factor + 2
        else:
            time_subsampled = grid.time_dim
            nsave = nt

        if h_sub_factor > 1:
            grid2 = Grid(shape=tuple([i//2 for i in u.data.shape[1:]]),
                         extent=u.grid.extent, dimensions=sub_ind(model))
        else:
            grid2 = model.grid

        usave = TimeFunction(name='us', grid=grid2, time_order=2, space_order=space_order,
                             time_dim=time_subsampled, save=nsave)
        vsave = TimeFunction(name='vs', grid=grid2, time_order=2, space_order=space_order,
                             time_dim=time_subsampled, save=nsave)
        eqsave = [Eq(usave.forward, u.forward), Eq(vsave.forward, v.forward)]
    elif save and t_sub_factor==1 and h_sub_factor==1:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order,
                         save=nt)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order,
                         save=nt)
        eqsave = []
    else:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
        eqsave = []
    # TTI stencil
    rho = model.rho
    if isinstance(rho, Function):
        FD_kernel = kernels['rho']
        H0, Hz = FD_kernel(u, v, rho, ang0, ang1, ang2, ang3, space_order)
    else:
        FD_kernel = kernels[len(model.shape)]
        H0, Hz = FD_kernel(u, v, ang0, ang1, ang2, ang3, space_order)
        H0 = H0
        Hz = Hz

    # Stencils
    s = model.grid.stepping_dim.spacing
    stencilp = damp * ( 2 * u - damp * u.backward + s**2 / m * (epsilon * H0 + delta * Hz))
    stencilr = damp * ( 2 * v - damp * v.backward + s**2 / m * (delta * H0 + Hz))
    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    expression = [first_stencil, second_stencil]

    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, expr=src * rho * dt**2 / m)
    src_term += src.inject(field=v.forward, expr=src * rho * dt**2 / m)

    # Data is sampled at receiver locations
    if rec_coords is not None:
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        expression += rec.interpolate(expr=u + v)

    # Create operator and run
    set_log_level('INFO')
    expression += src_term
    if save:
        expression += eqsave

    if free_surface:
        expression += freesurface(u, space_order//2, model.nbpml)
        expression += freesurface(v, space_order//2, model.nbpml)

    op = Operator(expression, subs=model.spacing_map, dse='aggressive', dle='advanced')
    # from IPython import embed; embed()
    if op_return is False:
        op()
        if save and (t_sub_factor>1 or h_sub_factor>1):
            if rec_coords is not None:
                return rec.data, usave, vsave
            else:
                return usave, vsave
        else:
            if rec_coords is not None:
                return rec.data, u, v
            else:
                return u, v
    else:
        return op


def adjoint_modeling(model, src_coords, rec_coords, rec_data, space_order=12, nb=40, dt=None, free_surface=False):
    clear_cache()

    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, damp, epsilon, delta, theta, phi = (model.m, model.damp, model.epsilon,
                                           model.delta, model.theta, model.phi)

    # Tilt and azymuth setup
    ang0 = cos(theta)
    ang1 = sin(theta)
    ang2 = 1
    ang3 = 0
    if len(model.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)

    # Create the adjoint wavefield
    p = TimeFunction(name="p", grid=model.grid, time_order=2, space_order=space_order)
    q = TimeFunction(name="q", grid=model.grid, time_order=2, space_order=space_order)

    rho = model.rho
    if isinstance(rho, Function):
        FD_kernel = kernels['rho']
        H0, Hz = FD_kernel(epsilon * p + delta * q, delta * p + q,
                           rho, ang0, ang1, ang2, ang3, space_order)
    else:
        FD_kernel = kernels[len(model.shape)]
        H0, Hz = FD_kernel(epsilon * p + delta * q,
                           delta * p + q, ang0, ang1, ang2, ang3, space_order)
        H0 = H0 / rho
        Hz = Hz / rho

    # Stencils
    s = model.grid.stepping_dim.spacing
    stencilp = damp * (2 * p - damp * p.forward + s**2 / m * H0)
    stencilr = damp * (2 * q - damp * q.forward + s**2 / m * Hz)
    first_stencil = Eq(p.backward, stencilp)
    second_stencil = Eq(q.backward, stencilr)
    expression = [first_stencil, second_stencil]

    # Adjoint source is injected at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec.data[:] = rec_data[:]
    adj_src = rec.inject(field=p.backward, expr=rec * dt**2 / m)
    adj_src += rec.inject(field=q.backward, expr=rec * dt**2 / m)
    # Data is sampled at source locations
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    adj_rec = src.interpolate(expr=p+q)

    # Create operator and run
    set_log_level('INFO')
    expression += adj_src + adj_rec

    if free_surface:
        expression += freesurface(p, space_order//2, model.nbpml, forward=False)
        expression += freesurface(q, space_order//2, model.nbpml, forward=False)

    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced')
    op()

    return src.data


def forward_born(model, src_coords, wavelet, rec_coords, space_order=12, nb=40, isic=False, dt=None, save=False, isiciso=False,
                 h_sub_factor=1, free_surface=False):
    clear_cache()

    # Parameters
    nt = wavelet.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, damp, epsilon, delta, theta, phi, dm = (model.m, model.damp, model.epsilon,
                                               model.delta, model.theta, model.phi,
                                               model.dm)

    # Tilt and azymuth setup
    ang0 = cos(theta)
    ang1 = sin(theta)
    ang2 = 1
    ang3 = 0
    if len(model.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)

    # Create the forward and linearized wavefield
    u = TimeFunction(name='u', grid=model.grid,
                     time_order=2, space_order=space_order)
    v = TimeFunction(name='v', grid=model.grid,
                     time_order=2, space_order=space_order)
    ul = TimeFunction(name='ul', grid=model.grid,
                     time_order=2, space_order=space_order)
    vl = TimeFunction(name='vl', grid=model.grid,
                     time_order=2, space_order=space_order)

    FD_kernel = kernels[len(model.shape)]
    H0, Hz = FD_kernel(u, v, ang0, ang1, ang2, ang3, space_order)
    H0l, Hzl = FD_kernel(ul, vl, ang0, ang1, ang2, ang3, space_order)

    # Stencils
    s = model.grid.stepping_dim.spacing
    stencilp = damp * (2 * u - damp * u.backward + s**2 / m * (epsilon * H0 + delta * Hz))
    stencilr = damp * (2 * v - damp * v.backward + s**2 / m * (delta * H0 + Hz))

    if isic:
    #     du_aux_x = first_derivative(u.dx * dm, fd_order=space_order, dim=u.indices[1], diff=h_sub_factor*u.indices[1].spacing)
    #     du_aux_y = first_derivative(u.dy * dm, fd_order=space_order, dim=u.indices[2], diff=h_sub_factor*u.indices[2].spacing)
    #     du2 =  (dm * u.dt2 * m - du_aux_x - du_aux_y)
    #     if len(model.shape) == 3:
    #         du2 -= first_derivative(u.dz * dm, fd_order=space_order, dim=u.indices[3], diff=h_sub_factor*u.indices[3].spacing)
    #
    #     dv_aux_x = first_derivative(v.dx * dm, fd_order=space_order, dim=u.indices[1], diff=h_sub_factor*u.indices[1].spacing)
    #     dv_aux_y = first_derivative(v.dy * dm, fd_order=space_order, dim=u.indices[2], diff=h_sub_factor*u.indices[2].spacing)
    #     dv2 =  (dm * v.dt2 * m - dv_aux_x - dv_aux_y)
    #     if len(model.shape) == 3:
    #         dv2 -= first_derivative(v.dz * dm, fd_order=space_order, dim=u.indices[3], diff=h_sub_factor*u.indices[3].spacing)
    #
    #     stencilpl = damp * (2 * ul - damp * ul.backward + s**2 / m * (epsilon * H0l + delta * Hzl - du2))
    #     stencilrl = damp * (2 * vl - damp * vl.backward + s**2 / m * (delta * H0l + Hzl - dv2))
    # elif isic:
        order_loc = int(space_order/2)
        lin_expru = dm * u.dt2 * m - Dx(Dx(u, ang0, ang1, ang2, ang3, order_loc) * dm,
                                                  ang0, ang1, ang2, ang3, order_loc)
        lin_expru -= Dz(Dz(u, ang0, ang1, ang2, ang3, order_loc) * dm,
                                   ang0, ang1, ang2, ang3, order_loc)
        lin_exprv = dm * v.dt2 * m - Dx(Dx(v, ang0, ang1, ang2, ang3, order_loc) * dm,
                                                ang0, ang1, ang2, ang3, order_loc)
        lin_exprv -= Dz(Dz(v, ang0, ang1, ang2, ang3, order_loc) * dm,
                        ang0, ang1, ang2, ang3, order_loc)
        if len(model.shape) == 3:
            lin_expru -= Dy(Dy(u, ang0, ang1, ang2, ang3, order_loc) * dm,
                                        ang0, ang1, ang2, ang3, order_loc)
            lin_exprv -= Dy(Dy(v, ang0, ang1, ang2, ang3, order_loc) * dm,
                                    ang0, ang1, ang2, ang3, order_loc)
        stencilpl = damp * (2 * ul - damp * ul.backward + s**2 / m * (epsilon * H0l + delta * Hzl - lin_expru))
        stencilrl = damp * (2 * vl - damp * vl.backward + s**2 / m * (delta * H0l + Hzl - lin_exprv))
    else:
        stencilpl = damp * (2 * ul - damp * ul.backward + s**2 / m * (epsilon * H0l + delta * Hzl - dm * u.dt2))
        stencilrl = damp * (2 * vl - damp * vl.backward + s**2 / m * (delta * H0l + Hzl - dm * v.dt2))

    first_stencil = Eq(u.forward, stencilp)
    second_stencil = Eq(v.forward, stencilr)
    first_stencill = Eq(ul.forward, stencilpl)
    second_stencill = Eq(vl.forward, stencilrl)
    expression_u = [first_stencil, second_stencil]
    expression_du = [first_stencill, second_stencill]
    # Define source symbol with wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
    src_term += src.inject(field=v.forward, expr=src * dt**2 / m)
    # Define receiver symbol
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=ul + vl)

    # Create operator and run
    set_log_level('INFO')
    expression = expression_u + expression_du + src_term

    if free_surface:
        expression += freesurface(u, space_order//2, model.nbpml)
        expression += freesurface(v, space_order//2, model.nbpml)
        expression += freesurface(ul, space_order//2, model.nbpml)
        expression += freesurface(vl, space_order//2, model.nbpml)

    expression += rec_term
    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced')
    op()

    return rec.data


def adjoint_born(model, rec_coords, rec_data, u=None, v=None, op_forward=None, is_residual=False,
                 space_order=12, nb=40, isic=False, dt=None, n_checkpoints=None, maxmem=None,
                 free_surface=False, t_sub_factor=1, h_sub_factor=1):
    clear_cache()
    factor_t = t_sub_factor
    factor_h = h_sub_factor
    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, damp, epsilon, delta, theta, phi = (model.m, model.damp, model.epsilon,
                                           model.delta, model.theta, model.phi)

    # Tilt and azymuth setup
    ang0 = cos(theta)
    ang1 = sin(theta)
    ang2 = 1
    ang3 = 0
    if len(model.shape) == 3:
        ang2 = cos(phi)
        ang3 = sin(phi)

    # Create adjoint wavefield and gradient
    p = TimeFunction(name='p', grid=model.grid, time_order=2, space_order=space_order)
    q = TimeFunction(name='q', grid=model.grid, time_order=2, space_order=space_order)
    gradient = Function(name='gradient', grid=model.grid)

    FD_kernel = kernels[len(model.shape)]
    # Create temps adjoint
    # temp_p = Function(name="temp_p", grid=model.grid, space_order=p.space_order)
    # temp_r = Function(name="temp_r", grid=model.grid, space_order=q.space_order)
    H0, Hz = FD_kernel(epsilon * p + delta * q, delta * p + q,
                                ang0, ang1, ang2, ang3, space_order)

    # H0, Hz = FD_kernel(temp_p, temp_r, ang0, ang1, ang2, ang3, space_order)
    # Stencils
    s = model.grid.stepping_dim.spacing
    stencilp = damp * (2 * p - damp * p.forward + s**2 / m * H0)
    stencilr = damp * (2 * q - damp * q.forward + s**2 / m * Hz)
    first_stencil = Eq(p.backward, stencilp)
    second_stencil = Eq(q.backward, stencilr)
    expression = [first_stencil, second_stencil]
    # expression += [Eq(temp_p, epsilon * p.backward + delta * q.backward), Eq(temp_r, delta * p.backward + q.backward)]
    # Data at receiver locations as adjoint source
    rec_g = Receiver(name='rec_g', grid=model.grid, ntime=nt, coordinates=rec_coords)
    if op_forward is None:
        max_in = np.max(rec_data[:])
        # rec_data = np.gradient(rec_data)[0]
        rec_g.data[:] = rec_data[:] #max_in * rec_data[:] / np.max(rec_data[:])

    adj_src = rec_g.inject(field=p.backward, expr=rec_g * dt**2 / m)
    adj_src += rec_g.inject(field=q.backward, expr=rec_g * dt**2 / m)
    # Gradient update
    if u is None:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
    if v is None:
        v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)

    if isic is True:
        # order_loc = int(space_order/2)
        # udx = epsilon * Dx(u, ang0, ang1, ang2, ang3, order_loc)
        # pdx = Dx(p, ang0, ang1, ang2, ang3, order_loc)
        # udz = delta * Dz(u, ang0, ang1, ang2, ang3, order_loc)
        # pdz = Dz(p, ang0, ang1, ang2, ang3, order_loc)
        # vdx = delta * Dx(v, ang0, ang1, ang2, ang3, order_loc)
        # qdx = Dx(q, ang0, ang1, ang2, ang3, order_loc)
        # vdz = Dz(v, ang0, ang1, ang2, ang3, order_loc)
        # qdz = Dz(q, ang0, ang1, ang2, ang3, order_loc)
        # grads = vdz * qdz + udz * pdz + udx * pdx + vdx * qdx
        # if len(model.shape) == 3:
        #     udy = epsilon * Dy(u, ang0, ang1, ang2, ang3, order_loc)
        #     pdy = Dy(p, ang0, ang1, ang2, ang3, order_loc)
        #     vdy = delta * Dy(v, ang0, ang1, ang2, ang3, order_loc)
        #     qdy = Dy(q, ang0, ang1, ang2, ang3, order_loc)
        #     grads += udy * pdy + vdy * qdy
        # gradient_update = [Inc(gradient, - factor_h * factor_t * dt * ((u * p.dt2 + v * q.dt2) * m + grads))]
    # elif isiciso is True:
        grads = u.dx * p.dx + u.dy * p.dy + v.dx * q.dx + v.dy * q.dy
        if len(model.shape) == 3:
            grads += u.dz * p.dz + v.dz * q.dz
        gradient_update = [Inc(gradient, - factor_h * dt * factor_t * ((u * p.dt2 + v * q.dt2) * m + grads))]
    # elif isicnothom is True:
    #             order_loc = int(space_order/2)
    #             udx = Dx(u, ang0, ang1, ang2, ang3, order_loc)
    #             pdx = Dx(p, ang0, ang1, ang2, ang3, order_loc)
    #             udz = Dz(u, ang0, ang1, ang2, ang3, order_loc)
    #             pdz = Dz(p, ang0, ang1, ang2, ang3, order_loc)
    #             vdx = Dx(v, ang0, ang1, ang2, ang3, order_loc)
    #             qdx = Dx(q, ang0, ang1, ang2, ang3, order_loc)
    #             vdz = Dz(v, ang0, ang1, ang2, ang3, order_loc)
    #             qdz = Dz(q, ang0, ang1, ang2, ang3, order_loc)
    #             grads = vdz * qdz + udz * pdz + udx * pdx + vdx * qdx
    #             if len(model.shape) == 3:
    #                 udy = epsilon * Dy(u, ang0, ang1, ang2, ang3, order_loc)
    #                 pdy = Dy(p, ang0, ang1, ang2, ang3, order_loc)
    #                 vdy = delta * Dy(v, ang0, ang1, ang2, ang3, order_loc)
    #                 qdy = Dy(q, ang0, ang1, ang2, ang3, order_loc)
    #                 grads += udy * pdy + vdy * qdy
    #             gradient_update = [Inc(gradient, - factor_h * factor_t * dt * ((u * p.dt2 + v * q.dt2) * m + grads))]
    else:
        gradient_update = [Inc(gradient, - factor_h * factor_t * dt * (u * p.dt2 + v * q.dt2))]

    # Create operator and run
    set_log_level('INFO')
    expression += adj_src + gradient_update

    if free_surface:
        expression += freesurface(p, space_order//2, model.nbpml, forward=False)
        expression += freesurface(q, space_order//2, model.nbpml, forward=False)

    op = Operator(expression, subs=model.spacing_map, dse='aggressive', dle='speculative')
    # Optimal checkpointing

    if op_forward is not None:
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        cp = DevitoCheckpoint([u, v])
        if maxmem is not None:
            n_checkpoints = int(np.floor(maxmem * 10**6 / (cp.size * u.data.itemsize)))
        wrap_fw = CheckpointOperator(op_forward, u=u, v=v, m=model.m, epsilon=model.epsilon,
                                     delta=model.delta, theta=model.theta, rec=rec)
        wrap_rev = CheckpointOperator(op, u=u, v=v, p=p, q=q, m=model.m, epsilon=model.epsilon,
                                      delta=model.delta, theta=model.theta, rec_g=rec_g)

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
    clear_cache()

    if u.indices[1].is_Conditional and u.indices[1].factor>1:
        grad = resample_grad(gradient, model, u.indices[1].factor)
    else:
        grad = gradient.data
    if op_forward is not None and is_residual is not True:
        return fval, grad
    else:
        return grad
#
# def adjoint_born_fake(model, rec_coords, rec_data, u=None, v=None, op_forward=None, is_residual=False, space_order=12, nb=40, isic=False, dt=None, isiciso=False):
#     clear_cache()
#
#     # Parameters
#     nt = rec_data.shape[0]
#     if dt is None:
#         dt = model.critical_dt
#     m, damp, epsilon, delta, theta, phi = (model.m, model.damp, model.epsilon,
#                                            model.delta, model.theta, model.phi)
#
#     # Tilt and azymuth setup
#     ang0 = cos(theta)
#     ang1 = sin(theta)
#     ang2 = 1
#     ang3 = 0
#     if len(model.shape) == 3:
#         ang2 = cos(phi)
#         ang3 = sin(phi)
#
#     # Create adjoint wavefield and gradient
#     p = TimeFunction(name='p', grid=model.grid, time_order=2, space_order=space_order)
#     q = TimeFunction(name='q', grid=model.grid, time_order=2, space_order=space_order)
#     gradient = Function(name='gradient', grid=model.grid)
#
#     FD_kernel = kernels[len(model.shape)]
#     H0, Hz = FD_kernel(p, q, ang0, ang1, ang2, ang3, space_order)
#
#     # Stencils
#     s = model.grid.stepping_dim.spacing
#     stencilp = damp * 2 * p - damp **2 * p.forward + s**2 / m * (epsilon * H0 + delta * Hz)
#     stencilr = damp * 2 * q - damp **2 * q.forward + s**2 / m * (delta * H0 +  Hz)
#     first_stencil = Eq(p.backward, stencilp)
#     second_stencil = Eq(q.backward, stencilr)
#     expression = [first_stencil, second_stencil]
#
#
#     # Data at receiver locations as adjoint source
#     rec_g = Receiver(name='rec_g', grid=model.grid, ntime=nt, coordinates=rec_coords)
#     if op_forward is None:
#         rec_g.data[:] = rec_data[:]
#     adj_src = rec_g.inject(field=p.backward, expr=rec_g * dt**2 / m)
#     adj_src += rec_g.inject(field=q.backward, expr=rec_g * dt**2 / m)
#     # Gradient update
#     if u is None:
#         u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
#     if v is None:
#         v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
#
#     if isiciso is True:
#         if len(model.shape) == 2:
#             gradient_update = [Inc(gradient, - dt * ((u.dt2 * p + v.dt2 * q) * m +
#                                                               u.dx * p.dx + u.dy * p.dy +
#                                                               v.dx * q.dx + v.dy * q.dy))]
#         else:
#             gradient_update = [Inc(gradient, - dt * ((u.dt2 * p + v.dt2 * q) * m +
#                                                               u.dx * p.dx + u.dy * p.dy + u.dz * p.dz +
#                                                               v.dx * q.dx + v.dy * q.dy + v.dz * q.dz))]
#     else:
#         gradient_update = [Inc(gradient, - dt * u.dt2 * p - dt * v.dt2 * q)]
#     # Create operator and run
#     set_log_level('INFO')
#     expression += adj_src + gradient_update
#     op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced',
#                   name="Gradient%s" % randint(1e5))
#
#     # Optimal checkpointing
#     if op_forward is not None:
#         rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
#         cp = DevitoCheckpoint([u, v])
#         n_checkpoints = None
#         wrap_fw = CheckpointOperator(op_forward, u=u, v=v, m=model.m.data, rec=rec, dt=dt)
#         wrap_rev = CheckpointOperator(op, u=u, v=v, p=p, q=q, m=model.m.data, rec_g=rec_g, dt=dt)
#
#         # Run forward
#         wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)
#         wrp.apply_forward()
#
#         # Residual and gradient
#         if is_residual is True:  # input data is already the residual
#             rec_g.data[:] = rec_data[:]
#         else:
#             rec_g.data[:] = rec.data[:] - rec_data[:]   # input is observed data
#             fval = .5*np.linalg.norm(rec_g.data[:])**2
#         wrp.apply_reverse()
#     else:
#         op(dt=dt)
#     clear_cache()
#
#     if op_forward is not None and is_residual is not True:
#         return fval, gradient.data
#     else:
#         return gradient.data


def Gzz_centered(field, costheta, sintheta, cosphi, sinphi, space_order):
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
    order1 = space_order / 2
    func = list(retrieve_functions(field))[0]
    x, y, z = func.space_dimensions
    Gz = -(sintheta * cosphi * first_derivative(field, dim=x,
                                                side=centered, fd_order=order1) +
           sintheta * sinphi * first_derivative(field, dim=y,
                                                side=centered, fd_order=order1) +
           costheta * first_derivative(field, dim=z,
                                       side=centered, fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta * cosphi,
                            dim=x, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * sintheta * sinphi,
                            dim=y, side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta,
                            dim=z, side=centered, fd_order=order1,
                            matvec=transpose))
    return Gzz


def Gzz_centered_2d(field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    order1 = space_order / 2
    func = list(retrieve_functions(field))[0]
    x, y = func.space_dimensions
    Gz = -(sintheta * first_derivative(field, dim=x, side=centered, fd_order=order1) +
           costheta * first_derivative(field, dim=y, side=centered, fd_order=order1))
    Gzz = (first_derivative(Gz * sintheta, dim=x,
                            side=centered, fd_order=order1,
                            matvec=transpose) +
           first_derivative(Gz * costheta, dim=y,
                            side=centered, fd_order=order1,
                            matvec=transpose))

    return Gzz


# Centered case produces directly Gxx + Gyy
def Gxxyy_centered(field, costheta, sintheta, cosphi, sinphi, space_order):
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
    func = list(retrieve_functions(field))[0]
    lap = sum([second_derivative(field, dim=d, fd_order=space_order) for d in func.space_dimensions])
    Gzz = Gzz_centered(field, costheta, sintheta, cosphi, sinphi, space_order)
    return lap - Gzz


def Gxx_centered_2d(field, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction x.
    As the Laplacian is rotation invariant, it is computed as the conventional
    Laplacian minus the second order rotated second order derivative in the direction z
    Gxx = field.laplace - Gzz
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: Sum of the 3D rotated second order derivative in the direction x
    """
    func = list(retrieve_functions(field))[0]

    lap = sum([second_derivative(field, dim=d, fd_order=space_order) for d in func.space_dimensions])
    return lap - Gzz_centered_2d(field, costheta, sintheta, space_order)

def kernel_centered_2d(u, v, costheta, sintheta, cosphi, sinphi, space_order):
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
    Gxx = Gxx_centered_2d(u, costheta, sintheta, space_order)
    Gzz = Gzz_centered_2d(v, costheta, sintheta, space_order)
    return Gxx, Gzz

def kernel_centered_3d(u, v, costheta, sintheta, cosphi, sinphi, space_order):
    """
    TTI finite difference kernel. The equation we solve is:

    u.dt2 = (1+2 *epsilon) (Gxx(u)+Gyy(u)) + sqrt(1+ 2*delta) Gzz(v)
    v.dt2 = sqrt(1+ 2*delta) (Gxx(u)+Gyy(u)) +  Gzz(v)

    where epsilon and delta are the thomsen parameters. This function computes
    H0 = Gxx(u) + Gyy(u)
    Hz = Gzz(v)

    :param u: first TTI field
    :param v: second TTI field
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: u and v component of the rotated Laplacian in 2D
    """
    Gxx = Gxxyy_centered(u, costheta, sintheta, cosphi, sinphi, space_order)
    Gzz = Gzz_centered(v, costheta, sintheta, cosphi, sinphi, space_order)
    return Gxx, Gzz

def Dx(field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    Rotated first derivative in x
    :param u: TTI field
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: du/dx in rotated coordinates
    """
    order1 = space_order
    func = list(retrieve_functions(field))
    for i in func:
        if isinstance(i, Function):
            dims = i.space_dimensions
            break
    Dx = (costheta * cosphi * first_derivative(field, dim=dims[0], side=centered, fd_order=order1) -
          sintheta * first_derivative(field, dim=dims[-1], side=centered, fd_order=order1))

    if len(dims) == 3:
        Dx += costheta * sinphi * first_derivative(field, dim=dims[1], side=centered, fd_order=order1)
    return Dx


def Dy(field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    Rotated first derivative in y
    :param u: TTI field
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: du/dy in rotated coordinates
    """
    order1 = space_order
    func = list(retrieve_functions(field))
    for i in func:
        if isinstance(i, Function):
            dims = i.space_dimensions
            break
    Dy = (-sinphi * first_derivative(field, dim=dims[0], side=centered, fd_order=order1) +
          cosphi * first_derivative(field, dim=dims[1],side=centered, fd_order=order1))

    return Dy


def Dz(field, costheta, sintheta, cosphi, sinphi, space_order):
    """
    Rotated first derivative in z
    :param u: TI field
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param cosphi: cosine of the azymuth angle
    :param sinphi: sine of the azymuth angle
    :param space_order: discretization order
    :return: du/dz in rotated coordinates
    """
    order1 = space_order
    func = list(retrieve_functions(field))
    for i in func:
        if isinstance(i, Function):
            dims = i.space_dimensions
            break
    Dz = (sintheta * cosphi * first_derivative(field, dim=dims[0], side=centered, fd_order=order1) +
          costheta * first_derivative(field, dim=dims[-1], side=centered, fd_order=order1))

    if len(dims) == 3:
        Dz += sintheta * sinphi * first_derivative(field, dim=dims[1], side=centered, fd_order=order1)
    return Dz


def kernel_rho_2d(u, v, rho, costheta, sintheta, cosphi, sinphi, space_order):
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
    Gxx = Gxx_rho_2d(u, rho, costheta, sintheta, space_order//2)
    Gzz = Gzz_rho_2d(v, rho, costheta, sintheta, space_order//2)
    return Gxx, Gzz

def Gzz_rho_2d(field, rho, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction z as an average of
    two non-centered rotated second order derivative in the direction z
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt
    :param sintheta:  sine of the tilt
    :param space_order: discretization order
    :return: rotated second order derivative wrt z
    """
    # Dx(1/rho Dx u) = 1/rho DxDx - Dx(1/rho) . Dx(u)
    dzu = Dz(field, costheta, sintheta, 1, 0, space_order)
    dzrho = -1/rho*Dz(rho, costheta, sintheta, 1, 0, space_order)
    dzz = Gzz_centered_2d(field, costheta, sintheta, 2*space_order)

    return dzz - dzu * dzrho


def Gxx_rho_2d(field, rho, costheta, sintheta, space_order):
    """
    2D rotated second order derivative in the direction x as an average of
    two non-centered rotated second order derivative in the direction x
    :param field: symbolic data whose derivative we are computing
    :param costheta: cosine of the tilt angle
    :param sintheta:  sine of the tilt angle
    :param space_order: discretization order
    :return: rotated second order derivative wrt x
    """
    # Dx(1/rho Dx u) = 1/rho DxDx - Dx(1/rho) . Dx(u)
    dxu = Dx(field, costheta, sintheta, 1, 0, space_order)
    dxrho = -1/rho*Dx(rho, costheta, sintheta, 1, 0, space_order)
    dxx = Gxx_centered_2d(field, costheta, sintheta, 2*space_order)

    return dxx - dxu * dxrho

kernels = {3: kernel_centered_3d, 2: kernel_centered_2d,
           'rho': kernel_rho_2d}

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
