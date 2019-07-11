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
from sympy import cos, sin, expand, symbols
from sympy import Function as fint
from devito.logger import set_log_level
from devito import Eq, Function, TimeFunction, Dimension, Operator, clear_cache, ConditionalDimension, DefaultDimension, Inc, solve
from devito import first_derivative, left, right
from PySource import PointSource, Receiver
from PyModel import Model
from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
from utils import freesurface

def J_adjoint(model, src_coords, wavelet, rec_coords, recin, space_order=12,
                t_sub_factor=20, h_sub_factor=2, checkpointing=False, free_surface=False,
                n_checkpoints=None, maxmem=None, dt=None, isic=False):
    clear_cache()
    if checkpointing:
        F = forward_modeling(model, src_coords, wavelet, rec_coords, save=True, space_order=space_order,
                             t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor, dt=dt, op_return=True,
                             free_surface=free_surface)
        grad = adjoint_born(model, rec_coords, recin, op_forward=F, space_order=space_order,
                            is_residual=True, isic=isic, n_checkpoints=n_checkpoints, maxmem=maxmem,
                            free_surface=free_surface)
    else:
        u0 = forward_modeling(model, src_coords, wavelet, None, save=True, space_order=space_order,
                              t_sub_factor=t_sub_factor, h_sub_factor=h_sub_factor, dt=dt, free_surface=free_surface)
        grad = adjoint_born(model, rec_coords, recin, u=u0, space_order=space_order, isic=isic, dt=dt, free_surface=free_surface)
    
    return grad

def acoustic_laplacian(v, rho):
    if rho is None:
        Lap = v.laplace
        rho = 1
    else:
        if isinstance(rho, Function):
            Lap = sum([first_derivative(first_derivative(v, fd_order=int(v.space_order/2), side=left, dim=d) / rho,
                       fd_order=int(v.space_order/2), dim=d, side=right) for d in v.space_dimensions])
        else:
            Lap = 1 / rho * v.laplace
    return Lap, rho


def forward_modeling(model, src_coords, wavelet, rec_coords, save=False, space_order=8,
                     free_surface=False, op_return=False, dt=None, tsub_factor=1, h_sub_factor=1):
    clear_cache()
    # If wavelet is file, read it
    if isinstance(wavelet, str):
        wavelet = np.load(wavelet)

    # Parameters
    nt = wavelet.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create the forward wavefield
    if save is False and rec_coords is not None:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        eqsave = []
    elif save is True and tsub_factor > 1:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
        time_subsampled = ConditionalDimension(name='t_sub', parent=u.grid.time_dim, factor=tsub_factor)
        nsave = (nt-1)//tsub_factor + 2
        usave = TimeFunction(name='us', grid=model.grid, time_order=2, space_order=space_order, time_dim=time_subsampled, save=nsave)
        eqsave = [Eq(usave.forward, u.forward)]
    else:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order, save=nt)
        eqsave = []

    # Set up PDE
    ulaplace, rho = acoustic_laplacian(u, rho)
    H = symbols('H')
    eqn = m / rho * u.dt2 - H + 2 * damp * u.dt + damp * damp * u
    stencil = solve(eqn, u.forward).subs({H: ulaplace})

    # Input source is wavefield
    if isinstance(wavelet, TimeFunction):
        wf_src = TimeFunction(name='wf_src', grid=model.grid, time_order=2, space_order=space_order, save=nt)
        wf_src._data = wavelet._data
        eqn -= wf_src

    # Rearrange expression
    # stencil = solve(eqn, u.forward)
    expression = [Eq(u.forward, stencil)]

    # Data is sampled at receiver locations
    if rec_coords is not None:
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        rec_term = rec.interpolate(expr=u)
        expression += rec_term

    # Source symbol with input wavelet
    if src_coords is not None:
        src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
        src.data[:] = wavelet[:]
        src_term = src.inject(field=u.forward, expr=src.dt * rho * dt**2 / m)
        expression += src_term

    # Free surface
    if free_surface is True:
        expression += freesurface(u, space_order//2, model.nbpml)

    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt

    op = Operator(expression, subs=subs, dse='advanced', dle='advanced', name='Forward')
    z_m = op.arguments()["y_m"]
    if op_return is False:
        if freesurface:
            z_m += model.nbpml
        op(y_m=z_m)
        if rec_coords is None:
            return u
        else:
           return rec.data, u
    # For optimal checkpointing, return operator only
    else:
        return op


def adjoint_modeling(model, src_coords, rec_coords, rec_data, space_order=8, free_surface=False, dt=None):
    clear_cache()

    # If wavelet is file, read it
    if isinstance(rec_data, str):
        rec_data = np.load(rec_data)

    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create the adjoint wavefield
    if src_coords is not None:
        v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=space_order)
    else:
        v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=space_order, save=nt)

    # Set up PDE and rearrange
    vlaplace, rho = acoustic_laplacian(v, rho)

    # Input data is wavefield
    full_q = 0
    if isinstance(rec_data, TimeFunction):
        wf_rec = TimeFunction(name='wf_rec', grid=model.grid, time_order=2, space_order=space_order, save=nt)
        wf_rec._data = rec_data._data
        full_q = wf_rec

    # Set up PDE and rearrange
    vlaplace, rho = acoustic_laplacian(v, rho)
    H = symbols('H')
    eqn = m / rho * v.dt2 - H + 2 * damp * v.dt.T + damp * damp * v
    stencil = solve(eqn, v.backward).subs({H: vlaplace + full_q})

    expression = [Eq(v.backward, stencil)]

    # Adjoint source is injected at receiver locations
    if rec_coords is not None:
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        rec.data[:] = rec_data[:]
        adj_src = rec.inject(field=v.backward, expr=rec * rho * dt**2 / m)
        expression += adj_src

    # Data is sampled at source locations
    if src_coords is not None:
        src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
        adj_rec = src.interpolate(expr=v.dt.T)
        expression += adj_rec

    # Free surface
    if free_surface is True:
        expression += freesurface(v, space_order//2, model.nbpml, forward=False)

    # Create operator and run

    subs = model.spacing_map
    subs[v.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced', name='Adjoint')
    z_m = op.arguments()["y_m"]
    if freesurface:
        z_m += model.nbpml
    op(y_m=z_m)

    if src_coords is None:
        return v
    else:
        return src.data

def forward_born(model, src_coords, wavelet, rec_coords, space_order=8, isic=False, dt=None, free_surface=False):
    clear_cache()

    # Parameters
    nt = wavelet.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, dm, damp = model.m, model.rho, model.dm, model.damp

    # Create the forward and linearized wavefield
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=space_order)
    du = TimeFunction(name="du", grid=model.grid, time_order=2, space_order=space_order)
    if len(model.shape) == 2:
        x,y = u.space_dimensions
    else:
        x,y,z = u.space_dimensions

    # Set up PDEs and rearrange
    ulaplace, rho = acoustic_laplacian(u, rho)
    dulaplace, _ = acoustic_laplacian(du, rho)

    if isic:
        # Sum ((u.dx * d, / rho).dx for x in dimensions)
        # space_order//2  so that u.dx.dx has the same radius as u.laplace
        du_aux = sum([first_derivative(first_derivative(u, dim=d, fd_order=space_order//2) * dm / rho,
                                       fd_order=space_order//2, dim=d)
                      for d in u.space_dimensions])
        lin_source = dm /rho * u.dt2 * m - du_aux
    else:
        lin_source = dm / rho * u.dt2

    # Set up PDE and rearrange
    H = symbols('H')
    eqn = m / rho * u.dt2 - H + 2 * damp * u.dt + damp * damp * u
    stencil_u = solve(eqn, u.forward).subs({H: ulaplace})

    eqn = m / rho * du.dt2 - H + 2 * damp * du.dt + damp * damp * du
    stencil_du = solve(eqn, du.forward).subs({H: dulaplace - lin_source})

    expression_u = [Eq(u.forward, stencil_u)]
    expression_du = [Eq(du.forward, stencil_du)]

    # Define source symbol with wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, expr=src.dt * rho * dt**2 / m)

    # Define receiver symbol
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=du)

    # Create operator and run

    expression = expression_u + src_term + expression_du + rec_term

    # Free surface
    if free_surface is True:
        expression += freesurface(u, space_order//2, model.nbpml)
        expression += freesurface(du, space_order//2, model.nbpml)

    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced', name='Born')
    z_m = op.arguments()["y_m"]
    if freesurface:
        z_m += model.nbpml
    op(y_m=z_m)
    return rec.data


def adjoint_born(model, rec_coords, rec_data, u=None, op_forward=None, is_residual=False,
                 space_order=8, isic=False, dt=None, n_checkpoints=None, maxmem=None,
                 free_surface=False, tsub_factor=1):
    clear_cache()

    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create adjoint wavefield and gradient
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
    gradient = Function(name='gradient', grid=model.grid)

    # Set up PDE and rearrange
    vlaplace, rho = acoustic_laplacian(v, rho)
    H = symbols('H')
    eqn = m / rho * v.dt2 - H + 2 * damp * v.dt.T + damp * damp * v
    stencil = solve(eqn, v.backward).subs({H: vlaplace})

    expression = [Eq(v.backward, stencil)]
    # Data at receiver locations as adjoint source
    rec_g = Receiver(name='rec_g', grid=model.grid, ntime=nt, coordinates=rec_coords)
    if op_forward is None:
        rec_g.data[:] = rec_data[:]
    adj_src = rec_g.inject(field=v.backward, expr=rec_g * rho * dt**2 / m)

    # Gradient update
    if u is None:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
    if isic is not True:
        gradient_update = [Inc(gradient, - tsub_factor * dt * u.dt2 / rho * v)]
    else:
        # summodel0.dm = dm u.dx * v.dx fo x in dimensions.
        # space_order//2
        diff_u_v = sum([first_derivative(u, dim=d, fd_order=space_order//2)*
                        first_derivative(v, dim=d, fd_order=space_order//2)
                        for d in u.space_dimensions])
        gradient_update = [Inc(gradient, - tsub_factor * dt * (u * v.dt2 * m + diff_u_v) / rho)]

    # Create operator and run

    expression += adj_src + gradient_update

    # Free surface
    if free_surface is True:
        expression += freesurface(v, space_order//2, model.nbpml, forward=False)

    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced', name='Forward')
    z_m = op.arguments()["y_m"]
    if freesurface:
        z_m += model.nbpml
    # Optimal checkpointing
    if op_forward is not None:
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        cp = DevitoCheckpoint([u])
        if maxmem is not None:
            n_checkpoints = int(np.floor(maxmem * 10**6 / (cp.size * u.data.itemsize)))
        wrap_fw = CheckpointOperator(op_forward, u=u, m=model.m, rec=rec)
        wrap_rev = CheckpointOperator(op, u=u, v=v, m=model.m, rec_g=rec_g)

        # Run forward
        wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt-2)
        wrp.apply_forward(y_m=z_m)

        # Residual and gradient
        if is_residual is True:  # input data is already the residual
            rec_g.data[:] = rec_data[:]
        else:
            rec_g.data[:] = rec.data[:] - rec_data[:]   # input is observed data
            fval = .5*np.dot(rec_g.data[:].flatten(), rec_g.data[:].flatten()) * dt
        wrp.apply_reverse(y_m=z_m)
    else:
        op(y_m=z_m)

    if op_forward is not None and is_residual is not True:
        return fval, gradient.data
    else:
        return gradient.data


########################################################################################################################

def forward_freq_modeling(model, src_coords, wavelet, rec_coords, freq, space_order=8, dt=None, factor=None, free_surface=False):
    # Forward modeling with on-the-fly DFT of forward wavefields
    clear_cache()

    # Parameters
    nt = wavelet.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    freq_dim = Dimension(name='freq_dim')
    time = model.grid.time_dim
    if factor is None:
        factor = int(1 / (dt*4*np.max(freq)))
        tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
    if factor==1:
        tsave = time
    else:
        tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
    print("DFT subsampling factor: ", factor)

    # Create wavefields
    nfreq = freq.shape[0]
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
    f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
    f.data[:] = freq[:]
    ufr = Function(name='ufr', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + model.shape_domain)
    ufi = Function(name='ufi', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + model.shape_domain)

    ulaplace, rho = acoustic_laplacian(u, rho)

    # Set up PDE and rearrange
    stencil = damp * (2.0 * u - damp * u.backward + dt**2 * rho / m * ulaplace)
    expression = [Eq(u.forward, stencil)]
    expression += [Inc(ufr, factor*u*cos(2*np.pi*f*tsave*factor*dt))]
    expression += [Inc(ufi, -factor*u*sin(2*np.pi*f*tsave*factor*dt))]

    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, expr=src * dt**2 / m)

    # Data is sampled at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=u)

    # Create operator and run

    expression += src_term + rec_term
    # Free surface
    if free_surface is True:
        expression += freesurface(u, space_order//2, model.nbpml)

    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
    op()

    return rec.data, ufr, ufi


def adjoint_freq_born(model, rec_coords, rec_data, freq, ufr, ufi, space_order=8, dt=None, isic=False, factor=None,
                      free_surface=False):
    clear_cache()

    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp
    nfreq = ufr.shape[0]
    time = model.grid.time_dim
    if factor is None:
        factor = int(1 / (dt*4*np.max(freq)))
        tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
    if factor==1:
        tsave = time
    else:
        tsave = ConditionalDimension(name='tsave', parent=model.grid.time_dim, factor=factor)
    dtf = factor * dt
    ntf = factor / nt
    print("DFT subsampling factor: ", factor)

    # Create the forward and adjoint wavefield
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
    f = Function(name='f', dimensions=(ufr.indices[0],), shape=(nfreq,))
    f.data[:] = freq[:]
    gradient = Function(name="gradient", grid=model.grid)
    vlaplace, rho = acoustic_laplacian(v, rho)

    # Set up PDE and rearrange
    stencil = damp * (2.0 * v - damp * v.forward + dt**2 * rho / m * vlaplace)
    expression = [Eq(v.backward, stencil)]

    # Data at receiver locations as adjoint source
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec.data[:] = rec_data[:]
    adj_src = rec.inject(field=v.backward, expr=rec * dt**2 / m)

    # Gradient update
    if isic is True:
        if len(model.shape) == 2:
            gradient_update = [Eq(gradient, gradient + (2*np.pi*f)**2*ntf*(ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v*model.m -
                                                       (ufr.dx*cos(2*np.pi*f*tsave*dtf) - ufi.dx*sin(2*np.pi*f*tsave*dtf))*v.dx*ntf -
                                                       (ufr.dy*cos(2*np.pi*f*tsave*dtf) - ufi.dy*sin(2*np.pi*f*tsave*dtf))*v.dy*ntf)]
        else:
            gradient_update = [Eq(gradient, gradient + (2*np.pi*f)**2*ntf*(ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v*model.m -
                                                       (ufr.dx*cos(2*np.pi*f*tsave*dtf) - ufi.dx*sin(2*np.pi*f*tsave*dtf))*v.dx*ntf -
                                                       (ufr.dy*cos(2*np.pi*f*tsave*dtf) - ufi.dy*sin(2*np.pi*f*tsave*dtf))*v.dy*ntf -
                                                       (ufr.dz*cos(2*np.pi*f*tsave*dtf) - ufi.dz*sin(2*np.pi*f*tsave*dtf))*v.dz*ntf)]
    else:
        gradient_update = [Eq(gradient, gradient + (2*np.pi*f)**2/nt*(ufr*cos(2*np.pi*f*tsave*dtf) - ufi*sin(2*np.pi*f*tsave*dtf))*v)]

    # Create operator and run

    # Free surface
    if free_surface is True:
        expression += freesurface(v, space_order//2, model.nbpml, forward=False)
    expression += adj_src + gradient_update
    subs = model.spacing_map
    subs[v.grid.time_dim.spacing] = dt
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced')
    op()
    clear_cache()
    return gradient.data
