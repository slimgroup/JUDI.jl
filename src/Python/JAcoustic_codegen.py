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
from devito import Eq, Function, TimeFunction, Dimension, Operator, clear_cache, ConditionalDimension, Constant, DefaultDimension, Inc
from devito import first_derivative, second_derivative
from devito import first_derivative, left, right
from PySource import PointSource, Receiver
from PyModel import Model
from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver

def acoustic_laplacian(v, rho):
    if rho is None:
        Lap = v.laplace
        rho = 1
    else:
        if isinstance(rho, Function):
            Lap = sum([first_derivative(first_derivative(v, order=int(v.space_order/2), side=left, dim=d) / rho,
                       order=int(v.space_order/2), dim=d, side=right) for d in v.space_dimensions])
        else:
            Lap = 1 / rho * v.laplace
    return Lap, rho

def forward_modeling(model, src_coords, wavelet, rec_coords, save=False,
                    space_order=8, nb=40, op_return=False, dt=None, freesurface=False):
    clear_cache()

    # Parameters
    nt = wavelet.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp
    # Create the forward wavefield
    if save is False:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
    else:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order, save=nt)

    # Set up PDE and rearrange
    ulaplace, rho = acoustic_laplacian(u, rho)
    s = model.grid.time_dim.spacing
    stencil = damp * 2.0 * u - damp**2 * u.backward + damp * s**2 / m * ulaplace
    expression = [Eq(u.forward, stencil)]

    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, offset=model.nbpml, expr=src * rho * dt**2 / m)

    # Data is sampled at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=u, offset=model.nbpml)

    # Create operator and run
    set_log_level('ERROR')
    expression += src_term + rec_term
    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    if freesurface:
        fs = DefaultDimension(name="fs", default_value=model.nbpml)
        expression += [Eq(u.forward.subs({u.indices[-1]: model.nbpml - fs - 1}),
                          -u.forward.subs({u.indices[-1]: model.nbpml + fs + 1}))]
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced',
                  name="Forward%s" % randint(1e5))

    if op_return is False:
        op()
        return rec.data, u
    else:
        return op


def adjoint_modeling(model, src_coords, rec_coords, rec_data, space_order=8, nb=40, dt=None, freesurface=False):
    clear_cache()

    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create the adjoint wavefield
    v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=space_order)

    # Set up PDE and rearrange
    vlaplace, rho = acoustic_laplacian(v, rho)
    s = model.grid.time_dim.spacing
    stencil = damp * 2.0 * v - damp**2 * v.forward + damp * s**2 / m * vlaplace
    expression = [Eq(v.backward, stencil)]
    # Adjoint source is injected at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec.data[:] = rec_data[:]
    adj_src = rec.inject(field=v.backward, offset=model.nbpml, expr=rec * rho * dt**2 / m)

    # Data is sampled at source locations
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    adj_rec = src.interpolate(expr=v, offset=model.nbpml)

    # Create operator and run
    set_log_level('ERROR')
    expression += adj_src + adj_rec
    subs = model.spacing_map
    subs[v.grid.time_dim.spacing] = dt
    if freesurface:
        fs = DefaultDimension(name="fs", default_value=int(space_order/2))
        expression += [Eq(v.backward.subs({u.indices[-1]: model.nbpml - fs - 1}),
                          -v.backward.subs({u.indices[-1]: model.nbpml + fs + 1}))]
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced',
                  name="Backward%s" % randint(1e5))
    op()

    return src.data


def forward_born(model, src_coords, wavelet, rec_coords, space_order=8, nb=40, isic=False, dt=None, freesurface=False):
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
    S = symbols('S')
    s = model.grid.time_dim.spacing
    stencil_u = damp * 2.0 * u - damp**2 * u.backward + damp * s**2 / m * ulaplace
    stencil_du = damp * 2.0 * du - damp**2 * du.backward + damp * s**2 / m * (dulaplace - S)
    if isic:
        # Sum ((u.dx * d, / rho).dx for x in dimensions)
        # space_order//2  so that u.dx.dx has the same radius as u.laplace
        du_aux = sum([first_derivative(first_derivative(u, dim=d, order=space_order//2) * dm / rho,
                                       order=space_order//2, dim=d)
                      for d in u.space_dimensions])
        lin_source = dm /rho * u.dt2 * m - du_aux
    else:
        lin_source = dm / rho * u.dt2

    expression_u = [Eq(u.forward, stencil_u)]
    expression_du = [Eq(du.forward, stencil_du.subs({S: lin_source}))]

    # Define source symbol with wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, offset=model.nbpml, expr=src * rho * dt**2 / m)

    # Define receiver symbol
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=du, offset=model.nbpml)

    # Create operator and run
    set_log_level('ERROR')
    expression = expression_u + src_term + expression_du + rec_term
    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    if freesurface:
        fs = DefaultDimension(name="fs", default_value=int(space_order/2))
        expression += [Eq(du.forward.subs({u.indices[-1]: model.nbpml - fs - 1}),
                          -du.forward.subs({u.indices[-1]: model.nbpml + fs + 1}))]
        expression += [Eq(u.forward.subs({u.indices[-1]: model.nbpml - fs - 1}),
                          -u.forward.subs({u.indices[-1]: model.nbpml + fs + 1}))]
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced',
                  name="Born%s" % randint(1e5))
    op()

    return rec.data


def adjoint_born(model, rec_coords, rec_data, u=None, op_forward=None, is_residual=False, space_order=8, nb=40, isic=False, dt=None, freesurface=False, isic2=False):
    clear_cache()

    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, rho, damp = model.m, model.rho, model.damp

    # Create adjoint wavefield and gradient
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
    gradient = Function(name='gradient', grid=model.grid, space_order=space_order)
    # Set up PDE and rearrange
    vlaplace, rho = acoustic_laplacian(v, rho)
    s = model.grid.time_dim.spacing
    stencil = damp * 2.0 * v - damp**2 * v.forward + damp * s**2 / m * vlaplace
    expression = [Eq(v.backward, stencil)]

    # Data at receiver locations as adjoint source
    rec_g = Receiver(name='rec_g', grid=model.grid, ntime=nt, coordinates=rec_coords)
    if op_forward is None:
        rec_g.data[:] = rec_data[:]
    adj_src = rec_g.inject(field=v.backward, offset=model.nbpml, expr=rec_g * rho * dt**2 / m)

    # Gradient update
    if u is None:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)

    if isic:
        # sum u.dx * v.dx fo x in dimensions.
        # space_order//2
        diff_u_v = sum([first_derivative(u, dim=d, order=space_order//2)*
                        first_derivative(v, dim=d, order=space_order//2)
                        for d in u.space_dimensions])
        gradient_update = [Inc(gradient, gradient - dt * (u * v.dt2 * m + diff_u_v) / rho)]

    elif isic2:
        gradient_update = [Inc(gradient, gradient - dt * u * v )]
        grad2 = Function(name='grad2', grid=model.grid)
        isic_update = [Eq(grad2, gradient.laplace)]
    else:
        gradient_update = [Inc(gradient, gradient - dt * u.dt2 / rho * v)]

    # Create operator and run
    set_log_level('INFO')
    expression += adj_src + gradient_update
    subs = model.spacing_map
    subs[u.grid.time_dim.spacing] = dt
    if freesurface:
        fs = DefaultDimension(name="fs", default_value=int(space_order/2))
        expression += [Eq(v.backward.subs({u.indices[-1]: model.nbpml - fs - 1}),
                         -v.backward.subs({u.indices[-1]: model.nbpml + fs + 1}))]

    if isic2:
        expression += [Eq(grad2, gradient.laplace)]
    op = Operator(expression, subs=subs, dse='advanced', dle='advanced',
                  name="Gradient%s" % randint(1e5))
    # Optimal checkpointing
    if op_forward is not None:
        rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
        cp = DevitoCheckpoint([u])
        n_checkpoints = None
        wrap_fw = CheckpointOperator(op_forward, u=u, m=model.m, rec=rec)
        wrap_rev = CheckpointOperator(op, u=u, v=v, m=model.m, rec_g=rec_g)

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

    if op_forward is not None and is_residual is not True:
        return fval, gradient.data
    else:
        return grad2.data if isic2 else gradient.data


########################################################################################################################

def forward_freq_modeling(model, src_coords, wavelet, rec_coords, freq, space_order=8, nb=40, dt=None, freesurface=False):
    # Forward modeling with on-the-fly DFT of forward wavefields
    clear_cache()

    # Parameters
    nt = wavelet.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, damp = model.m, model.damp
    freq_dim = Dimension(name='freq_dim')
    time = model.grid.time_dim

    # Create wavefields
    nfreq = freq.shape[0]
    u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
    f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
    f.data[:] = freq[:]
    ufr = Function(name='ufr', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + model.shape_domain)
    ufi = Function(name='ufi', dimensions=(freq_dim,) + u.indices[1:], shape=(nfreq,) + model.shape_domain)

    # Set up PDE and rearrange
    eqn = m * u.dt2 - u.laplace + damp * u.dt
    stencil = solve(eqn, u.forward, simplify=False, rational=False)[0]
    expression = [Eq(u.forward, stencil)]
    expression += [Eq(ufr, ufr + u*cos(2*np.pi*f*time*dt))]
    expression += [Eq(ufi, ufi + u*sin(2*np.pi*f*time*dt))]

    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, offset=model.nbpml, expr=src * dt**2 / m)

    # Data is sampled at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=u, offset=model.nbpml)

    # Create operator and run
    set_log_level('ERROR')
    expression += src_term + rec_term
    if freesurface:
        fs = DefaultDimension(name="fs", default_value=int(space_order/2))
        expression += [Eq(u.forward.subs({u.indices[-1]: model.nbpml - fs - 1}),
                          -u.forward.subs({u.indices[-1]: model.nbpml + fs + 1}))]
    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced',
                  name="Forward%s" % randint(1e5))
    op()

    return rec.data, ufr, ufi


def adjoint_freq_born(model, rec_coords, rec_data, freq, ufr, ufi, space_order=8, nb=40, dt=None, freesurface=False):
    clear_cache()

    # Parameters
    nt = rec_data.shape[0]
    if dt is None:
        dt = model.critical_dt
    m, damp = model.m, model.damp
    nfreq = ufr.shape[0]
    time = model.grid.time_dim

    # Create the forward and adjoint wavefield
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
    f = Function(name='f', dimensions=(ufr.indices[0],), shape=(nfreq,))
    f.data[:] = freq[:]
    gradient = Function(name="gradient", grid=model.grid)

    # Set up PDE and rearrange
    eqn = m * v.dt2 - v.laplace - damp * v.dt
    stencil = solve(eqn, v.backward, simplify=False, rational=False)[0]
    expression = [Eq(v.backward, stencil)]

    # Data at receiver locations as adjoint source
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec.data[:] = rec_data[:]
    adj_src = rec.inject(field=v.backward, offset=model.nbpml, expr=rec * dt**2 / m)

    # Gradient update
    gradient_update = [Inc(gradient, gradient + (2*np.pi*f)**2/nt*(ufr*cos(2*np.pi*f*time*dt) + ufi*sin(2*np.pi*f*time*dt))*v)]

    # Create operator and run
    set_log_level('ERROR')
    expression += adj_src + gradient_update
    if freesurface:
        fs = DefaultDimension(name="fs", default_value=int(space_order/2))
        expression += [Eq(v.backward.subs({u.indices[-1]: model.nbpml - fs - 1}),
                         -v.backward.subs({u.indices[-1]: model.nbpml + fs + 1}))]
    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced',
                  name="Gradient%s" % randint(1e5))
    op()
    clear_cache()

    return gradient.data
