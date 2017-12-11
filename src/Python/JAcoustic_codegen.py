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
from sympy import solve, cos, sin
from sympy import Function as fint
from devito.logger import set_log_level
from devito import Eq, Function, TimeFunction, Forward, Backward, Dimension, Operator, clear_cache
from PySource import PointSource, Receiver
from PyModel import Model

def forward_modeling(shape, spacing, origin, slowness_sqr, src_coords, wavelet, rec_coords, save=False, space_order=8, nb=40):
    clear_cache()

    # Model
    model = Model(vp=np.sqrt(1/slowness_sqr), origin=origin, shape=shape, spacing=spacing, nbpml=nb)

    # Parameters
    nt = wavelet.shape[0]
    dt = model.critical_dt
    m, damp = model.m, model.damp

    # Create the forward wavefield
    if save is False:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order)
    else:
        u = TimeFunction(name='u', grid=model.grid, time_order=2, space_order=space_order, save=True, time_dim=nt)

    # Set up PDE and rearrange 
    eqn = m * u.dt2 - u.laplace + damp * u.dt
    stencil = solve(eqn, u.forward)[0]
    expression = [Eq(u.forward, stencil)]

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
    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced',
                  time_axis=Forward, name="Forward%s" % randint(1e5))
    op(dt=dt)

    return rec.data, u


def adjoint_modeling(shape, spacing, origin, slowness_sqr, src_coords, rec_coords, rec_data, space_order=8, nb=40):
    clear_cache()

    # Model
    model = Model(vp=np.sqrt(1/slowness_sqr), origin=origin, shape=shape, spacing=spacing, nbpml=nb)

    # Parameters
    nt = rec_data.shape[0]
    dt = model.critical_dt
    m, damp = model.m, model.damp
 
    # Create the adjoint wavefield
    v = TimeFunction(name="v", grid=model.grid, time_order=2, space_order=space_order)

    # Set up PDE and rearrange 
    eqn = m * v.dt2 - v.laplace - damp * v.dt
    stencil = solve(eqn, v.backward)[0]
    expression = [Eq(v.backward, stencil)]
    
    # Adjoint source is injected at receiver locations
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec.data[:] = rec_data[:]
    adj_src = rec.inject(field=v.backward, offset=model.nbpml, expr=rec * dt**2 / m)

    # Data is sampled at source locations
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    adj_rec = src.interpolate(expr=v, offset=model.nbpml)

    # Create operator and run
    set_log_level('ERROR')
    expression += adj_src + adj_rec
    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced',
                  time_axis=Backward, name="Backward%s" % randint(1e5))
    op(dt=dt)
    
    return src.data


def forward_born(shape, spacing, origin, slowness_sqr, src_coords, wavelet, rec_coords, perturbation, space_order=8, nb=40):
    clear_cache()

    # Model
    model = Model(vp=np.sqrt(1/slowness_sqr), origin=origin, shape=shape, spacing=spacing, dm=perturbation, nbpml=nb)

    # Parameters
    nt = wavelet.shape[0]
    dt = model.critical_dt
    m, dm, damp = model.m, model.dm, model.damp

    # Create the forward and linearized wavefield
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=space_order)
    du = TimeFunction(name="du", grid=model.grid, time_order=2, space_order=space_order)

    # Set up PDEs and rearrange 
    eqn = m * u.dt2 - u.laplace + damp * u.dt
    stencil1 = solve(eqn, u.forward)[0]
    eqn_lin = m * du.dt2 - du.laplace + damp * du.dt + dm * u.dt2
    stencil2 = solve(eqn_lin, du.forward)[0]
    expression_u = [Eq(u.forward, stencil1)]
    expression_du = [Eq(du.forward, stencil2)]

    # Define source symbol with wavelet
    src = PointSource(name='src', grid=model.grid, ntime=nt, coordinates=src_coords)
    src.data[:] = wavelet[:]
    src_term = src.inject(field=u.forward, offset=model.nbpml, expr=src * dt**2 / m)
    
    # Define receiver symbol
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec_term = rec.interpolate(expr=du, offset=model.nbpml)

    # Create operator and run
    set_log_level('ERROR')
    expression = expression_u + src_term + expression_du + rec_term
    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced',
                  time_axis=Forward, name="Born%s" % randint(1e5))
    op(dt=dt)

    return rec.data


def adjoint_born(shape, spacing, origin, slowness_sqr, src_coords, rec_coords, rec_data, u, space_order=8, nb=40):
    clear_cache()
    
    # Model
    model = Model(vp=np.sqrt(1/slowness_sqr), origin=origin, shape=shape, spacing=spacing, nbpml=nb)

    # Parameters
    nt = rec_data.shape[0]
    dt = model.critical_dt
    m, damp = model.m, model.damp

    # Create the forward and adjoint wavefield
    v = TimeFunction(name='v', grid=model.grid, time_order=2, space_order=space_order)
    gradient = Function(name="gradient", grid=model.grid)

    # Set up PDE and rearrange 
    eqn = m * v.dt2 - v.laplace - damp * v.dt
    stencil = solve(eqn, v.backward)[0]
    expression = [Eq(v.backward, stencil)]

    # Data at receiver locations as adjoint source
    rec = Receiver(name='rec', grid=model.grid, ntime=nt, coordinates=rec_coords)
    rec.data[:] = rec_data[:]
    adj_src = rec.inject(field=v.backward, offset=model.nbpml, expr=rec * dt**2 / m)

    # Gradient update
    gradient_update = [Eq(gradient, gradient - u * v.dt2)]

    # Create operator and run
    set_log_level('ERROR')
    expression += adj_src + gradient_update
    op = Operator(expression, subs=model.spacing_map, dse='advanced', dle='advanced',
                  time_axis=Backward, name="Gradient%s" % randint(1e5))
    op(dt=dt)
    clear_cache()

    return gradient.data

  
