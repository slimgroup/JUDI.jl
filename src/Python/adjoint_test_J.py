from __future__ import print_function
import numpy as np
import psutil, os, gc
from numpy.random import randint
from sympy import solve, cos, sin
from sympy import Function as fint
from devito.logger import set_log_level
from devito import Eq, Function, TimeFunction, Dimension, Operator, clear_cache
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model
from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
import matplotlib.pyplot as plt
from TTI_operators import forward_modeling, forward_born, adjoint_born
import time

# Model
shape = (101, 101)
spacing = (10., 10.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 2.0
v[:, 51:] = 4.5

# Density
# Velocity [km/s]
epsilon = (v[:, :] - 2.0)/10.0
delta = (v[:, :] - 2.0)/20.0
theta = (v[:, :] - 2.0)/5.0

# Set up model structures
model = Model(shape=shape, origin=origin, spacing=spacing, vp=v, epsilon=epsilon, delta=delta, theta=theta)

# Smooth background model
dm1 = .1 * np.ones(shape)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=v - .1, dm=dm1, epsilon=epsilon, delta=delta, theta=theta)

# Constant background model
v_const = np.empty(shape, dtype=np.float32)
v_const[:,:] = 1.5
dm_const =  (1/v)**2 - (1/v_const)**2
model_const = Model(shape=shape, origin=origin, spacing=spacing, vp=v_const, dm=dm_const, epsilon=epsilon, delta=delta, theta=theta)

# Time axis
t0 = 0.
tn = 1400.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0,tn,nt)

# Source
f0 = 0.010
src = RickerSource(name='src', grid=model.grid, f0=f0, time=time_axis)
src.coordinates.data[0,:] = np.array(model.domain_size) * 0.5
src.coordinates.data[0,-1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=401, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(100, 900, num=401)
rec_t.coordinates.data[:, 1] = 20.

# Linearized data J * dm_const
print("Forward J")
t1 = time.time()
dD_hat = forward_born(model_const, src.coordinates.data, src.data, rec_t.coordinates.data, isic=False, dt=dt)
dm_hat = model0.dm.data
t2 = time.time()
print(t2 - t1)

# Forward J * dm
print("Forward J")
t1 = time.time()
dD = forward_born(model0, src.coordinates.data, src.data, rec_t.coordinates.data, isic=True, dt=dt)
t2 = time.time()
print(t2 - t1)

# Adjoint J' * dD_hat
print("Adjoint J")
d0, u0, v0 = forward_modeling(model0, src.coordinates.data, src.data, rec_t.coordinates.data, dt=dt, save=True)
t1 = time.time()
dm = adjoint_born(model0, rec_t.coordinates.data, dD_hat.data, u0, isic=True, dt=dt)
t2 = time.time()
print(t2 - t1)

# Adjoint test
a = np.dot(dD_hat.flatten(), dD.flatten())
b = np.dot(dm_hat.flatten(), dm.flatten())
print("Adjoint test J")
print("Difference: ", a - b)
print("Relative error: ", a/b - 1)
