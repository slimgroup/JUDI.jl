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
from JAcoustic_codegen import forward_modeling, forward_born, adjoint_born, forward_freq_modeling, adjoint_freq_born
import time

# Model
shape = (101, 101)
spacing = (10., 10.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 3.5

# Density
rho = np.empty(shape, dtype=np.float32)
rho[:, :51] = 1.0
rho[:, 51:] = 2.0

# Set up model structures
model = Model(shape=shape, origin=origin, spacing=spacing, vp=v, rho=rho)

def smooth10(vel, shape):
    if np.isscalar(vel):
        return .9 * vel * np.ones(shape, dtype=np.float32)
    out = np.copy(vel)
    nz = shape[-1]

    for a in range(5, nz-6):
        if len(shape) == 2:
            out[:, a] = np.sum(vel[:, a - 5:a + 5], axis=1) / 10
        else:
            out[:, :, a] = np.sum(vel[:, :, a - 5:a + 5], axis=2) / 10
    return out

# Smooth background model
v0 = smooth10(v, shape)
rho0 = smooth10(rho, shape)
dm = (1/v)**2 - (1/v0)**2
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=v0, rho=rho0, dm=dm)

# Constant background model
v_const = np.empty(shape, dtype=np.float32)
v_const[:,:] = 1.5
rho_const = np.empty(shape, dtype=np.float32)
rho_const[:, :] = 1.0
dm_const =  (1/v)**2 - (1/v_const)**2
model_const = Model(shape=shape, origin=origin, spacing=spacing, vp=v_const, rho=rho_const, dm=dm_const)

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

# Linearized data
dD_hat = forward_born(model_const, src.coordinates.data, src.data, rec_t.coordinates.data, isic=False, dt=dt)
dm_hat = model0.dm.data

# Forward
dD = forward_born(model0, src.coordinates.data, src.data, rec_t.coordinates.data, isic=False, dt=dt)

# Adjoint
d0, u0 = forward_modeling(model0, src.coordinates.data, src.data, rec_t.coordinates.data, save=True, dt=dt)
dm = adjoint_born(model0, rec_t.coordinates.data, dD_hat.data, u0, isic=False, dt=dt)

# Adjoint test
a = np.dot(dD_hat.flatten(), dD.flatten())
b = np.dot(dm_hat.flatten(), dm.flatten())
print("Difference: ", a - b)
print("Relative error: ", a/b - 1)

plt.figure(); plt.imshow(np.transpose(model.m.data))
plt.figure(); plt.imshow(np.transpose(model0.m.data))
plt.figure(); plt.imshow(np.transpose(model_const.m.data))

plt.figure(); plt.imshow(np.transpose(model0.dm.data))
plt.figure(); plt.imshow(np.transpose(model_const.dm.data))

plt.figure(); plt.imshow(dD_hat)
plt.figure(); plt.imshow(dD)

plt.figure(); plt.imshow(np.transpose(dm_hat))
plt.figure(); plt.imshow(np.transpose(dm))



