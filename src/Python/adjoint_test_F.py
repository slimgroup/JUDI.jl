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
from JAcoustic_codegen import forward_modeling, adjoint_modeling
import time

# Model
shape = (301, 301)
spacing = (10., 10.)
origin = (0., 0.)
v1 = np.empty(shape, dtype=np.float32)
v1[:, :51] = 1.5
v1[:, 51:] = 3.5

# Density
rho = np.empty(shape, dtype=np.float32)
rho[:, :51] = 1.0
rho[:, 51:] = 2.0

# Set up model structures
model1 = Model(shape=shape, origin=origin, spacing=spacing, vp=v1, rho=rho)

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
v2 = np.empty(shape, dtype=np.float32)
v2[:, :41] = 1.5
v2[:, 41:71] = 2.5
v2[:, 71:] = 4.5
model2 = Model(shape=shape, origin=origin, spacing=spacing, vp=v2, rho=rho)

# Time axis
t0 = 0.
tn = 1000.
dt = model2.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0,tn,nt)

# Source
f1 = 0.008
src1 = RickerSource(name='src', grid=model1.grid, f0=f1, time=time_axis)
src1.coordinates.data[0,:] = np.array(model1.domain_size) * 0.5
src1.coordinates.data[0,-1] = 20.

f2 = 0.012
src2 = RickerSource(name='src', grid=model2.grid, f0=f2, time=time_axis)
src2.coordinates.data[0,:] = np.array(model2.domain_size) * 0.5
src2.coordinates.data[0,-1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model1.grid, npoint=401, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(100, 900, num=401)
rec_t.coordinates.data[:, 1] = 20.

# Test data and source
d_hat, _ = forward_modeling(model1, src1.coordinates.data, src1.data, rec_t.coordinates.data, dt=dt)
q_hat = src2.data

# Forward
d0, _ = forward_modeling(model2, src2.coordinates.data, src2.data, rec_t.coordinates.data, dt=dt)

# Adjoint
q0 = adjoint_modeling(model2, src2.coordinates.data, rec_t.coordinates.data, d_hat, dt=dt)

# Adjoint test
a = np.dot(d_hat.flatten(), d0.flatten())
b = np.dot(q_hat.flatten(), q0.flatten())
print("Adjoint test F")
print("Difference: ", a - b)
print("Relative error: ", a/b - 1)
