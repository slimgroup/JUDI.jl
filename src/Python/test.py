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
from JAcoustic_codegen import forward_modeling, adjoint_born, adjoint_modeling, forward_born
from scipy import ndimage

# Model
shape = (101, 101)
spacing = (10., 10.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
v[:, :51] = 1.5
v[:, 51:] = 3
v0 = np.empty(shape, dtype=np.float32)
v0[:, :] = 1.5
m = (1./v)**2
m0 = (1./v0)**2
dm = m - m0

w = np.random.randn(shape[0], shape[1]).astype('float32')

model = Model(shape=shape, origin=origin, spacing=spacing, vp=v)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=v0, dm=dm)

# Time axis
t0 = 0.
tn = 800.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time = np.linspace(t0,tn,nt)

# Source
f0 = 0.010
src = RickerSource(name='src', grid=model.grid, f0=f0, time=time)
src.coordinates.data[0,:] = np.array(model.domain_size) * 0.5
src.coordinates.data[0,-1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=101, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=101)
rec_t.coordinates.data[:, 1] = 20.

# Observed data
#dobs, utrue = forward_modeling(model, None, src.data, rec_t.coordinates.data, weight=w)
#w2 = adjoint_modeling(model, None, rec_t.coordinates.data, dobs.data, wavelet=src.data)

# Linearized modeling
dlin = forward_born(model0, None, src.data, rec_t.coordinates.data, weight=w)

# Adjoint modeling
u0 = forward_modeling(model0, None, src.data, None, weight=w, save=True)
g = adjoint_born(model0, rec_t.coordinates.data, dlin.data, u=u0)

plt.imshow(dlin.data, aspect='auto', cmap='gray')
plt.figure(); plt.imshow(np.transpose(g.data), vmin=np.min(g.data), vmax=np.max(g.data), aspect='auto', cmap='gray'); plt.show()
