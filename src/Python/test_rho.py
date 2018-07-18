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
from JAcoustic_codegen import forward_modeling, adjoint_born
from scipy import ndimage

# Model
shape = (301, 301)
spacing = (12.5, 12.5)
origin = (0., 0.)
nrec = 301
v = np.empty(shape, dtype=np.float32)
v[:, :150] = 1.5
v[:, 150:] = 2.5
v[:, 230:] = 3.5
v[105:195, 2*95:] = 6.5

v0 = ndimage.gaussian_filter(v, sigma=5)

# Density
# rho = np.empty(shape, dtype=np.float32)
# rho[:, :51] = 1.0
# rho[:, 51:] = 2.0
# rho0 = np.empty(shape, dtype=np.float32)
# rho0[:, :] = 1.0

# Set up model structures
model = Model(shape=shape, origin=origin, spacing=spacing, vp=v)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=v0)

# Time axis
t0 = 0.
tn = 3500.
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
dobs, utrue = forward_modeling(model, src.coordinates.data, src.data, rec_t.coordinates.data)

##################################################################################################################

# Receiver for predicted data
rec = Receiver(name='rec', grid=model0.grid, npoint=101, ntime=nt)
rec.coordinates.data[:, 0] = np.linspace(0, model0.domain_size[0], num=101)
rec.coordinates.data[:, 1] = 20.

# Save wavefields
d0, u0 = forward_modeling(model0, src.coordinates.data, src.data, rec.coordinates.data, save=True, dt=dt)
g = adjoint_born(model0, rec.coordinates.data, d0[:] - dobs[:], u=u0, dt=dt)
g1 = adjoint_born(model0, rec.coordinates.data, d0[:] - dobs[:], u=u0, dt=dt, isic=True)
g2 = adjoint_born(model0, rec.coordinates.data, d0[:] - dobs[:], u=u0, dt=dt, isic2=True)
# plt.figure(); plt.imshow(d0, vmin=-1, vmax=1)
plt.figure(); plt.imshow(np.transpose(g.data), vmin=-10, vmax=10, cmap="seismic")
plt.figure(); plt.imshow(np.transpose(g1.data), vmin=-.1, vmax=.1, cmap="seismic")
plt.figure(); plt.imshow(np.transpose(g2.data), vmin=-.1, vmax=.1, cmap="seismic")
plt.show()
from IPython import embed; embed()
