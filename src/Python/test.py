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
v[:, 51:] = 5
m = (1./v)**2

#w = np.random.randn(shape[0], shape[1]).astype('float32')
w = np.ones(shape, dtype=np.float32)

model = Model(shape=shape, origin=origin, spacing=spacing, vp=v, nbpml=0)

# Time axis
t0 = 0.
tn = 1000.
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
dobs, utrue = forward_modeling(model, src.coordinates.data, np.asarray(src.data)[:,0], rec_t.coordinates.data, weight=w)
w2 = adjoint_modeling(model, src.coordinates, rec_t.coordinates.data, dobs.data, wavelet=np.asarray(src.data)[:,0])

plt.imshow(dobs.data, vmin=-4, vmax=4, aspect='auto', cmap='gray'); plt.show()

