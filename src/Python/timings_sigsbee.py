from __future__ import print_function
import numpy as np
from PySource import RickerSource
from PyModel import Model
from PySource import Receiver
from JAcoustic_codegen import forward_born, forward_freq_modeling, adjoint_freq_born, forward_modeling, adjoint_born
import time
import h5py
import matplotlib.pyplot as plt

# Load Sigsbee model
sigsbee = h5py.File('/scratch/slim/shared/mathias-philipp/sigsbee2A/Sigsbee_LSRTM.h5','r+')
m0 = np.transpose(np.array(sigsbee['m0']))
dm = np.transpose(np.array(sigsbee['dm']))

# Model
shape = (3201, 1201)
spacing = (7.62, 7.62)
origin = (0., 0.)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0), dm=dm)

# Time axis
t0 = 0.
tn = 10000.
dt = model0.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0,tn,nt)

# Source
f0 = 0.015
src = RickerSource(name='src', grid=model0.grid, f0=f0, time=time_axis)
src.coordinates.data[0,:] = np.array(4617.)
src.coordinates.data[0,-1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model0.grid, npoint=1200, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(4717., 16617., num=1200)
rec_t.coordinates.data[:, 1] = 50.

# Compute LS-RTM gradient w/ on-the-fly DFTs
num_frequencies = [1, 2, 4, 8, 16, 32, 64, 128, 256]
timings = np.zeros(len(num_frequencies))
for j in range(len(num_frequencies)):
    f = np.linspace(0.01, 0.01, num_frequencies[j])    # always use 10 Hz
    t1 = time.time()
    d0, ufr, ufi = forward_freq_modeling(model0, src.coordinates.data, src.data, rec_t.coordinates.data, freq=f, dt=dt, factor=8)
    t2 = time.time()
    print('Forward: ', t2 - t1)

    t3 = time.time()
    dm = adjoint_freq_born(model0, rec_t.coordinates.data, d0.data, f, ufr, ufi, isic=True, dt=dt, factor=8)
    t4 = time.time()
    print('Adjoint: ', t4 - t3)
    print('Total: ', (t2 - t1) + (t4 - t3))
    timings[j] = (t2 - t1) + (t4 - t3)

timings.dump('timing_sigsbee_frequencies.dat')

# Checkpointing
d0, _ = forward_modeling(model0, src.coordinates.data, src.data, rec_t.coordinates.data)
ta = time.time()
op_predicted = forward_modeling(model0, src.coordinates.data, src.data, rec_t.coordinates.data, op_return=True, dt=dt)
f1, g1 = adjoint_born(model0, rec_t.coordinates.data, d0.data, op_forward=op_predicted, dt=dt)
tb = time.time()
print('Optimal checkpointing: ', tb - ta)
timings_oc = np.array(tb - ta)
timings_oc.dump('timing_sigsbee_optimal_checkpointing.dat')


