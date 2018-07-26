from __future__ import print_function
import numpy as np
from PySource import RickerSource
from PyModel import Model
from PySource import Receiver
from JAcoustic_codegen import forward_born, forward_modeling, forward_freq_modeling, adjoint_freq_born, adjoint_born
import time
import h5py
import matplotlib.pyplot as plt

# Load Sigsbee model
sigsbee = h5py.File('/scratch/slim/shared/mathias-philipp/bp_synthetic_2004/model/vp_fine.h5','r+')
m0 = np.transpose(np.array(sigsbee['m0']))
dm = np.transpose(np.array(sigsbee['dm']))

# Model
shape = (10789, 1911)
spacing = (6.25, 6.26)
origin = (0., 0.)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=np.sqrt(1/m0), dm=dm)

# Time axis
t0 = 0.
tn = 14000.
dt = model0.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0,tn,nt)

# Source
f0 = 0.020
src = RickerSource(name='src', grid=model0.grid, f0=f0, time=time_axis)
src.coordinates.data[0,:] = np.array(4617.)
src.coordinates.data[0,-1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model0.grid, npoint=1201, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(4717., 19717., num=1201)
rec_t.coordinates.data[:, 1] = 50.

# Compute LS-RTM gradient w/ on-the-fly DFTs
num_frequencies = [1, 2, 4, 8, 16, 32, 64, 128]
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

timings.dump('timing_bp_frequencies.dat')

# Checkpointing
d0, _ = forward_modeling(model0, src.coordinates.data, src.data, rec_t.coordinates.data)
ta = time.time()
op_predicted = forward_modeling(model0, src.coordinates.data, src.data, rec_t.coordinates.data, op_return=True, dt=dt)
f1, g1 = adjoint_born(model0, rec_t.coordinates.data, d0.data, op_forward=op_predicted, dt=dt)
tb = time.time()
print('Optimal checkpointing: ', tb - ta)
timings_oc = np.array(tb - ta)
timings_oc.dump('timing_bp_optimal_checkpointing.dat')


