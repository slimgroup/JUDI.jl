import numpy as np

from scipy import ndimage

from devito.logger import set_log_level
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model

import matplotlib.pyplot as plt
from TTI_Staggered import forward_modeling, adjoint_born, forward_born

# Model
shape = (301, 301)
spacing = (12.5, 12.5)
origin = (0., 0.)
nrec = 301
v = np.empty(shape, dtype=np.float32)
v[:, :150] = 1.5
v[:, 150:] = 2.5
v[:, 230:] = 3.5
v0 = ndimage.gaussian_filter(v, sigma=3)
v[105:195, 2*95:] = 6.5
v0[105:195, 2*95:] = 6.5
epsilon = .15*(v - 1.5)
epsilon[105:195, 2*95:] = 0.
delta = .1 * (v - 1.5)
delta[105:195, 2*95:] = 0.
theta = .2 * (v - 1.5)
theta[105:195, 2*95:] = 0.
model = Model(shape=shape, origin=origin, spacing=spacing, vp=v, epsilon=epsilon, delta = delta, theta=theta)
model0 = Model(shape=shape, origin=origin, spacing=spacing, vp=v0, epsilon=epsilon, delta = delta, theta=theta, dm=1/v**2 - 1/v0**2)
# Time axis
t0 = 0.
tn = 3500.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time = np.linspace(t0,tn,nt)
################### source in the center ################### 
# Source
f0 = 0.0196
src = RickerSource(name='src', grid=model.grid, f0=f0, time=time)
src.coordinates.data[0,:] = np.array(model.domain_size) * 0.5
src.coordinates.data[0,-1] = 20.
src.data[1:, 0] = -np.diff(src.data[:,0])
# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=nrec)
rec_t.coordinates.data[:, 1] = 20.
# Observed data
dobs, utrue, v1 = forward_modeling(model, src.coordinates.data, src.data, rec_t.coordinates.data)
# Receiver for predicted data
rec = Receiver(name='rec', grid=model0.grid, npoint=nrec, ntime=nt)
rec.coordinates.data[:, 0] = np.linspace(0, model0.domain_size[0], num=nrec)
rec.coordinates.data[:, 1] = 20.
# Save wavefields
dpred_data, u0, v0 = forward_modeling(model0, src.coordinates.data, src.data, rec.coordinates.data)
