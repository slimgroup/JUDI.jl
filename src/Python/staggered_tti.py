import numpy as np

from devito.logger import warning

from PySource import RickerSource, PointSource, Receiver
from PyModel import Model

from VTI_elas import ForwardOperator

shape = (301, 301)
spacing = (12.5, 12.5)
origin = (0., 0.)
nrec = 301

v = np.empty(shape, dtype=np.float32)
v[:, :150] = 1.5
v[:, 150:] = 2.5
v[:, 230:] = 3.5
v[105:195, 2*95:] = 6.5

rho = (0.31 * (1e3*v)**0.25)

epsilon = .15*(v - 1.5)
epsilon[105:195, 2*95:] = 0.

delta = .1 * (v - 1.5)
delta[105:195, 2*95:] = 0.

theta = .2 * (v - 1.5)
theta[105:195, 2*95:] = 0.

model = Model(shape=shape, origin=origin, spacing=spacing, vp=v, epsilon=epsilon, delta = delta, theta=theta, rho=rho)

# Time axis
t0 = 0.
tn = 3500.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time = np.linspace(t0,tn,nt)
f0 = 0.010
# Define source geometry (center of domain, just below surface)
src = RickerSource(name='src', grid=model.grid, f0=f0, time=time)
src.coordinates.data[0,:] = np.array(model.domain_size) * 0.5
src.coordinates.data[0,-1] = 20.
src.data[1:, 0] = -np.diff(src.data[:,0])
# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0, model.domain_size[0], num=nrec)
rec_t.coordinates.data[:, 1] = 20.


op = ForwardOperator(model, src, rec_t, space_order=8)

from IPython import embed; embed()
