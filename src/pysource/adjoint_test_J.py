import numpy as np
from argparse import ArgumentParser
from scipy import ndimage

from sources import RickerSource, Receiver
from models import Model

from propagators import forward, born, gradient


parser = ArgumentParser(description="Adjoint test args")
parser.add_argument("--tti", default=False, action='store_true',
                    help="Test acoustic or tti")
parser.add_argument('-nlayer', dest='nlayer', default=3, type=int,
                    help="Number of layers in model")

args = parser.parse_args()
is_tti = args.tti

# Model
shape = (301, 151)
spacing = (10., 10.)
origin = (0., 0.)
v = np.empty(shape, dtype=np.float32)
rho = np.empty(shape, dtype=np.float32)
v[:] = 1.5  # Top velocity (background)
rho[:] = 1.0
vp_i = np.linspace(1.5, 4.5, args.nlayer)
for i in range(1, args.nlayer):
    v[..., i*int(shape[-1] / args.nlayer):] = vp_i[i]  # Bottom velocity

v0 = ndimage.gaussian_filter(v, sigma=10)
v0[v < 1.51] = v[v < 1.51]
v0 = ndimage.gaussian_filter(v0, sigma=3)
rho0 = (v0+.5)/2
dm = v0**(-2) - v**(-2)
dm[:, -1] = 0.
# Set up model structures
if is_tti:
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  vp=v0, epsilon=.045*(v0-1.5), delta=.03*(v0-1.5),
                  rho=rho0, theta=.1*(v0-1.5), dm=dm, space_order=8)
else:
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  vp=v0, rho=rho0, dm=dm)

# Time axis
t0 = 0.
tn = 2000.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0,tn,nt)

# Source
f1 = 0.008
src = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=301, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_t.coordinates.data[:, 1] = 20.
# Linearized data
print("Forward J")
dD_hat, u0l = born(model, src.coordinates.data, rec_t.coordinates.data, src.data, save=True)

# Adjoint
print("Adjoint J")
_, u0 = forward(model, src.coordinates.data, rec_t.coordinates.data, src.data, save=True)

# Adjoint
print("Adjoint J")
dm_hat = gradient(model, dD_hat, rec_t.coordinates.data, u0l)

# Adjoint test
a = np.dot(dD_hat.flatten(), dD_hat.flatten())
b = np.dot(dm_hat.flatten(), model.dm.data.flatten())
if is_tti:
    c = np.linalg.norm(u0[0].data.flatten()- u0l[0].data.flatten())
else:
    c = np.linalg.norm(u0.data.flatten()- u0l.data.flatten())
print("Difference between saving with forward and born", c)

print("Adjoint test J")
print("Difference: ", a - b)
print("Relative error: ", a/b - 1)

from IPython import embed; embed()
