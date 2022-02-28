import numpy as np
from argparse import ArgumentParser
from scipy import ndimage
from devito import inner
from devito.logger import info

from sources import RickerSource, Receiver
from models import Model

from propagators import forward, born, gradient


parser = ArgumentParser(description="Adjoint test args")
parser.add_argument("--tti", default=False, action='store_true',
                    help="Test acoustic or tti")
parser.add_argument("--viscoacoustic", default=False, action='store_true',
                    help="Test viscoacoustic")
parser.add_argument("--fs", default=False, action='store_true',
                    help="Test with free surface")
parser.add_argument('-so', dest='space_order', default=8, type=int,
                    help="Spatial discretization order")
parser.add_argument('-nlayer', dest='nlayer', default=3, type=int,
                    help="Number of layers in model")

args = parser.parse_args()
is_tti = args.tti
is_viscoacoustic = args.viscoacoustic
so = args.space_order

dtype = np.float32

# Model
shape = (301, 151)
spacing = (10., 10.)
origin = (0., 0.)
m = np.empty(shape, dtype=dtype)
rho = np.empty(shape, dtype=dtype)
m[:] = 1/1.5**2  # Top velocity (background)
rho[:] = 1.0
m_i = np.linspace(1/1.5**2, 1/4.5**2, args.nlayer)
for i in range(1, args.nlayer):
    m[..., i*int(shape[-1] / args.nlayer):] = m_i[i]  # Bottom velocity

m0 = ndimage.gaussian_filter(m, sigma=10)
m0[m > 1/1.51**2] = m[m > 1/1.51**2]
m0 = ndimage.gaussian_filter(m0, sigma=3)
rho0 = (m0**(-.5)+.5)/2
dm = m0 - m
dm[:, -1] = 0.
# Set up model structures
v0 = m0**(-.5)
if is_tti:
    model = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                  m=m0, epsilon=.045*(v0-1.5), delta=.03*(v0-1.5),
                  fs=args.fs, rho=rho0, theta=.1*(v0-1.5), dm=dm, space_order=so)
elif is_viscoacoustic:
    qp0 = np.empty(shape, dtype=np.float32)
    qp0[:] = 3.516*((v0[:]*1000.)**2.2)*10**(-6)
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  fs=args.fs, m=m0, rho=rho0, qp=qp0, dm=dm, space_order=so,
                  abc_type=True)
else:
    model = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                  fs=args.fs, m=m0, rho=rho0, dm=dm, space_order=so)

# Time axis
t0 = 0.
tn = 2000.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0, tn, nt)

# Source
f1 = 0.008
src = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 20.
weights = np.zeros(model.grid.shape)
weights[141:161, 141:161] = np.ones((20, 20))

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=301, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_t.coordinates.data[:, 1] = 20.

# Linearized data
print("Forward J")
dD_hat, u0l, _ = born(model, src.coordinates.data, rec_t.coordinates.data,
                      src.data, save=True)
# dD_hat, u0l, _ = born(model, None, rec_t.coordinates.data,
#                       src.data, save=True, ws=weights)
# Forward
print("Forward")
_, u0, _ = forward(model, src.coordinates.data, rec_t.coordinates.data,
                   src.data, save=True)
# _, u0, _ = forward(model, None, rec_t.coordinates.data,
#                    src.data, save=True, ws=weights)

# gradient
print("Adjoint J")
dm_hat, _ = gradient(model, dD_hat, rec_t.coordinates.data, u0, f0=f1)


a2 = model.critical_dt * np.dot(dD_hat.data.reshape(-1), dD_hat.data.reshape(-1))
b2 = np.dot(dm_hat.data.reshape(-1), model.dm.data.reshape(-1))


if is_tti:
    c = np.linalg.norm(u0[0].data.flatten() - u0l[0].data.flatten(), np.inf)
else:
    c = np.linalg.norm(u0.data.flatten() - u0l.data.flatten(), np.inf)

print("Difference between saving with forward and born", c)

print("Adjoint test J")
print("a = %2.5e, b = %2.5e, diff = %2.5e: rerr=%2.5e" % (a, b, a - b, (a-b)/(a+b)))


print("Adjoint test J")
print("a = %2.5e, b = %2.5e, diff = %2.5e: rerr=%2.5e" %
      (a2, b2, a2 - b2, (a2-b2)/(a2+b2)))
