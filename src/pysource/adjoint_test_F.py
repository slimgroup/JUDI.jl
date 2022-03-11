import numpy as np
from argparse import ArgumentParser
from devito import inner

from sources import RickerSource, Receiver
from models import Model

from propagators import forward, adjoint

parser = ArgumentParser(description="Adjoint test args")
parser.add_argument("--tti", default=False, action='store_true',
                    help="Test acoustic or tti")
parser.add_argument("--fs", default=False, action='store_true',
                    help="Test with free surface")
parser.add_argument('-nlayer', dest='nlayer', default=3, type=int,
                    help="Number of layers in model")
parser.add_argument('-so', dest='space_order', default=8, type=int,
                    help="Spatial discretization order")
args = parser.parse_args()
is_tti = args.tti
so = args.space_order

dtype = np.float32

# Model
shape = (301, 301)
spacing = (10., 10.)
origin = (0., 0.)
m = np.empty(shape, dtype=dtype)
rho = np.empty(shape, dtype=dtype)
m[:] = 1/1.5**2  # Top velocity (background)
rho[:] = 1.0
m_i = np.linspace(1/1.5**2, 1/4.5**2, args.nlayer)
rho_i = np.linspace(1.0, 2.8, args.nlayer)
for i in range(1, args.nlayer):
    m[..., i*int(shape[-1] / args.nlayer):] = m_i[i]  # Bottom velocity
    rho[..., i*int(shape[-1] / args.nlayer):] = rho_i[i]  # Bottom velocity

v = np.sqrt(1./m)
# Set up model structures
if is_tti:
    model = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                  m=m, epsilon=.09*(v-1.5), delta=.075*(v-1.5),
                  fs=args.fs, theta=.1*(v-1.5), rho=1, space_order=so)
else:
    model = Model(shape=shape, origin=origin, spacing=spacing, dtype=dtype,
                  fs=args.fs, m=m, rho=rho, space_order=so)

# Time axis
t0 = 0.
tn = 1300.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0, tn, nt)

# Source
f1 = 0.008
src1 = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src1.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src1.coordinates.data[0, -1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=301, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_t.coordinates.data[:, 1] = 20.

# Test data and source
d_hat, u1, _ = forward(model, src1.coordinates.data, rec_t.coordinates.data, src1.data)

# Adjoint
q0, _, _ = adjoint(model, d_hat, src1.coordinates.data, rec_t.coordinates.data)

# Adjoint test
a = inner(d_hat, d_hat)
b = inner(q0, src1)
print("Adjoint test F")
print("a = %2.5e, b = %2.5e, diff = %2.5e: rerr=%2.5e" % (a, b, a - b, (a-b)/(a+b)))
