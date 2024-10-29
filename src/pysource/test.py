import numpy as np
from argparse import ArgumentParser

from sources import RickerSource, Receiver
from models import Model

from propagators import *
from interface import *

import matplotlib.PythonPlot as plt

parser = ArgumentParser(description="Adjoint test args")

parser.add_argument("--tti", default=False, action='store_true',
                    help="Test acoustic or tti")

parser.add_argument('-nlayer', dest='nlayer', default=2, type=int,
                    help="Number of layers in model")

parser.add_argument("--dft", default=False, action='store_true',
                    help="Gradients with on-the-gfly dft")

parser.add_argument("--fs", default=False, action='store_true',
                    help="Free surface")

args = parser.parse_args()
is_tti = args.tti
dft = args.dft

# Model
shape = (201, 201)
spacing = (5., 5.)
origin = (0., 0.)

m = np.empty(shape, dtype=np.float32)
m0 = np.empty(shape, dtype=np.float32)
rho = np.empty(shape, dtype=np.float32)
rho0 = np.empty(shape, dtype=np.float32)

m[:] = 1/1.5**2  # Top velocity (background)
m0[:] = 1.5**2  # Top velocity (background)
rho[:] = 1.0
rho0[:] = 1.0

m_i = np.linspace(1/1.5**2, 1/4.5**2, args.nlayer)
rho_i = np.linspace(1.0, 2.8, args.nlayer)

for i in range(1, args.nlayer):
    m[..., i*int(shape[-1] / args.nlayer):] = m_i[i]  # Bottom velocity
    rho[..., i*int(shape[-1] / args.nlayer):] = rho_i[i]  # Bottom velocity

dm = m0 - m
v = m**(-.5)
# Set up model structures
if is_tti:
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  m=m, epsilon=.09*(v-1.5), delta=.075*(v-1.5),
                  rho=1, space_order=8, fs=args.fs)

    model0 = Model(shape=shape, origin=origin, spacing=spacing,
                   m=m0, epsilon=.09*(v-1.5), delta=.075*(v-1.5),
                   rho=1, space_order=8, dt=model.critical_dt, dm=dm, fs=args.fs)
else:
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  m=m, space_order=8, fs=args.fs)
    model0 = Model(shape=shape, origin=origin, spacing=spacing, dm=dm,
                   m=m0, space_order=8, dt=model.critical_dt, fs=args.fs)

# Time axis
t0 = 0.
tn = 1500.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0, tn, nt)

# Source
f1 = 0.030  # kHz
src = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 6.0

# Receiver for observed data
nrec = shape[0]
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., (shape[0]-1)*spacing[0], num=nrec)
rec_t.coordinates.data[:, 1] = 14.

# Interface (Level 1)
d_obs = forward_rec(model, src.coordinates.data, src.data, rec_t.coordinates.data,
                    space_order=8)

if dft:
    N = 10
    a = .003
    b = .030
    freq_list = np.sort(a + (b - a) *
                        (np.random.randint(N, size=(10,)) - 1) / (N - 1.))
    dft_sub = 1
    save = False
else:
    freq_list = None
    save = True
    dft_sub = 1
# Propagators (Level 2)
d_obs, _, _ = forward(model, src.coordinates.data, rec_t.coordinates.data, src.data)

d_lin, _, _ = born(model0, src.coordinates.data, rec_t.coordinates.data,
                   src.data, isic=True)


d0, u0, _ = forward(model0, src.coordinates.data, rec_t.coordinates.data, src.data,
                    dft_sub=dft_sub, space_order=8, save=save, freq_list=freq_list)

g, _ = gradient(model0, d_lin, rec_t.coordinates.data, u0,
                space_order=8, freq=freq_list, dft_sub=dft_sub)

g2, _ = gradient(model0, d_lin, rec_t.coordinates.data, u0,
                 space_order=8, freq=freq_list, dft_sub=dft_sub, isic=True)
# Plot
plt.figure()
plt.imshow(d_lin.data, vmin=-.1, vmax=.1, cmap='RdGy', aspect='auto')
plt.figure()
plt.imshow(d0.data, vmin=-.1, vmax=.1, cmap='RdGy', aspect='auto')
plt.figure()
plt.imshow(g.data[model.nbl:-model.nbl, model.nbl:-model.nbl].T,
           vmin=-1e0, vmax=1e0, cmap='Greys', aspect='auto')
plt.figure()
plt.imshow(g2.data[model.nbl:-model.nbl, model.nbl:-model.nbl].T,
           vmin=-1e0, vmax=1e0, cmap='Greys', aspect='auto')
plt.show()
