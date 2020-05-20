import numpy as np
from argparse import ArgumentParser

from sources import RickerSource, Receiver
from models import Model

from propagators import *

parser = ArgumentParser(description="Adjoint test args")

parser.add_argument("--tti", default=False, action='store_true',
                    help="Test acoustic or tti")

parser.add_argument('-nlayer', dest='nlayer', default=2, type=int,
                    help="Number of layers in model")

parser.add_argument("--ssa", default=False, action='store_true',
                    help="Test tti or ssa tti")

args = parser.parse_args()
is_tti = args.tti


# Model
shape = (401, 401)
spacing = (5., 5.)
origin = (0., 0.)

v = np.empty(shape, dtype=np.float32)
v0 = np.empty(shape, dtype=np.float32)
rho = np.empty(shape, dtype=np.float32)
rho0 = np.empty(shape, dtype=np.float32)

v[:] = 1.5  # Top velocity (background)
v0[:] = 1.5  # Top velocity (background)
rho[:] = 1.0
rho0[:] = 1.0

vp_i = np.linspace(1.5, 4.5, args.nlayer)
rho_i = np.linspace(1.0, 2.8, args.nlayer)

for i in range(1, args.nlayer):
    v[..., i*int(shape[-1] / args.nlayer):] = vp_i[i]  # Bottom velocity
    rho[..., i*int(shape[-1] / args.nlayer):] = rho_i[i]  # Bottom velocity

dm = 1/v**2 - 1/v0**2
# Set up model structures
if is_tti:
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  vp=v, epsilon=.09*(v-1.5), delta=.075*(v-1.5),
                  rho=1, space_order=8)

    model0 = Model(shape=shape, origin=origin, spacing=spacing,
                   vp=v0, epsilon=.09*(v-1.5), delta=.075*(v-1.5),
                   rho=1, space_order=8, dt=model.critical_dt, dm=dm)
else:
    modelfs = Model(shape=shape, origin=origin, spacing=spacing,
                    vp=v, space_order=8, fs=True)
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  vp=v, space_order=8)
    model0 = Model(shape=shape, origin=origin, spacing=spacing, dm=dm,
                   vp=v0, space_order=8, dt=model.critical_dt)

# Time axis
t0 = 0.
tn = 3000.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0, tn, nt)

# Source
f1 = 0.030  # kHz
src = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 9.0

# Receiver for observed data
nrec = shape[0]
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., (shape[0]-1)*spacing[0], num=nrec)
rec_t.coordinates.data[:, 1] = 14.

# Interface (Level 1)
# d_obs = forward_rec(model, src.coordinates.data, src.data, rec_t.coordinates.data,
#                     space_order=8, free_surface=True)

N = 1000
a = .003
b = .030
freq_list = np.sort(a + (b - a) * (np.random.randint(N, size=(10,)) - 1) / (N - 1.))
dft_sub = 1
# Propagators (Level 2)
d_obs, _ = forward(modelfs, src.coordinates.data, rec_t.coordinates.data, src.data,
                   free_surface=True)
d_obs, _ = forward(model, src.coordinates.data, rec_t.coordinates.data, src.data,
                   free_surface=False)

# d_lin, _ = born(model0, src.coordinates.data, rec_t.coordinates.data,
#                 src.data, isic=True)


# d0, u0 = forward(model0, src.coordinates.data, rec_t.coordinates.data, src.data,
#                  dft_sub=dft_sub, space_order=8, save=False, freq_list=freq_list)

# g = gradient(model0, d_lin, rec_t.coordinates.data, u0, return_op=False, space_order=8,
#              w=None, free_surface=False, freq=freq_list, dft_sub=dft_sub)

# g2 = gradient(model0, d_lin, rec_t.coordinates.data, u0, return_op=False, space_order=8,
#               w=None, free_surface=False, freq=freq_list, dft_sub=dft_sub, isic=True)
# # Plot
# plt.figure()
# plt.imshow(d_lin, vmin=-1, vmax=1, cmap='gray', aspect='auto')
# plt.figure()
# plt.imshow(d0.data, vmin=-1, vmax=1, cmap='gray', aspect='auto')
# plt.figure()
# plt.imshow(g[model.nbl:-model.nbl, model.nbl:-model.nbl].T,
#            vmin=-1e1, vmax=1e1, cmap='gray', aspect='auto')
# plt.figure()
# plt.imshow(g2[model.nbl:-model.nbl, model.nbl:-model.nbl].T,
#            vmin=-1e1, vmax=1e1, cmap='gray', aspect='auto')
# plt.show()
