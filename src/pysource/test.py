import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from sources import RickerSource, Receiver
from models import Model

from kernels import wave_kernel
from geom_utils import src_rec
from wave_utils import wavefield, grad_expr, lin_src, otf_dft

from devito import Operator, Function
from propagators import forward, adjoint, born, gradient
from interface import forward_rec

parser = ArgumentParser(description="Adjoint test args")

parser.add_argument("--tti", default=False, action='store_true',
                    help="Test acoustic or tti")

parser.add_argument('-nlayer', dest='nlayer', default=3, type=int,
                    help="Number of layers in model")

parser.add_argument("--ssa", default=False, action='store_true',
                    help="Test tti or ssa tti")

args = parser.parse_args()
is_tti = args.tti


# Model
shape = (301, 301)
spacing = (10., 10.)
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

# Set up model structures
if is_tti:
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  vp=v, epsilon=.09*(v-1.5), delta=.075*(v-1.5),
                  rho=1, space_order=8)

    model0 = Model(shape=shape, origin=origin, spacing=spacing,
                  vp=v0, epsilon=.09*(v-1.5), delta=.075*(v-1.5),
                  rho=1, space_order=8, dt=model.critical_dt)            
else:
    model = Model(shape=shape, origin=origin, spacing=spacing,
                  vp=v, rho=rho, space_order=8)

    model0 = Model(shape=shape, origin=origin, spacing=spacing,
                   vp=v0, rho=rho0, space_order=8, dt=model.critical_dt)

# Time axis
t0 = 0.
tn = 2000.
dt = model.critical_dt
nt = int(1 + (tn-t0) / dt)
time_axis = np.linspace(t0, tn, nt)

# Source
f1 = 0.015  # kHz
src = RickerSource(name='src', grid=model.grid, f0=f1, time=time_axis)
src.coordinates.data[0, :] = np.array(model.domain_size) * 0.5
src.coordinates.data[0, -1] = 20.

# Receiver for observed data
rec_t = Receiver(name='rec_t', grid=model.grid, npoint=301, ntime=nt)
rec_t.coordinates.data[:, 0] = np.linspace(0., 3000., num=301)
rec_t.coordinates.data[:, 1] = 50.

# Interface (Level 1)
# d_obs = forward_rec(model, src.coordinates.data, src.data, rec_t.coordinates.data, 
#     space_order=8, free_surface=False, dt_comp=dt)

# Propagators (Level 2)
d_obs, _ = forward(model, src.coordinates.data, rec_t.coordinates.data, src.data, space_order=8, save=False,
    q=0, free_surface=False, return_op=False, freq_list=None)

_, u0 = forward(model0, src.coordinates.data, rec_t.coordinates.data, src.data, space_order=8, save=True,
    q=0, free_surface=False, return_op=False, freq_list=None)

g = gradient(model0, d_obs, rec_t.coordinates.data, u0, return_op=False, space_order=8,
             w=None, free_surface=False, freq=None, dft_sub=None)

# Plot
plt.figure(); plt.imshow(d_obs, vmin=-1, vmax=1, cmap='gray', aspect='auto')
#plt.figure(); plt.imshow(d_pred.data, vmin=-1, vmax=1, cmap='gray', aspect='auto')
plt.figure(); plt.imshow(np.transpose(g), vmin=-1e2, vmax=1e2, cmap='gray', aspect='auto')
plt.show()