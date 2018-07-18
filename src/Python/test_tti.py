from __future__ import print_function
import numpy as np
import psutil, os, gc
from numpy.random import randint
from sympy import solve, cos, sin
from sympy import Function as fint
from scipy import ndimage
from devito.logger import set_log_level
from devito import Eq, Function, TimeFunction, Dimension, Operator, clear_cache
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model
from checkpoint import DevitoCheckpoint, CheckpointOperator
from pyrevolve import Revolver
import matplotlib.pyplot as plt
from TTI_Staggered import forward_modeling, adjoint_born, forward_born
from JAcoustic_codegen import forward_modeling as fwd
from JAcoustic_codegen import adjoint_born as grad
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
theta = .7 * (v - 1.5)
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
dpred_data, u0, v0 = forward_modeling(model0, src.coordinates.data, src.data, rec.coordinates.data, save=True)
g1 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u0, pv=v0, isic='noop')
g2 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u0, pv=v0, isic='isotropic')
g3 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u0, pv=v0, isic='rotated')
# g5 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u0, pv=v0, dt=dt)
# du = forward_born(model0, src.coordinates.data, src.data, rec.coordinates.data, dt=dt)
# Acoustic for reference
dobs, utrue = fwd(model, src.coordinates.data, src.data, rec_t.coordinates.data)
dpred_data, u03 = fwd(model0, src.coordinates.data, src.data, rec.coordinates.data, save=True)
g4 = grad(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], u=u03, dt=dt, isic=True)
g5 = grad(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], u=u03)

################### source on the left ################### 
# src.coordinates.data[0,:] = np.array(model.domain_size) * 0.1
# src.coordinates.data[0,-1] = 20.
# dobs, utrue, v1 = forward_modeling(model, src.coordinates.data, src.data, rec_t.coordinates.data)
# dpred_data, u01, v01 = forward_modeling(model0, src.coordinates.data, src.data, rec.coordinates.data, save=True)
# g12 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u01, pv=v01, isic='noop')
# g22 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u01, pv=v01, isic='isotropic')
# g32 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u01, pv=v01, isic='rotated')
# # g52 = adjoint_born(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], ph=u01, pv=v01, dt=dt)
# # # Acoustic for reference
# dobs, utrue = fwd(model, src.coordinates.data, src.data, rec_t.coordinates.data)
# dpred_data, u03 = fwd(model0, src.coordinates.data, src.data, rec.coordinates.data, save=True)
# g42 = grad(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], u=u03, isic=True)
# g52 = grad(model0, rec.coordinates.data, dpred_data[:] - dobs.data[:], u=u03)
#
scacle1= .1*1e-2
scacle2= .1*1e-2
scale = .5*1e-1
plt.figure()
plt.subplot(351)
plt.imshow(np.transpose(g1[40:-40, 40:-40]), vmin=-scacle2, vmax=scacle2, cmap="seismic")
plt.title("No isic")
plt.subplot(352)
plt.imshow(np.transpose(g2[40:-40, 40:-40]), vmin=-scacle1, vmax=scacle1, cmap="seismic")
plt.title("isotropic isic")
plt.subplot(353)
plt.imshow(np.transpose(g3[40:-40, 40:-40]), vmin=-scacle1, vmax=scacle1, cmap="seismic")
plt.title("roated isic")
plt.subplot(354)
plt.imshow(np.transpose(g4[40:-40, 40:-40]), vmin=-scale, vmax=scale, cmap="seismic")
plt.title("isotropic isic isotropic media")
plt.subplot(355)
plt.imshow(np.transpose(g5[40:-40, 40:-40]), vmin=-scacle2, vmax=scacle2, cmap="seismic")
plt.title("isotropic no isic")
#
#
# plt.subplot(356)
# plt.imshow(np.transpose(g12[40:-40, 40:-40]), vmin=-scacle2, vmax=scacle2, cmap="seismic")
# plt.subplot(357)
# plt.imshow(np.transpose(g22[40:-40, 40:-40]), vmin=-scacle1, vmax=scacle1, cmap="seismic")
# plt.subplot(358)
# plt.imshow(np.transpose(g32[40:-40, 40:-40]), vmin=-scacle1, vmax=scacle1, cmap="seismic")
# plt.subplot(359)
# plt.imshow(np.transpose(g42[40:-40, 40:-40]), vmin=-scale, vmax=scale, cmap="seismic")
# plt.subplot(3, 5, 10)
# plt.imshow(np.transpose(g52[40:-40, 40:-40]), vmin=-scacle2, vmax=scacle2, cmap="seismic")
#
# plt.subplot(3, 5, 11)
# plt.imshow(np.transpose(g12[40:-40, 40:-40]+g1[40:-40, 40:-40]), vmin=-scacle2, vmax=scacle2, cmap="seismic")
# plt.subplot(3,5, 12)
# plt.imshow(np.transpose(g22[40:-40, 40:-40]+g2[40:-40, 40:-40]), vmin=-scacle1, vmax=scacle1, cmap="seismic")
# plt.subplot(3,5,13)
# plt.imshow(np.transpose(g32[40:-40, 40:-40]+g3[40:-40, 40:-40]), vmin=-scacle1, vmax=scacle1, cmap="seismic")
# plt.subplot(3,5,14)
# plt.imshow(np.transpose(g42[40:-40, 40:-40]+g4[40:-40, 40:-40]), vmin=-scale, vmax=scale, cmap="seismic")
# plt.subplot(3,5,15)
# plt.imshow(np.transpose(g52[40:-40, 40:-40]+g5[40:-40, 40:-40]), vmin=-scacle2, vmax=scacle2, cmap="seismic")

# plt.subplot(355)
# plt.imshow(np.transpose(g5[40:-40, 40:-40]), vmin=-1e2*scale, vmax=1e2*scale, cmap="seismic")
# plt.title("RTM")
#
# plt.subplot(356)
# plt.imshow(np.transpose(g12[40:-40, 40:-40]), vmin=-sclae, vmax=sclae, cmap="seismic")
# plt.subplot(357)
# plt.imshow(np.transpose(g22[40:-40, 40:-40]), vmin=-sclae, vmax=sclae, cmap="seismic")
# plt.subplot(358)
# plt.imshow(np.transpose(g32[40:-40, 40:-40]), vmin=-sclae, vmax=sclae, cmap="seismic")
# plt.subplot(359)
# plt.imshow(np.transpose(g42[40:-40, 40:-40]), vmin=-scale, vmax=scale, cmap="seismic")
# plt.subplot(3,5,10)
# plt.imshow(np.transpose(g52[40:-40, 40:-40]), vmin=-1e2*scale, vmax=1e2*scale, cmap="seismic")
#
#
# plt.subplot(3,5,11)
# plt.imshow(np.transpose((g1 + g12)[40:-40, 40:-40]), vmin=-sclae, vmax=sclae, cmap="seismic")
# plt.subplot(3,5,12)
# plt.imshow(np.transpose((g2 + g22)[40:-40, 40:-40]), vmin=-sclae, vmax=sclae, cmap="seismic")
# plt.subplot(3,5,13)
# plt.imshow(np.transpose((g3 + g32)[40:-40, 40:-40]), vmin=-sclae, vmax=sclae, cmap="seismic")
# plt.subplot(3,5,14)
# plt.imshow(np.transpose((g4+ g42)[40:-40, 40:-40]), vmin=-scale, vmax=scale, cmap="seismic")
# plt.subplot(3,5,15)
# plt.imshow(np.transpose((g5+ g52)[40:-40, 40:-40]), vmin=-1e2*scale, vmax=1e2*scale, cmap="seismic")
# plt.subplots_adjust(wspace=0, hspace=0)
# # plt.tight_layout()
plt.show()
from IPython import embed;embed()
