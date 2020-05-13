# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, SegyIO, LinearAlgebra, PyPlot, IterativeSolvers, JOLI

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 5f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# Source wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

###################################################################################################

# Write shots as segy files to disk
opt = Options(return_array = true)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)

# Random weights (size of the model)
w = randn(Float32, model.n)

# Create operator for injecting the weights, multiplied by the provided wavelet(s)
Pw = judiLRWF(info, wavelet)

# Model observed data w/ extended source
lambda = 1e2
I = joDirac(info.n, DDT=Float32, RDT=Float32)
F = Pr*F*adjoint(Pw)
F̄ = [F; lambda*I]

# Simultaneous observed data
d_sim = F*vec(w)

# Adjoint operation
w_adj = adjoint(F)*d_sim

# LSQR
w_inv = zeros(Float32, info.n)
lsqr!(w_inv, F̄, [d_sim; vec(w)]; maxiter=20, verbose=true, damp=1e2)

d_pred = F*vec(w_inv);

# Plot results
figure()
subplot(1,2,1)
imshow(reshape(d_sim, recGeometry.nt[1], nxrec), vmin=-5e2, vmax=5e2, cmap="gray"); title("Observed data")
subplot(1,2,2)
imshow(reshape(d_pred, recGeometry.nt[1], nxrec), vmin=-2e2, vmax=2e2, cmap="gray"); title("Predicted data")

figure()
subplot(1,3,1)
imshow(adjoint(reshape(w, model.n)), vmin=-3, vmax=3, cmap="gray"); title("Weights")
subplot(1,3,2)
imshow(adjoint(reshape(w_adj, model.n)), vmin=minimum(w_adj), vmax=maximum(w_adj), cmap="gray"); title("Adjoint")
subplot(1,3,3)
imshow(adjoint(reshape(w_inv, model.n)), vmin=minimum(w_inv), vmax=maximum(w_inv), cmap="gray"); title("LSQR")