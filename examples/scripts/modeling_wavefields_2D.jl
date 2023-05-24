# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using LinearAlgebra, Random
using JUDI, SegyIO

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v0 = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 3f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup model structure
nsrc = 2	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
timeR = 1000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([400f0, 800f0])
ysrc = convertToCell([0f0, 0f0])
zsrc = convertToCell([20f0, 20f0])

# source sampling and number of time steps
timeS = 1000f0  # ms
dtS = 4f0   # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)
###################################################################################################

# Write shots as segy files to disk
opt = Options()

# Setup operators
Pr = judiProjection(recGeometry)
F = judiModeling(model; options=opt)
F0 = judiModeling(model0; options=opt)
Ps = judiProjection(srcGeometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q
qad = Ps*adjoint(F)*adjoint(Pr)*dobs

# Return wavefields
u = F*adjoint(Ps)*q
v = adjoint(F)*adjoint(Pr)*dobs

# Modify wavefields
v = abs(v)  # take absolute value
u = 2*u # multiple by scalar

# Wavefields as source
dnew = Pr*F*v
qnew = Ps*adjoint(F)*u

# Create custom wavefield as source (needs to be on computational time axis and contain padding)
dtComp = get_dt(model)
ntComp = get_computational_nt(q.geometry, model)
u0 = zeros(Float32, ntComp[1], model.n[1] + 2*model.nb, model.n[2] + 2*model.nb)
wavelet = -ricker_wavelet(timeS, dtComp, f0)
u0[1:length(wavelet), 100, 45] .= wavelet
uf = judiWavefield(dtComp, u0)
dobs2 = Pr*F*uf # same as dobs

# Wavefields as source + return wavefields
u2 = F*u
v2 = F*v

# Supported algebraic operations
u_add = u + u
u_sub = u - v
u_mult = u * 2f0
u_div = u / 2f0
u_norm = norm(u)
u_dot = dot(u, u)
u_abs = abs(u)

