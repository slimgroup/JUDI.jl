# Adjoint test for F and J
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, JUDI.TimeModeling, Test, LinearAlgebra

## Set up model structure
n = (160, 170)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) .* 2.0f0
v[:,Int(round(end/3)):end] .= 4f0
v0 = smooth10(v,n)
rho = ones(Float32, n)
rho[:, Int(round(end/2)):end] .= 1.5f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1
model = Model(n,d,o,m,rho=rho)
model0 = Model(n,d,o,m0,rho=rho)

## Set up receiver geometry
nxrec = 141
xrec = range(600f0,stop=1000f0,length=nxrec)
yrec = 0f0
zrec = range(100f0,stop=100f0,length=nxrec)

# receiver sampling and recording time
timeR = 800f0	# receiver recording time [ms]
dtR = calculate_dt(n,d,o,v,rho)    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = 800f0
ysrc = 0f0
zsrc = 50f0

# source sampling and number of time steps
timeS = 800f0
dtS = calculate_dt(n,d,o,v,rho) # receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# Info structure
ntComp = get_computational_nt(srcGeometry,recGeometry,model0)
info = Info(prod(n), nsrc, ntComp)

# setup wavelet
f0 = 0.008f0
wavelet = ricker_wavelet(timeS,dtS,f0)
wave_rand = wavelet.*rand(Float32,size(wavelet))

###################################################################################################

# Modeling operators
opt = Options(sum_padding=true)
F = judiModeling(info,model0,srcGeometry,recGeometry; options=opt)
q = judiVector(srcGeometry,wavelet)

# Nonlinear modeling
d_hat = F*q

# Generate random noise data vector with size of d_hat in the range of F
qr = judiVector(srcGeometry,wave_rand)
d1 = F*qr

# Adjoint computation
q_hat = adjoint(F)*d1

# Result F
a = dot(d1, d_hat)
b = dot(q, q_hat)
@test isapprox(a/b - 1, 0, atol=1f-4)

# Linearized modeling
J = judiJacobian(F,q)

dD_hat = J*dm
dm_hat = adjoint(J)*d_hat

c = dot(dD_hat, d_hat)
d = dot(dm, dm_hat)
@test isapprox(c/d - 1, 0, atol=1f-4)
