# Adjoint test for F and J
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, JUDI.TimeModeling

## Set up model structure
n = (161, 171)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) * 2.0f0
v[:,Int(round(end/3)):end] = 3f0
v[:,Int(2*round(end/3)):end] = 4f0
v0 = ones(Float32,n) * 2.0f0
epsilon = (v[:, :] - 2.0f0)/5.0f0
delta = (v[:, :] - 2.0f0)/10.0f0
theta = (v[:, :] - 2.0f0)/2.0
# Slowness squared [s^2/km^2]
m = (1f0./v).^2
m0 = (1f0./v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1
model = Model(n,d,o,m)
model0 = Model(n,d,o,m0)
# model = Model(n,d,o,m)
# model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 161
xrec = linspace(0f0,1600f0,nxrec)
yrec = 0f0
zrec = linspace(50f0, 50f0,nxrec)

# receiver sampling and recording time
timeR = 2000f0	# receiver recording time [ms]
dtR = 2.0f0    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = 50f0
ysrc = 0f0
zsrc = 50f0

# source sampling and number of time steps
timeS = 2000f0
dtS = 2.0f0 # receiver sampling interval

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
opt = Options(sum_padding=true, isic=true, t_sub=2, h_sub=2)
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
q = judiVector(srcGeometry, wavelet)

# Nonlinear modeling
d_hat = F*q

# Generate random noise data vector with size of d_hat in the range of F
qr = judiVector(srcGeometry,wave_rand)
d1 = F*qr

# Adjoint computation
q_hat = F'*d1

# Result F
println(abs(dot(d1,d_hat)))
println(abs(dot(q,q_hat)))
println("Residual: ", abs(dot(d1,d_hat) - dot(q,q_hat)))
println("Ratio: ", abs(dot(d1,d_hat)/dot(q,q_hat)) - 1.0)

# Linearized modeling
J = judiJacobian(F,q)

dD_hat = J*dm
dm_hat = J'*dD_hat

# Result J
println(dot(dD_hat,dD_hat))
println(dot(dm,dm_hat))
println("Residual: ", abs(dot(dD_hat,dD_hat) - dot(dm,dm_hat)))
println("Ratio: ", abs(dot(dD_hat,dD_hat)/dot(dm,dm_hat)) - 1.0)
