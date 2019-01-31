# Adjoint test for F and J
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#


using Pkg; Pkg.activate("JUDI")
using PyCall, PyPlot, JUDI.TimeModeling, Images, LinearAlgebra, Test

## Set up model structure
n = (251, 251)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

function smooth(array, sigma)
	return typeof(array)(imfilter(array, Kernel.gaussian(sigma)))
end
# Velocity [km/s]
v = ones(Float32,n) * 2.0f0
v[:,trunc(Integer, end/3):end] .= 3f0
v[:,trunc(Integer, 2*end/3):end] .= 4f0
v0 = smooth(v, 10)
rho = v[:, :] / 2.0f0
epsilon = smooth((v[:, :] .- 2.0f0)/5.0f0, 3)
delta =  smooth((v[:, :] .- 2.0f0)/10.0f0, 3)
theta =  smooth((v[:, :] .- 2.0f0)/2.0, 3)
# Slowness squared [s^2/km^2]
m = (1f0./v).^2
m0 = (1f0./v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1
model0 = Model_TTI(n,d,o,m0; epsilon=epsilon, delta=delta, theta=theta)
model = Model_TTI(n,d,o,m; epsilon=epsilon, delta=delta, theta=theta)
# model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 151
xrec = range(0f0, stop=2500f0, length=nxrec)
yrec = range(0f0, stop=0f0, length=nxrec)
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
timeR = 1000f0	# receiver recording time [ms]
dtR = 2.0f0    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = 1250f0
ysrc = 0f0
zsrc = 50f0

# source sampling and number of time steps
timeS = 1000f0
dtS = 2.0f0 # receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# Info structure
ntComp = get_computational_nt(srcGeometry,recGeometry,model0)
info = Info(prod(n), nsrc, ntComp)

# setup wavelet
f0 = 0.015f0
wavelet = ricker_wavelet(timeS,dtS,f0)
wave_rand = wavelet.*rand(Float32,size(wavelet))
###################################################################################################

# Modeling operators
opt = Options(sum_padding=true) #, isic=false, t_sub=2, h_sub=2)
F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
q = judiVector(srcGeometry, wavelet)

# Nonlinear modeling
d_hat = F*q

# Generate random noise data vector with size of d_hat in the range of F
qr = judiVector(srcGeometry, wave_rand)
d1 = F*qr

# Adjoint computation
q_hat = adjoint(F)*d1

# Result F
a = dot(d1, d_hat)
b = dot(q, q_hat)
println(a, ", ", b)
@test isapprox(a/b - 1, 0, atol=1f-4)

# Linearized modeling
J = judiJacobian(F,q)

dD_hat = J*dm
dm_hat = adjoint(J)*d_hat

c = dot(dD_hat, d_hat)
d = dot(dm, dm_hat)
@test isapprox(c/d - 1, 0, atol=1f-4)
