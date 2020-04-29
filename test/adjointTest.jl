# Adjoint test for F and J
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyPlot, JUDI.TimeModeling, Test, LinearAlgebra

## Set up model structure
n = (160, 170)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) .* 2.0f0
v[:,Int(round(end/3)):end] .= 4f0
v0 = smooth10(v,n)

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1
model = Model(n,d,o,m,)
model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 141
xrec = range(600f0,stop=1000f0,length=nxrec)
yrec = 0f0
zrec = range(100f0,stop=100f0,length=nxrec)

# Sampling and recording time
time = 800f0	# receiver recording time [ms]
dt = 1.0

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dt,t=time,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = 800f0
ysrc = 0f0
zsrc = 50f0

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dt,t=time)

# Info structure
ntComp = get_computational_nt(srcGeometry,recGeometry,model0)
info = Info(prod(n), nsrc, ntComp)

# setup wavelet
f0 = 0.008f0
wavelet = ricker_wavelet(time,dt,f0)
wave_rand = wavelet.*rand(Float32,size(wavelet))

###################################################################################################
# Modeling

# Modeling operators
opt = Options(sum_padding=true, dt_comp=dt)

F = judiModeling(info,model0,srcGeometry,recGeometry; options=opt)
q = judiVector(srcGeometry,wavelet)

# Nonlinear modeling
y = F*q

# Generate random noise data vector with size of d_hat in the range of F
x = judiVector(srcGeometry, wave_rand)

# Forward-adjoint 
y_hat = F*x
x_hat = adjoint(F)*y

# Result F
a = dot(y, y_hat)
b = dot(x, x_hat)
@test isapprox(a/b - 1f0, 0, atol=1f-5)

# Linearized modeling
J = judiJacobian(F,q)
x = vec(dm)

y_hat = J*x
x_hat = adjoint(J)*y

c = dot(y, y_hat)
d = dot(x, x_hat)
@test isapprox(c/d - 1f0, 0, atol=1f-3)


###################################################################################################
# Extended source modeling

opt = Options(return_array=true, sum_padding=true, dt_comp=dt)

Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model0; options=opt)
Pw = judiLRWF(info, wavelet)
F = Pr*F*adjoint(Pw)

# Extended source weights
w = vec(randn(Float32, model0.n))
x = vec(randn(Float32, model0.n))

# Generate random noise data vector with size of d_hat in the range of F
y = F*w

# Forward-Adjoint computation
y_hat = F*x
x_hat = adjoint(F)*y

# Result F
a = dot(y, y_hat)
b = dot(x, x_hat)
@test isapprox(a/b - 1, 0, atol=1f-5)

# Linearized modeling
J = judiJacobian(F, w)
x = vec(dm)

y_hat = J*x
x_hat = adjoint(J)*y

c = dot(y, y_hat)
d = dot(x, x_hat)
@test isapprox(c/d - 1, 0, atol=1f-1)
