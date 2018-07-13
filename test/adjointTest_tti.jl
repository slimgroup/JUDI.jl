# Adjoint test for F and J
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, JUDI.TimeModeling, Images
@pyimport scipy.ndimage as ndi
## Set up model structure
n = (151, 151)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) * 2.0f0
v[:,Int(round(end/3)):end] = 3f0
v[:,Int(2*round(end/3)):end] = 4f0
v0 = ndi.gaussian_filter(v, sigma=10)
rho = v[:, :] / 2.0f0
epsilon = ndi.gaussian_filter((v[:, :] - 2.0f0)/5.0f0, sigma=3)
delta =  ndi.gaussian_filter((v[:, :] - 2.0f0)/10.0f0, sigma=3)
theta =  ndi.gaussian_filter((v[:, :] - 2.0f0)/2.0, sigma=3)
# Slowness squared [s^2/km^2]
m = (1f0./v).^2
m0 = (1f0./v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1
model0 = Model_TTI(n,d,o,m0; epsilon=epsilon, delta=delta, theta=theta, rho=rho)
model = Model_TTI(n,d,o,m; epsilon=epsilon, delta=delta, theta=theta, rho=rho)
# model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 151
xrec = linspace(0f0,1500f0,nxrec)
yrec = linspace(0f0, 0f0,nxrec)
zrec = linspace(50f0, 50f0,nxrec)

# receiver sampling and recording time
timeR = 2500f0	# receiver recording time [ms]
dtR = 2.0f0    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = 50f0
ysrc = 0f0
zsrc = 50f0

# source sampling and number of time steps
timeS = 2500f0
dtS = 2.0f0 # receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# Info structure
ntComp = get_computational_nt(srcGeometry,recGeometry,model0)
info = Info(prod(n), nsrc, ntComp)

# setup wavelet
f0 = 0.008f0
wavelet = 1e1*ricker_wavelet(timeS,dtS,f0)

###################################################################################################

# Modeling operators
opt = Options(sum_padding=true,  isic="rotated", t_sub=1, h_sub=2, space_order=16)
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
q = judiVector(srcGeometry, wavelet)

# Nonlinear modeling
d_hat = F*q

# Adjoint computation
q_hat = F'*d_hat

# Result F
println(dot(d_hat,d_hat))
println(dot(q,q_hat))
println("Residual: ", abs(dot(d_hat,d_hat) - dot(q,q_hat)))
println("Ratio: ", abs(dot(d_hat,d_hat)/dot(q,q_hat)) - 1.0)
#
# Linearized modeling
J = judiJacobian(F,q)

dD_hat = J*dm
dm_hat = J'*dD_hat

# Result J
println(dot(dD_hat, dD_hat))
println(dot(dm, dm_hat))
println("Residual: ", abs(dot(dD_hat,dD_hat) - dot(dm,dm_hat)))
println("Ratio: ", abs(dot(dD_hat,dD_hat)/dot(dm,dm_hat)) - 1.0)
