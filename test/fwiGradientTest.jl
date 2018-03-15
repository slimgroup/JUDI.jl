# 2D FWI gradient test with 4 sources
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, JUDI.TimeModeling

## Set up model structure
n = (120,100)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# # Velocity [km/s]
# v = ones(Float32,n) * 2.0f0
# v[:,Int(round(end/2)):end] = 3.0f0
# v0 = smooth10(v,n)
# rho = ones(Float32, n)
# rho[:, Int(round(end/2)):end] = 1.5f0
#
# # Slowness squared [s^2/km^2]
# m = (1f0./v).^2
# m0 = (1f0./v0).^2
# dm = m0 - m
#
# # Setup info and model structure
nsrc = 1	# number of sources
ntComp = 250
info = Info(prod(n), nsrc, ntComp)	# number of gridpoints, number of experiments, number of computational time steps
# model = Model(n,d,o,m;rho=rho)
# model0 = Model(n,d,o,m0;rho=rho)


# Velocity [km/s]
v = ones(Float32,n) * 2.0f0
v[:,Int(round(end/3)):end] = 4f0
v0 = smooth10(v,n)
epsilon = (v[:, :] - 2.0f0)/10.0f0
delta = (v[:, :] - 2.0f0)/20.0f0
theta = (v[:, :] - 2.0f0)/5.0f0
# Slowness squared [s^2/km^2]
m = (1f0./v).^2
m0 = (1f0./v0).^2
dm = m - m0

# Setup info and model structure
nsrc = 1
model = Model_TTI(n,d,o,m; epsilon=epsilon, delta=delta, theta=theta)
model0 = Model_TTI(n,d,o,m0; epsilon=epsilon, delta=delta, theta=theta)

## Set up receiver geometry
nxrec = 81
xrec = linspace(200f0,1000f0,nxrec)
yrec = 0.
zrec = linspace(100f0,100f0,nxrec)

# receiver sampling and recording time
timeR = 1000f0	# receiver recording time [ms]
dtR = calculate_dt(model)	# receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([600f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([50f0])

# source sampling and number of time steps
timeS = 1000f0
dtS = calculate_dt(model)	# receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.01f0
wavelet = ricker_wavelet(timeS,dtS,f0)

###################################################################################################

# Gradient test
h = 1f0
iter = 8
error1 = zeros(iter)
error2 = zeros(iter)
h_all = zeros(iter)
srcnum = 1:nsrc
modelH = deepcopy(model0)

# Observed data
opt = Options(sum_padding=true)
F = judiModeling(info,model,srcGeometry,recGeometry;options=opt)
q = judiVector(srcGeometry,wavelet)
d = F*q

# # FWI gradient and function value for m0
# Jm0, grad = fwi_objective(model0,q,d;options=opt)
#
# for j=1:iter
# 	# FWI gradient and function falue for m0 + h*dm
# 	modelH.m = model0.m + h*dm
# 	Jm, gradm = fwi_objective(modelH,q,d;options=opt)
#
# 	dJ = dot(grad,vec(dm))
#
# 	# Check convergence
# 	error1[j] = abs(Jm - Jm0)
# 	error2[j] = abs(Jm - (Jm0 + h*dJ))
#
# 	println(h, " ", error1[j], " ", error2[j])
# 	h_all[j] = h
# 	h = h/2f0
# end
#
# # Plot errors
# loglog(h_all, error1); loglog(h_all, 1e2*h_all)
# loglog(h_all, error2); loglog(h_all, 1e2*h_all.^2)
# legend([L"$\Phi(m) - \Phi(m0)$", "1st order", L"$\Phi(m) - \Phi(m0) - \nabla \Phi \delta m$", "2nd order"], loc="lower right")
# xlabel("h")
# ylabel(L"Error $||\cdot||^\infty$")
# title("FWI gradient test")
# #axis((h_all[end], h_all[1], 1.0e-8,500))


# FWI gradient and function value for m0
F0 = judiModeling(info,model0,srcGeometry,recGeometry;options=opt)
J = judiJacobian(F0,q)
dD_hat = J*vec(dm)
d0 = F0*q

for j=1:iter
	# FWI gradient and function falue for m0 + h*dm
	modelH.m = model0.m + h*dm
	# FWI gradient and function value for m0
	Fh = judiModeling(info,modelH,srcGeometry,recGeometry;options=opt)
    dh = Fh*q
	# dJ = dot(grad,vec(dm))

	# Check convergence
	error1[j] = norm(dh - d0)
	error2[j] = norm(dh - d0 - h * dD_hat)

	println(h, " ", error1[j], " ", error2[j])
	h_all[j] = h
	h = h/2f0
end

# Plot errors
loglog(h_all, error1); loglog(h_all, 1e2*h_all)
loglog(h_all, error2); loglog(h_all, 1e2*h_all.^2)
legend([L"$\Phi(m) - \Phi(m0)$", "1st order", L"$\Phi(m) - \Phi(m0) - \nabla \Phi \delta m$", "2nd order"], loc="lower right")
xlabel("h")
ylabel(L"Error $||\cdot||^\infty$")
title("FWI gradient test")
#axis((h_all[end], h_all[1], 1.0e-8,500))
