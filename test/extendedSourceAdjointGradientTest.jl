# 2D FWI gradient test with 4 sources
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, JUDI.TimeModeling, Test, LinearAlgebra, JUDI4Flux
import LinearAlgebra.norm

## Set up model structure
n = (120,100)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) .* 2.0f0
v[:,Int(round(end/2)):end] .= 3.0f0
v0 = smooth10(v,n)


# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = m0 - m

# Setup info and model structure
nsrc = 1	# number of sources
ntComp = 250
info = Info(prod(n), nsrc, ntComp)	# number of gridpoints, number of experiments, number of computational time steps
model = Model(n,d,o,m)
model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 81
xrec = range(200f0,stop=1000f0,length=nxrec)
yrec = 0.
zrec = range(100f0,stop=100f0,length=nxrec)

# receiver sampling and recording time
timeR = 1000f0	# receiver recording time [ms]
dtR = 4f0	# receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([600f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([50f0])

# source sampling and number of time steps
timeS = 1000f0
dtS = 4f0	# receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.01f0
wavelet = ricker_wavelet(timeS,dtS,f0)

nt = get_computational_nt(srcGeometry, recGeometry, model0)
dt_comp = timeR / (nt[1] - 1)

###################################################################################################

# Gradient test
h = 1f0
iter = 8
error1 = zeros(Float32, iter)
error2 = zeros(Float32, iter)
h_all = zeros(Float32, iter)
srcnum = 1:nsrc

# Setup operators
opt = Options(sum_padding=true, return_array=true)
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Pw = judiLRWF(info, wavelet)

# Random weights (size of the model)
weights = zeros(Float32, model.n)
weights[:, 5] = randn(Float32, n[1])/1f2

# Combined operators and Jacobian
F = Pr*F*adjoint(Pw)
F0 = Pr*F0*adjoint(Pw)

function my_norm(x; dt=1, p=2)
    x = dt * sum(abs.(vec(x)).^p)
    return x^(1.f0/p)
end

function objective_function(F, w, m0, d)
    𝒢_T = ExtendedQAdjoint(F)
    r = 𝒢_T(d, m0) - w
    J = judiJacobian(𝒢_T.F, judiWeights(r[:,:,1,1]))
    f = .5f0*my_norm(vec(r); dt=dt_comp)^2
    g = adjoint(J)*d
    return f, g
end

# Observed data
d = F*vec(weights)
w = adjoint(F)*d
w = reshape(w, n[1], n[2], 1, 1)

# FWI gradient and function value for m0
Jm0, grad = objective_function(F0, w, m0, d)

for j=1:iter

	# Adjoint extended source function gradient and function falue
	Jm, gradm = objective_function(F0, w, m0 + h*dm, d)
	dJ = dot(grad,vec(dm))

	# Check convergence
	error1[j] = abs(Jm - Jm0)
	error2[j] = abs(Jm - (Jm0 + h*dJ))

	println(h, " ", error1[j], " ", error2[j])
	h_all[j] = h
	global h = h/2f0
end

# Check error decay
rate_0th_order = 2^(iter - 1)   # error decays w/ factor 2
rate_1st_order = 4^(iter - 1)   # error decays w/ factor 4

# Plot errors
if isinteractive()
    loglog(h_all, error1); loglog(h_all, 1e9*h_all)
    loglog(h_all, error2); loglog(h_all, 1e6*h_all.^2)
    legend([L"$\Phi(m) - \Phi(m0)$", "1st order", L"$\Phi(m) - \Phi(m0) - \nabla \Phi \delta m$", "2nd order"], loc="lower right")
    xlabel("h")
    ylabel(L"Error $||\cdot||^\infty$")
    title("Adjoint extended source gradient test")
end

@test error1[end] <= error1[1] / rate_0th_order
@test error2[end] <= error2[1] / rate_1st_order
