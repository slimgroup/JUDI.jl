# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot, Images, Dierckx

## Set up model structure
n = (201,201)   # (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v0 = ones(Float32,n) + 1.0f0

x = linspace(0.0f0, 2000f0, 201)
z = linspace(0.0f0, 2000f0, 201)

v = 2.0f0 - .6f0*exp.(-(x - 1000.0f0).^2/400f0^2)*exp.(-(x - 1000.0f0).^2/400f0^2)';

# Slowness squared [s^2/km^2]
m = (1./v).^2
m0 = (1./v0).^2
# Slowness squared [s^2/km^2]
mmin = (1f0./v).^2
mmax = (1f0./v).^2
# Setup info and model structure
model = Model(n,d,o,m)
model0 = Model(n,d,o,m0)

container = segy_scan("/data/mlouboutin3/Gaussian_sing/8Hz", "gaussian", ["GroupX", "GroupY", "ElevationScalar", "dt"])
d_obs = judiVector(container; segy_depth_key="RecGroupElevation")

# setup wavelet
src_geometry = Geometry(container; key="source")
f0 = 0.005
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1], f0)
q = judiVector(src_geometry, wavelet)


############################### FWI ###########################################
opt = Options(normalize=true)

# Bound projection
ProjBound(x) = boundproject(x, maximum(m), .9*minimum(m))

fevals = 20
batchsize = 21
fmin = 1.0
# Optimization parameters
srand(1)    # set seed of random number generator


# Objective function for minConf library
count = 0
function objective_function(x, opt)
	model0.m = reshape(x,model0.n);
	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	d_sub = get_data(d_obs[i])
	fval, grad = fwi_objective(model0, q[i], d_sub; options=opt)
	grad = reshape(grad, model0.n)
	grad[:, 1:5] = 0
	grad[:, end-5:end] = 0
	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

# FWI with SPG
options = pqn_options(verbose=3, maxIter=fevals, corrections=10)
# options = spg_options(verbose=3, maxIter=fevals, memory=1)
x = vec(m0)
fmax=5.0
fwi_obj(x) = objective_function(x, opt)
f, g = fwi_obj(x)
# save("first_grad.jld", "m0", m0, "g", reshape(g, model0.n))
x, fsave, funEvals= minConf_PQN(fwi_obj, x, ProjBound, options)
save("FWI.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

# fmax = 8.0
# fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWI-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
#
# fmax = 12.0
# fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWI-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

############################### GS-FWI-shot ###########################################
opt = Options(normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "shot"))
srand(1)    # set seed of random number generator
# Objective function for minConf library


# FWI with SPG
x = vec(m0)
fwi_obj(x) = objective_function(x, opt)
fgss, ggss = fwi_obj(x)
# save("first_gradgss.jld", "m0", m0, "g", reshape(g, model0.n))
x, fsave, funEvals= minConf_PQN(fwi_obj, vec(x), ProjBound, options)
save("FWIgss.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

# fmax = 8.0
# opt.gs["maxshift"] = Float32(2000./fmax)
# fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWIgss-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
#
# fmax = 12.0
# opt.gs["maxshift"] = Float32(2000./fmax)
# fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWIgss-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

############################### GS-FWI-trace ###########################################
opt = Options(normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "trace"))
srand(1)    # set seed of random number generator


# FWI with SPG
x = vec(m0)
# fmax=5.0
fwi_obj(x) = objective_function(x, opt)
fgst, gst = fwi_obj(x)
# save("first_gradgst.jld", "m0", m0, "g", reshape(g, model0.n))
x, fsave, funEvals= minConf_PQN(fwi_obj, x, ProjBound, options)
save("FWIgst.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
#
# fmax = 8.0
# opt.gs["maxshift"] = Float32(2000./fmax)
# fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWIgst-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
#
# fmax = 12.0
# opt.gs["maxshift"] = Float32(2000./fmax)
# fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWIgst-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
