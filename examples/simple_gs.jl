# 2D FWI on Overthrust model using minConf library
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#

using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot, Images, Dierckx

# Load starting model
# n,d,o,m00 = read(h5open("../data/overthrust_model.h5","r"), "n", "d", "o", "m0")
datapath = "/data/mlouboutin3/devito_data/Simple2D/vp_marmousi_bi"
f = open(datapath, "r")
v = read(f, Float32, 1601*401)
v = reshape(v, 401, 1601)'

n = (481, 121)
d = (25., 25.)
o = (0., 0.)

# put on 15m grid
x = linspace(0., 12000., 1601)
y = linspace(0., 3000., 401)
spl = Spline2D(x, y, v)
x0 = linspace(0., 12000., 481)
y0 = linspace(0., 3000., 121)
v_coarse = evalgrid(spl, x0, y0)
m = 1f0./v_coarse.^2
m0 = 1f0./v_coarse.^2

m0[:, 8:end] = imfilter(m[:,8:end] , Kernel.gaussian(15));
model0 = Model(n, d, o, m0)

# Bound constraints
v0 = sqrt.(1./model0.m)
vmin = 1.02f0
vmax = 5f0

# Slowness squared [s^2/km^2]
mmin = (1f0./vmax).^2
mmax = (1f0./vmin).^2

# Load data
# block = segy_read("../data/overthrust_shot_records.segy")
# d_obs = judiVector(block)
container = segy_scan("/data/mlouboutin3/devito_data/Simple2D/marine/nofs", "marine", ["GroupX", "GroupY", "ElevationScalar", "dt"])
d_obs = judiVector(container; segy_depth_key="RecGroupElevation")

# Set up wavelet
src_geometry = Geometry(container; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.025)	# 8 Hz wavelet
wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=5.0)
q = judiVector(src_geometry, wave_low)

# shot1 = d_obs.data[1][1]
# shot1_filtered = low_filter(Float32.(shot1.data), 4.0; fmin=1.0, fmax=5.0)


############################### FWI ###########################################
opt = Options(normalize=true, limit_m=true, buffer_size=1000f0)

# Bound projection
ProjBound(x) = boundproject(x, maximum(m0), .9*minimum(m0))

fevals = 10
batchsize = 160
fmin = 1.0
options = spg_options(verbose=3, maxIter=fevals, memory=5)
# Optimization parameters
srand(1)    # set seed of random number generator


# Objective function for minConf library
count = 0
function objective_function(x, fmin, fmax, opt)
	model0.m = reshape(x,model0.n);
	# println("Frequency band: ", fmin, " ,", fmax)
	# println("Gradient sampling: ", ~isempty(opt.gs))
	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	d_sub = get_data(d_obs[i])
	MF = judiFilter(d_sub.geometry, fmin, fmax)
	wave_low = low_filter(wavelet, 4.0; fmin=fmin, fmax=fmax)
	q = judiVector(src_geometry, wave_low)
	fval, grad = fwi_objective(model0, q[i], MF*d_sub; options=opt)
	grad = reshape(grad, model0.n)
	grad[v_coarse .< 1.51] = 0
	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=1)
x = vec(m0)
fmax=5.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# f, g = fwi_obj(x)
# save("first_grad.jld", "m0", m0, "g", reshape(g, model0.n))
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWI-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

fmax = 8.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWI-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

fmax = 12.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWI-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

############################### GS-FWI-shot ###########################################
opt = Options(limit_m=true, buffer_size=1000f0, freesurface=true, normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "shot"))
srand(1)    # set seed of random number generator
# Objective function for minConf library


# FWI with SPG
x = vec(m0)
fmax = 5.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# f, g = fwi_obj(x)
# save("first_gradgss.jld", "m0", m0, "g", reshape(g, model0.n))
x, fsave, funEvals= minConf_SPG(fwi_obj, vec(x), ProjBound, options)
save("FWIgss-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

fmax = 8.0
opt.gs["maxshift"] = Float32(2000./fmax)
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWIgss-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

fmax = 12.0
opt.gs["maxshift"] = Float32(2000./fmax)
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWIgss-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

############################### GS-FWI-trace ###########################################
opt = Options(limit_m=true, buffer_size=1000f0, freesurface=true, normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "trace"))
srand(1)    # set seed of random number generator


# FWI with SPG
x = vec(m0)
fmax=5.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
# f, g = fwi_obj(x)
# save("first_gradgst.jld", "m0", m0, "g", reshape(g, model0.n))
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWIgst-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

fmax = 8.0
opt.gs["maxshift"] = Float32(2000./fmax)
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWIgst-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

fmax = 12.0
opt.gs["maxshift"] = Float32(2000./fmax)
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
save("FWIgst-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
