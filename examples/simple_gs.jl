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
opt = Options(normalize=true)

# Optimization parameters
srand(1)	# set seed of random number generator
fevals = 15
batchsize = 15

# Objective function for minConf library
count = 0
function objective_function(x)
	model0.m = reshape(x,model0.n);

	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	d_sub = get_data(d_obs[i])
	if count < 15
		println("low")
    	MF = judiFilter(d_sub.geometry, 3.0, 5.0)
		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=5.0)
		q = judiVector(src_geometry, wave_low)
	elseif 14 < count < 30
		println("medium")
	    MF = judiFilter(d_sub.geometry, 3.0, 8.0)
		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=8.0)
		q = judiVector(src_geometry, wave_low)
	else
		println("high")
		MF = judiFilter(d_sub.geometry, 3.0, 10.0)
		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=10.0)
		q = judiVector(src_geometry, wave_low)
	end
	fval, grad = fwi_objective(model0, q[i], MF*d_sub; options=opt)
	grad = reshape(grad, model0.n)
	grad[:, 1:8] = 0
	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

# Bound projection
ProjBound(x) = boundproject(x,mmax,mmin)

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)
x = vec(model0.m)
x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
x = reshape(x, model0.n)
figure(); imshow(sqrt.(1f0./x)', vmin=1.5, vmax=4.5, cmap="jet"); title("FWI")
# figure(); imshow(sqrt.(1f0./x)' - sqrt.(1f0./m0)'); title("Diff")

#
# #########################  GS Trace ####################################################
#
# opt = Options(normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "traces"))
# model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)
# # Optimization parameters
# srand(1)	# set seed of random number generator
# # Objective function for minConf library
# count = 0
# function objective_function(x)
# 	model0.m = reshape(x,model0.n);
#
# 	# fwi function value and gradient
# 	i = randperm(d_obs.nsrc)[1:batchsize]
# 	d_sub = get_data(d_obs[i])
# 	if count < 15
# 		println("low")
#     	MF = judiFilter(d_sub.geometry, 3.0, 5.0)
# 		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=5.0)
# 		q = judiVector(src_geometry, wave_low)
# 	elseif 14 < count < 30
# 		println("medium")
# 		opt.gs["maxshift"] = Float32(2000./8.0)
# 	    MF = judiFilter(d_sub.geometry, 3.0, 8.0)
# 		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=8.0)
# 		q = judiVector(src_geometry, wave_low)
# 	else
# 		println("high")
# 		opt.gs["maxshift"] = Float32(2000./10.0)
# 		MF = judiFilter(d_sub.geometry, 3.0, 10.0)
# 		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=10.0)
# 		q = judiVector(src_geometry, wave_low)
# 	end
# 	fval, grad = fwi_objective(model0, q[i], MF*d_sub; options=opt)
# 	grad = reshape(grad, model0.n)
# 	grad[:, 1:8] = 0
# 	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search
#
# 	global count; count+= 1
#     return fval, vec(grad)
# end
#
# x2, fsave2, funEvals2= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
# x2, fsave2, funEvals2= minConf_SPG(objective_function, x2, ProjBound, options)
# x2, fsave2, funEvals2= minConf_SPG(objective_function, x2, ProjBound, options)
# x2 = reshape(x2, model0.n)
#
# figure(); imshow(sqrt.(1f0./x2)'); title("FWI-gs-trace")

#########################  GS Shots ####################################################
opt = Options(normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "shot"))
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)
# Optimization parameters
srand(1)	# set seed of random number generator

# Objective function for minConf library
count = 0
function objective_function(x)
	model0.m = reshape(x,model0.n);

	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	d_sub = get_data(d_obs[i])
	if count < 15
		println("low")
    	MF = judiFilter(d_sub.geometry, 3.0, 5.0)
		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=5.0)
		q = judiVector(src_geometry, wave_low)
	elseif 14 < count < 30
		println("medium")
		opt.gs["maxshift"] = Float32(2000./8.0)
	    MF = judiFilter(d_sub.geometry, 3.0, 8.0)
		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=8.0)
		q = judiVector(src_geometry, wave_low)
	else
		println("high")
		opt.gs["maxshift"] = Float32(2000./10.0)
		MF = judiFilter(d_sub.geometry, 3.0, 10.0)
		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=10.0)
		q = judiVector(src_geometry, wave_low)
	end
	fval, grad = fwi_objective(model0, q[i], MF*d_sub; options=opt)
	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search
	grad = reshape(grad, model0.n)
	grad[:, 1:8] = 0
	global count; count+= 1
    return fval, vec(grad)
end

x3, fsave3, funEvals3= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
x3, fsave3, funEvals3= minConf_SPG(objective_function, x3, ProjBound, options)
x3, fsave3, funEvals3= minConf_SPG(objective_function, x3, ProjBound, options)
x3 = reshape(x3, model0.n)

figure(); imshow(sqrt.(1f0./x3)', vmin=1.5, vmax=4.5, cmap="jet"); title("FWI-gs-shot")
