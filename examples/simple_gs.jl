# 2D FWI on Overthrust model using minConf library
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#

using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot, Images

# Load starting model
n,d,o,m00 = read(h5open("../data/overthrust_model.h5","r"), "n", "d", "o", "m0")


m0 = imfilter(m00, Kernel.gaussian(10));
m0[:, 1:21] = m00[:, 1:21]
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1./model0.m)
vmin = 1.3f0
vmax = 6.5f0

# Slowness squared [s^2/km^2]
mmin = (1f0./vmax).^2
mmax = (1f0./vmin).^2

# Load data
block = segy_read("../data/overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)	# 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

############################### FWI ###########################################
opt = Options(normalize=true)

# Optimization parameters
srand(1)	# set seed of random number generator
fevals = 25
batchsize = 15

# Objective function for minConf library
count = 0
function objective_function(x)
	model0.m = reshape(x,model0.n);

	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	fval, grad = fwi_objective(model0, q[i], d_obs[i]; options=opt)
	grad = reshape(grad, model0.n)
	grad[:, 1:21] = 0
	grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

# Bound projection
ProjBound(x) = boundproject(x,mmax,mmin)

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)
x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
x = reshape(x, model0.n)

figure(); imshow(sqrt.(1f0./x)'); title("FWI with SGD")
figure(); imshow(sqrt.(1f0./x)' - sqrt.(1f0./m0)'); title("Diff")


#########################  GS Trace ####################################################

opt = Options(normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "traces"))
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)
# Optimization parameters
srand(1)	# set seed of random number generator
# Objective function for minConf library
count = 0
function objective_function(x)
	model0.m = reshape(x,model0.n);

	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	fval, grad = fwi_objective(model0, q[i], d_obs[i]; options=opt)
	grad = reshape(grad, model0.n)
	grad[:, 1:21] = 0
	grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

x2, fsave2, funEvals2= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
x2 = reshape(x2, model0.n)

figure(); imshow(sqrt.(1f0./x2)'); title("FWI with SGD")
figure(); imshow(sqrt.(1f0./x2)' - sqrt.(1f0./m0)'); title("Diff")


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
	fval, grad = fwi_objective(model0, q[i], d_obs[i]; options=opt)
	grad = .125f0*grad/maximum(abs.(grad))  # scale for line search
	grad = reshape(grad, model0.n)
	grad[:, 1:21] = 0
	global count; count+= 1
    return fval, vec(grad)
end

x3, fsave3, funEvals3= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
x3 = reshape(x3, model0.n)

figure(); imshow(sqrt.(1f0./x3)'); title("FWI with SGD")
figure(); imshow(sqrt.(1f0./x3)' - sqrt.(1f0./m0)'); title("Diff")
