using JUDI.TimeModeling, JUDI.SLIM_optim, SeisIO, JLD, PyPlot, DelimitedFiles, Random

vp = segy_read("/Users/mathiaslouboutin//data/ChevronSEG2014/SEG14.Vpsmoothstarting.segy")  # IBM Float32 format [m/s]

vp = Float32.(vp.data)[1:5:end, 1:2:end] ./ 1f3
# Pad for sources outside
last = vp[:, end]
vp = hcat(vp, last*ones(Float32, 1, 50))'

rho = Float32.(0.31 * (1f3*vp).^0.25)
rho[vp .< 1.55] .= 1.0

d = (25, 25)
n = size(vp)
o = (0., 0.)
m0 = 1f0 ./ vp.^2

model0 = Model(n, d, o, m0; nb=40, rho=rho)

# Read datasets
container = segy_scan("/Users/mathiaslouboutin/data/ChevronSEG2014/", "Piso", ["GroupX", "GroupY", "RecGroupElevation", "dt"])
d_obs = judiVector(container; segy_depth_key="RecGroupElevation")

# read source and resample

wavelet = readdlm("/Users/mathiaslouboutin/data/ChevronSEG2014/Wavelet.txt", ',')[2:end]
dtwavelet = 2.0/3.0
wavelet = [wavelet; zeros(length(0:dtwavelet:8000) - length(wavelet), 1)]
wavelet = time_resample(wavelet, dtwavelet, Geometry(d_obs[1].geometry))

full_wavelet = copy(wavelet)
wave_low = low_filter(wavelet, 4.0; fmin=.1, fmax=4.0)
src_geometry = Geometry(container; key = "source", segy_depth_key="SourceDepth")
q = judiVector(src_geometry, wave_low)


shot1 = d_obs.data[1][1]
shot1_filtered = low_filter(Float32.(shot1.data), 4.0; fmin=.1, fmax=4.0)


############################### Linear operators for testing ############
#
ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), d_obs.nsrc, ntComp)
# Enable optimal checkpointing
opt = Options(optimal_checkpointing = false,
              limit_m = true,
              buffer_size = 1000f0,
              free_surface=true,
			  normalize=true,
			  space_order=12)
#
# Setup operators
Pr = judiProjection(info, d_obs.geometry)
F = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, src_geometry)
# q = judiVector(src_geometry, wavelet)
J = judiJacobian(Pr*F*Ps', q)

# D0 = Pr[1]*F[1]*Ps[1]'*q[1]
#
# D0.data[1] = D0.data[1]/maximum(D0.data[1]) - shot1_filtered/maximum(shot1_filtered)
#
# g1 = adjoint(J[1])*D0

############################### FWI ###########################################

# Bound projection
ProjBound(x) = boundproject(x, maximum(m0), .9*minimum(m0))

fevals = 10
batchsize = 80
fmin = 1.0
options = spg_options(verbose=3, maxIter=fevals, memory=1)
# Optimization parameters
srand(1)    # set seed of random number generator


# Objective function for minConf library
count = 0
function objective_function(x, fmin, fmax, opt)
	model0.m = reshape(x,model0.n);
	println("Frequency band: ", fmin, " ,", fmax)
	println("Gradient sampling: ", ~isempty(opt.gs))
	# fwi function value and gradient
	# i = randperm(d_obs.nsrc)[1:batchsize]
	i=1:batchsize
	d_sub = get_data(d_obs[i])
	d_sub = low_filter(d_sub, 4.0; fmin=fmin, fmax=fmax)
	wave_low = low_filter(wavelet, 4.0; fmin=fmin, fmax=fmax)
	q = judiVector(src_geometry, wave_low)
	fval, grad = fwi_objective(model0, q[i], d_sub; options=opt)
	grad = reshape(grad, model0.n)
	grad[vp .< 1.55] = 0
	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=1)
x = vec(m0)
fmin=0.1
fmax=5.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
f, g = fwi_obj(x)
# save("first_grad.jld", "m0", m0, "g", reshape(g, model0.n))
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWI-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
#
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
opt = Options(limit_m=true, buffer_size=1000f0, free_surface=true, normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "shot"))
srand(1)    # set seed of random number generator
# Objective function for minConf library


# FWI with SPG
# x = vec(m0)
fmin=0.1
fmax=5.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
f2, g2 = fwi_obj(x)
# # save("first_gradgss.jld", "m0", m0, "g", reshape(g, model0.n))
# x, fsave, funEvals= minConf_SPG(fwi_obj, vec(x), ProjBound, options)
# save("FWIgss-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
#
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
opt = Options(limit_m=true, buffer_size=1000f0, free_surface=true, normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "trace"))
srand(1)    # set seed of random number generator


# FWI with SPG
x = vec(m0)
fmin=0.1
fmax=5.0
fwi_obj(x) = objective_function(x, fmin, fmax, opt)
f3, g3 = fwi_obj(x)
# save("first_gradgst.jld", "m0", m0, "g", reshape(g, model0.n))
# x, fsave, funEvals= minConf_SPG(fwi_obj, x, ProjBound, options)
# save("FWIgst-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
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
