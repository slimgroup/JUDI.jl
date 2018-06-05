using JUDI.TimeModeling, JUDI.SLIM_optim, SeisIO, JLD, PyPlot

vp = segy_read("/data/mlouboutin3/Chevron2014/SEG14.Vpsmoothstarting.segy")  # IBM Float32 format [m/s]

vp = Float32.(vp.data)' / 1f3
rho = Float32.(0.31 * (1f3*vp).^0.25)
rho[vp .< 1.55] = 1.0

d = (12.5, 12.5)
n = size(vp)
o = (0., 0.)
m0 = 1./vp.^2

model0 = Model(n,d,o,1./vp.^2, rho; nb=40)

# Read datasets
container = segy_scan("/data/mlouboutin3/Chevron2014/", "Piso", ["GroupX", "GroupY", "RecGroupElevation", "dt"])
d_obs = judiVector(container; segy_depth_key="RecGroupElevation")

# read source and resample

wavelet = readdlm("/data/mlouboutin3/Chevron2014/Wavelet.txt", ',')[2:end]
dtwavelet = 2.0/3.0
wavelet = [wavelet; zeros(length(0:dtwavelet:8000) - length(wavelet), 1)]
wavelet = time_resample(wavelet, dtwavelet, Geometry(d_obs[1].geometry))

full_wavelet = copy(wavelet)
wave_low = low_filter(wavelet, 4.0; fmin=1.0, fmax=5.0)
src_geometry = Geometry(container; key = "source", segy_depth_key="SourceDepth")
q = judiVector(src_geometry, wave_low)


shot1 = d_obs.data[1][1]
shot1_filtered = low_filter(Float32.(shot1.data), 4.0; fmin=1.0, fmax=5.0)


############################### Linear operators for testing ############

ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), d_obs.nsrc, ntComp)
# Enable optimal checkpointing
opt = Options(optimal_checkpointing = false,
              limit_m = true,
              buffer_size = 1500f0,
              freesurface=true)

# Setup operators
Pr = judiProjection(info, d_obs.geometry)
F = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, src_geometry)
# q = judiVector(src_geometry, wavelet)
J = judiJacobian(Pr*F*Ps', q)

D0 = Pr[1]*F[1]*Ps[1]'*q[1]

############################### FWI ###########################################
opt = Options(limit_m=true, buffer_size=1000f0, normalize=true, freesurface=true)

# Bound projection
ProjBound(x) = boundproject(x, maximum(m0), .9*minimum(m0))

fevals = 30
batchsize = 160
options = spg_options(verbose=3, maxIter=fevals, memory=5)
# Optimization parameters
srand(1)    # set seed of random number generator


# Objective function for minConf library
count = 0
function objective_function(x)
	model0.m = reshape(x,model0.n);

	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	d_sub = get_data(d_obs[i])
	if count < 10
		println("low")
    	MF = judiFilter(d_sub.geometry, 3.0, 5.0)
		wave_low = low_filter(wavelet, 4.0; fmin=3.0, fmax=5.0)
		q = judiVector(src_geometry, wave_low)
	elseif 9 < count < 20
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
	grad[vp .< 1.55] = 0
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
save("FWI-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
save("FWI-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
save("FWI-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
# x = reshape(x, model0.n)
# figure(); imshow(sqrt.(1f0./x)', vmin=1.5, vmax=4.5, cmap="jet"); title("FWI")
# figure(); imshow(sqrt.(1f0./x)' - sqrt.(1f0./m0)'); title("Diff")
# save("FWI.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

############################### GS-FWI-shot ###########################################
opt = Options(limit_m=true, buffer_size=1000f0, freesurface=true, normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "shot")
model0 = Model(n,d,o,1./vp.^2, rho; nb=40)
srand(1)    # set seed of random number generator
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
	grad = reshape(grad, model0.n)
	grad[vp .< 1.55] = 0
	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end


# FWI with SPG
x = vec(model0.m)
x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
save("FWIgss-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
save("FWIgss-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
save("FWIgss-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)

############################### GS-FWI-trace ###########################################
opt = Options(limit_m=true, buffer_size=1000f0, freesurface=true, normalize=true, gs=Dict("maxshift" => 400.0f0, "strategy" => "trace")
model0 = Model(n,d,o,1./vp.^2, rho; nb=40)
srand(1)    # set seed of random number generator

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
	grad = reshape(grad, model0.n)
	grad[vp .< 1.55] = 0
	grad = .5f0*grad/maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

# FWI with SPG
x = vec(model0.m)
x, fsave, funEvals= minConf_SPG(objective_function, vec(model0.m), ProjBound, options)
save("FWIgst-5.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
save("FWIgst-8.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
x, fsave, funEvals= minConf_SPG(objective_function, x, ProjBound, options)
save("FWIgst-10.jld", "m0", m0, "x", reshape(x, model0.n), "fval", fsave, "funEvals", funEvals)
