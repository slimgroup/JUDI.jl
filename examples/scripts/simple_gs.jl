# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, JUDI.SLIM_optim, PyPlot, Random

## Set up model structure
n = (201,201)   # (x,y,z) or (x,z)
d = (25.,25.)
o = (0.,0.)

# Velocity [km/s]
v0 = ones(Float32,n) .+ 1.0f0

x = Float32.(range(0.0f0, stop=5000f0, length=201))

v = 2.0f0 .- .6f0*exp.(-(x .- 2500.0f0).^2/600f0^2)*exp.(-(x .- 2500.0f0).^2/600f0^2)';

# Slowness squared [s^2/km^2]
m = (1 ./v).^2
m0 = (1 ./v0).^2
# Slowness squared [s^2/km^2]
mmin = (1f0 ./v).^2
mmax = (1f0 ./v).^2
# Setup info and model structure
model = Model(n,d,o,m)
model0 = Model(n,d,o,m0)

## Set up receiver geometry
nsrc = 51
nxrec = 201
xrec = range(4950f0, stop=4950f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=4950f0, length=nxrec)

# receiver sampling and recording time
timeR = 5000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(50f0, stop=50f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(50f0, stop=4950f0, length=nsrc))

# source sampling and number of time steps
timeS = 5000f0  # ms
dtS = 4f0   # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.005f0     # MHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)


# Write shots as segy files to disk
opt = Options()

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
Ps = judiProjection(info, srcGeometry)


d_obs = Pr * F * Ps' * q

############################### FWI ###########################################
opt = Options(normalized="trace")

# Bound projection
ProjBound(x) = boundproject(x, maximum(m), .9*minimum(m))

fevals = 20
batchsize = 25
fmin = 1.0
# Optimization parameters


# Objective function for minConf library
count = 0
function objective_function(x, opt)
	model0.m = reshape(x,model0.n);
	# fwi function value and gradient
	i = randperm(d_obs.nsrc)[1:batchsize]
	d_sub = d_obs[i]
	fval, grad = fwi_objective(model0, q[i], d_sub; options=opt)
	grad = reshape(grad, model0.n)
	grad[1:15, :] .= 0
	grad[end-15:end, :] .= 0
	grad = .5f0 .* grad ./maximum(abs.(grad))  # scale for line search

	global count; count+= 1
    return fval, vec(grad)
end

# FWI with SPG
# options = pqn_options(verbose=3, maxIter=fevals, corrections=10)
options = spg_options(verbose=3, maxIter=fevals, memory=5)
fmax = 5.0
############################### FWI-shot-n ###########################################
opt = Options(normalized="shot")
# Objective function for minConf library


# FWI with SPG
x = vec(m0)
fwi_obj(x) = objective_function(x, opt)
# ffwis, gfwis = fwi_obj(x)
# save("first_gradgss.jld", "m0", m0, "g", reshape(g, model0.n))
xs, fsave, funEvals= minConf_SPG(fwi_obj, vec(x), ProjBound, options)
# save("FWIs.jld", "m0", m0, "x", reshape(xs, model0.n), "fval", fsave, "funEvals", funEvals)

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

############################### FWI-trace-n ###########################################
opt = Options(normalized="trace")
# Objective function for minConf library


# FWI with SPG
x = vec(m0)
fwi_obj(x) = objective_function(x, opt)
# ffwit, gfwit = fwi_obj(x)
# save("first_gradgss.jld", "m0", m0, "g", reshape(g, model0.n))
# xt, fsave, funEvals= minConf_SPG(fwi_obj, vec(x), ProjBound, options)
# save("FWIt.jld", "m0", m0, "x", reshape(xt, model0.n), "fval", fsave, "funEvals", funEvals)

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

############################### GS-FWI-shot ###########################################
opt = Options(normalized="shot", gs=Dict("maxshift" => 400.0f0, "strategy" => "shot"))
# Objective function for minConf library


# FWI with SPG
x = vec(m0)
fwi_obj(x) = objective_function(x, opt)
# fgss, ggss = fwi_obj(x)
# save("first_gradgss.jld", "m0", m0, "g", reshape(g, model0.n))
xgs, fsave, funEvals= minConf_SPG(fwi_obj, vec(x), ProjBound, options)
# save("FWIt.jld", "m0", m0, "x", reshape(xgs, model0.n), "fval", fsave, "funEvals", funEvals)

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
opt = Options(normalized="trace", gs=Dict("maxshift" => 400.0f0, "strategy" => "trace"))


# FWI with SPG
x = vec(m0)
# fmax=5.0
fwi_obj(x) = objective_function(x, opt)
# fgst, ggst = fwi_obj(x)
# save("first_gradgst.jld", "m0", m0, "g", reshape(g, model0.n))
xgt, fsave, funEvals= minConf_SPG(fwi_obj, vec(x), ProjBound, options)
# save("FWIt.jld", "m0", m0, "x", reshape(xgt, model0.n), "fval", fsave, "funEvals", funEvals)
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


figure();imshow(reshape(sqrt.(1 ./xs), model0.n)', vmin=1.5, vmax=2.0, cmap="seismic")
# figure();imshow(reshape(sqrt.(1./xt), model0.n)', vmin=1.5, vmax=2.0 cmap="seismic")
figure();imshow(reshape(sqrt.(1 ./xgs), model0.n)', vmin=1.5, vmax=2.0, cmap="seismic")
figure();imshow(reshape(sqrt.(1 ./xgt), model0.n)', vmin=1.5, vmax=2.0, cmap="seismic")
# figure();imshow(reshape(v, model0.n)', vmin=1.5, vmax=2.0, cmap="seismic")
