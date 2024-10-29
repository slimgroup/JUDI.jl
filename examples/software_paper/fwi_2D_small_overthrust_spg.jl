# 2D FWI on a small version of the Overthrust model with SPG from the minConf library
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI, SlimOptim, HDF5, SegyIO, PythonPlot

# Load starting model
n,d,o,m0 = read(h5open("$(JUDI.JUDI_DATA)/small_overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)
vmin = ones(Float32,model0.n) .* 1.3f0
vmax = ones(Float32,model0.n) .* 6.5f0
vmin[:,1:21] .= v0[:,1:21]   # keep water column fixed
vmax[:,1:21] .= v0[:,1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data
block = segy_read("$(JUDI.JUDI_DATA)/small_overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)  # 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

############################### FWI ###########################################

# Optimization parameters
fevals = 10
batchsize = 8
fvals = []

# Objective function for minConf library
function objective_function(x)
    model0.m .= x

    # fwi function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, grad = fwi_objective(model0, q[i], d_obs[i])
    grad = .125f0*grad/norm(grad, Inf)  # scale for line search

    global fvals; fvals = [fvals; fval]
    return fval, vec(grad.data)
end

# Bound projection
ProjBound(x) = median([mmin x mmax]; dims=2)[1:end]

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)
x, fsave, funEvals= spg(objective_function, vec(m0), ProjBound, options)

# Save results
h5open("result_2D_small_overthrust_spg.h5", "w") do file
    write(file, "x", sqrt.(1f0 ./ reshape(x, model0.n)), "fsave", fsave, "fhistory", convert(Array{Float32, 1}, fvals))
end

# Plot convergence and final result
figure(); plot(fvals/norm(fvals, Inf));
xlabel("Iteration no."); ylabel("Normalized residual"); title("Convergence of FWI w/ SPG")
figure(); imshow(sqrt.(1f0 ./ adjoint(reshape(x, model0.n))), extent=(0, 20.0, 5.15, 0)); title("FWI with SPG");
xlabel("Lateral position [km]"); ylabel("Depth [km]")
