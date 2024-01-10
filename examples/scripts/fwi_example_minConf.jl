# 2D FWI on Overthrust model with SPG using minConf library
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#

using Statistics, Random, LinearAlgebra, PyPlot
using JUDI, SlimOptim, HDF5, SegyIO

# Load starting model
n,d,o,m0 = read(h5open("$(JUDI.JUDI_DATA)/overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ m0)
vmin = ones(Float32,model0.n) .* 1.3f0
vmax = ones(Float32,model0.n) .* 6.5f0
vmin[:,1:21] .= v0[:,1:21]   # keep water column fixed
vmax[:,1:21] .= v0[:,1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data
block = segy_read("$(JUDI.JUDI_DATA)/overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)  # 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

############################### FWI ###########################################
M = judiTopmute(model0; taperwidth=0)

# Optimization parameters
fevals = parse(Int, get(ENV, "NITER", "10"))
batchsize = 8

# Objective function for minConf library
count = 0

function objective_function(x, model0, d_obs, q, M, batchsize)
    model0.m .= reshape(x, model0.n);

    # fwi function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, grad = fwi_objective(model0, q[i], d_obs[i])
    grad = M * grad
    grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

    global count; count+= 1

    return fval, grad
end

phi(x) = objective_function(x, model0, d_obs, q, M, batchsize)

# Bound projection
proj(x::AbstractArray{T, N}) where {T, N} = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2), model0.n)

# FWI with SPG
m0 = model0.m
options = spg_options(verbose=3, maxIter=fevals, memory=5)
sol = spg(phi, m0, proj, options)

# Plot result
imshow(reshape(sqrt.(1f0 ./ sol.x), model0.n)', extent=[0, 10, 3, 0])
xlabel("Lateral position [km]")
ylabel("Depth [km]")
