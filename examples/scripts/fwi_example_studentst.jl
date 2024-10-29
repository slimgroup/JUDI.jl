# 2D FWI with student's t misfit on Overthrust model with SPG using minConf library
# Author: Mathias Louboutin
# Date: September 2022
#

using Statistics, Random, LinearAlgebra, PythonPlot, SlimPlotting
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


## Add outliers to the data
for s=1:d_obs.nsrc
    # wrongly scale a few traces
    nrec = d_obs[s].geometry.nrec[1]
    inds = rand(1:nrec, 10)
    d_obs.data[s][:, inds] .*= 20
end

############################### FWI ###########################################


# Optimization parameters
fevals = parse(Int, get(ENV, "NITER", "10"))
batchsize = 8

# Objective function for minConf library
count = 0
function objective_function(x, misfit=mse)
    model0.m .= reshape(x,model0.n);

    # fwi function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, grad = fwi_objective(model0, q[i], d_obs[i]; misfit=misfit)
    grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

    global count; count+= 1
    return fval, vec(grad.data)
end

# Bound projection
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2), size(x))

# Compare l2 with students t
ϕmse = x->objective_function(x)
ϕst = x->objective_function(x, studentst)

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)
solmse = spg(ϕmse, vec(m0), proj, options)
solst = spg(ϕst, vec(m0), proj, options)

# Plot result
figure(figsize=(10, 10))
subplot(211)
plot_velocity(reshape(solmse.x.^(-.5), model0.n)', d; name="MSE", new_fig=false)
subplot(212)
plot_velocity(reshape(solst.x.^(-.5), model0.n)', d; name="Student's t", new_fig=false)
