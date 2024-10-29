# FWI on the 2D Overthrust model using spectral projected gradient descent
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI, SlimOptim, HDF5, SegyIO, PythonPlot

# Load starting model
if ~isfile("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5")
end
n, d, o, m0 = read(h5open("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5", "r"), "n", "d", "o", "m0")
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)
vmin = ones(Float32,model0.n) .* 1.3f0
vmax = ones(Float32,model0.n) .* 6.5f0
vmin[:, 1:21] .= v0[:, 1:21]   # keep water column fixed
vmax[:, 1:21] .= v0[:, 1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data and create data vector
if ~isfile("$(JUDI.JUDI_DATA)/overthrust_2D.segy")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D.segy")
end
block = segy_read("$(JUDI.JUDI_DATA)/overthrust_2D.segy")
d_obs = judiVector(block)

# Set up wavelet and source vector
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008f0) # 8 Hz wavelet
q = judiVector(src_geometry, wavelet)

########################################### FWI ####################################################

# Optimization parameters
fevals = 20
batchsize = 20
fvals = []

# Objective function for library
function objective_function(x)
    model0.m .= x;

    # select batch          "elapsed_time", elapsed_time
    idx = randperm(d_obs.nsrc)[1:batchsize]
    f, g = fwi_objective(model0, q[idx], d_obs[idx])

    global fvals; fvals = [fvals; f]
    return f, vec(g.data/norm(g, Inf))    # normalize gradient for line search
end

# Bound projection
ProjBound(x) = median([mmin x mmax], dims=2)[1:end]

# FWI with SPG
options = spg_options(verbose = 3, maxIter = fevals, memory = 3, iniStep = 1f0)
x, fsave, funEvals = spg(objective_function, vec(m0), ProjBound, options)

# Save results
h5open("result_2D_overthrust_spg.h5", "w") do file
    write(file, "x", sqrt.(1f0 ./ reshape(x, model0.n)), "fsave", fsave, "fhistory", convert(Array{Float32, 1}, fvals))
end

# Plot convergence and final result
figure(); plot(fvals/norm(fvals, Inf));
xlabel("Iteration no."); ylabel("Normalized residual"); title("Convergence of FWI w/ SPG")
figure(); imshow(sqrt.(1f0 ./ adjoint(reshape(x, model0.n))), extent=(0, 20.0, 5.15, 0)); title("FWI with SPG");
xlabel("Lateral position [km]"); ylabel("Depth [km]")
