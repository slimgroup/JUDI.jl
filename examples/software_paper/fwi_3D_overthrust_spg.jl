# FWI on the 3D Overthrust model using spectral projected gradient descent
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI, SlimOptim, HDF5, SegyIO

# Load overthrust model
if ~isfile("$(JUDI.JUDI_DATA)/overthrust_3D_initial_model.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_initial_model.h5")
end
n, d, o, m0 = read(h5open("$(JUDI.JUDI_DATA)/overthrust_3D_initial_model.h5", "r"), "n", "d", "o", "m0")

# Set up model structure
model0 = Model((n[1], n[2], n[3]), (d[1], d[2], d[3]), (o[1], o[2], o[3]), m0)

# Bound constraints
vmin = ones(Float32, model0.n) .+ 0.4f0
vmax = ones(Float32, model0.n) .+ 5.5f0
mmin = vec((1f0 ./ vmax).^2); vmax = []
mmax = vec((1f0 ./ vmin).^2); vmin = []

# Scan directory for segy files and create out-of-core data container
container = segy_scan("/path/to/shot/records/", "overthrust_3D_shot",
                     ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container)

# Set up source
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008)  # 8 Hz peak frequency
q = judiVector(src_geometry, wavelet)

########################################### FWI ####################################################

# Optimization parameters
fevals = 15
batchsize = 1080
count = 0
fvals = []
opt = Options(limit_m = true,   # model in only in area with receivers
              buffer_size = 500f0,  # w/ 500 m buffer zone
              optimal_checkpointing = true
              )

# Objective function for minConf library
function objective_function(x)

    # update model and save snapshot
    model0.m = reshape(x, model0.n);
    write(h5open(join(["snapshot_3D_FWI_iteration_", string(count),".h5"]), "w"), "v", sqrt.(1f0 ./ model0.m))

    # select batch
    idx = randperm(d_obs.nsrc)[1:batchsize]

    # fwi function value and gradient
    f, g = fwi_objective(model0, q[idx], d_obs[idx]; options = opt)
    g = reshape(g, model0.n); g[:, :, 1:21] .= 0f0    # reset gradient in water column to 0.
    g = .125f0*g/maximum(abs.(g))   # scale gradient to help line search

    global count; count += 1;
    global fvals; fvals = [fvals; f]
    return f, vec(g)
end

# Bound projection
ProjBound(x) = median([mmin x mmax], dims=2)

# FWI with SPG
spg_opt = spg_options(verbose = 3, maxIter = fevals, memory = 1)
x, fsave, funEvals = minConf_SPG(objective_function, vec(model0.m), ProjBound, spg_opt)

# Save results
h5open("result_3D_overthrust_spg.h5", "w") do file
    write(file, "x", sqrt.(1 ./reshape(x, model0.n)), "fsave", fsave, "fhistory", convert(Array{Float32, 1}, fvals))
end
