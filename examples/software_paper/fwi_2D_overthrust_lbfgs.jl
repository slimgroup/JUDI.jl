# FWI on the 2D Overthrust model with L-BFGS from the NLopt library
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI, NLopt, HDF5, SegyIO, PythonPlot

# Load starting model
if ~isfile("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5")
end
n, d, o, m0 = read(h5open("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5", "r"), "n", "d", "o", "m0")
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Bound constraints
vmin = 1.4f0
vmax = 6.5f0

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

############################################## FWI #################################################

# optimization parameters
fevals = 10    # allow 10 function evalutations
batchsize = 40
count = 0
fhistory = zeros(Float32, fevals+1)

# NLopt objective function
println("No.  ", "fval         ", "norm(gradient)")
function f!(x, grad)

    # Update model
    model0.m .= x

    # Select batch and calculate gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0, q[i], d_obs[i])

    # Reset gradient in water column to zero
    gradient = reshape(gradient, model0.n); gradient[:, 1:21] .= 0f0
    grad[1:end] .= gradient[1:end]

    global count; count += 1
    println(count, "    ", fval, "    ", norm(grad))
    fhistory[count] = fval
    return convert(Float64, fval)
end

# Optimization parameters
opt = Opt(:LD_LBFGS, prod(model0.n))
min_objective!(opt, f!)
mmin = (1f0 ./ vmax)^2   # bound constraints on slowness squared
mmax = (1f0 ./ vmin)^2
lower_bounds!(opt, mmin); upper_bounds!(opt, mmax)
maxeval!(opt, fevals)
(minf, minx, ret) = optimize(opt, copy(model0.m))

# Save results, function values and elapsed time
h5open("result_2D_overthrust_lbfgs.h5", "w") do file
    write(file, "x", sqrt.(1f0 ./ reshape(minx, model0.n)), "fhistory", fhistory)
end

# Plot convergence and final result
figure(); plot(fhistory/norm(fhistory, Inf));
xlabel("Iteration no."); ylabel("Normalized residual"); title("Convergence of FWI w/ L-BFGS")
figure(); imshow(sqrt.(1f0 ./ adjoint(model0.m)), extent=(0, 20.0, 5.15, 0)); title("FWI with L-BFGS");
xlabel("Lateral position [km]"); ylabel("Depth [km]")
