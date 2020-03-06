# 2D FWI on a small version of the Overthrust model with L-BFGS from the NLopt library
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI.TimeModeling, HDF5, NLopt, SegyIO, PyPlot

# Load starting model
n,d,o,m0 = read(h5open("../data/small_overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)
vmin = ones(Float32, model0.n) .* 1.3f0
vmax = ones(Float32, model0.n) .* 6.5f0

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data
block = segy_read("../data/small_overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008f0)    # 8 Hz wavelet
q = judiVector(src_geometry, wavelet)

############################### FWI ###########################################

# optimization parameters
fevals = 10
batchsize = 16
count = 0
fhistory = zeros(Float32, fevals+1)

# NLopt objective function
println("No.  ", "fval         ", "norm(gradient)")
function f!(x,grad)

    # Update model
    model0.m = convert(Array{Float32, 2}, reshape(x, model0.n))

    # Seclect batch and calculate gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0, q[i], d_obs[i])

    # Reset gradient in water column to zero
    gradient = reshape(gradient, model0.n); gradient[:, 1:21] .= 0f0
    grad[1:end] = vec(gradient)

    global count; count += 1
    println(count, "    ", fval, "    ", norm(grad))
    fhistory[count] = fval
    return convert(Float64, fval)
end

# Optimization parameters
opt = Opt(:LD_LBFGS, prod(model0.n))
lower_bounds!(opt, mmin); upper_bounds!(opt, mmax)
min_objective!(opt, f!)
maxeval!(opt, fevals)
(minf, minx, ret) = optimize(opt, vec(model0.m))

# Save results, function values and elapsed time
h5open("result_2D_small_overthrust_lbfgs.h5", "w") do file
    write(file, "x", sqrt.(1f0 ./ reshape(minx, model0.n)), "fhistory", fhistory)
end

# Plot convergence and final result
figure(); plot(fhistory/norm(fhistory, Inf));
xlabel("Iteration no."); ylabel("Normalized residual"); title("Convergence of FWI w/ L-BFGS")
figure(); imshow(sqrt.(1f0./adjoint(model0.m)), extent=(0, 20.0, 5.15, 0)); title("FWI with L-BFGS");
xlabel("Lateral position [km]"); ylabel("Depth [km]")
