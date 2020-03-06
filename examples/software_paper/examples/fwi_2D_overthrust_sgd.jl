# FWI on the 2D Overthrust model with stochastic gradient descent
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SegyIO, PyPlot

# Load starting model
if ~isfile("overthrust_model_2D.h5")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5`)
end
n, d, o, m0 = read(h5open("overthrust_model_2D.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)
vmin = ones(Float32, model0.n) .* 1.3f0
vmax = ones(Float32, model0.n) .* 6.5f0
vmin[:, 1:21] .= v0[:, 1:21]   # keep water column fixed
vmax[:, 1:21] .= v0[:, 1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0 ./ vmax).^2)
mmax = vec((1f0 ./ vmin).^2)

# Load data and create data vector
if ~isfile("overthrust_2D.segy")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_2D.segy`)
end
block = segy_read("overthrust_2D.segy")
d_obs = judiVector(block)

# Set up wavelet and source vector
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.008f0) # 8 Hz wavelet
q = judiVector(src_geometry, wavelet)

########################################### FWI ####################################################

# Optimization parameters
maxiter = 20
batchsize = 20
fhistory_SGD = zeros(Float32, maxiter)
fTerm = 1e3
gTerm = 1e1

# Projection operator for bound constraints
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)], dims=2), model0.n)

# Define misfit function
function fwi_misfit(model::Model, q::judiVector, d::judiVector; misfit = "L2", compute_gradient = true)

    # Set up operators
    nt = get_computational_nt(q.geometry, d.geometry, model)
    info = Info(prod(model.n), d.nsrc, nt)
    M = judiModeling(info, model, q.geometry, d.geometry)
    J = judiJacobian(M, q)

    # Data residual, function value and gradient
    if misfit == "L2"
        r = M*q - d
        f = .5f0*norm(r)^2
        compute_gradient == true && (g = adjoint(J)*r)    # gradient not necessary for line search
    elseif misifit == "huber"
        r = M*q - d
        f = eps^2*sqrt(1f0 + dot(r, r)/eps^2) - eps^2
        compute_gradient == true && (g = adjoint(J)*r/sqrt(1f0 + dot(r, r)/eps^2))
    else
        throw("Wrong misfit method specified")
    end
    if compute_gradient == true
        return f, g
    else
        return f
    end
end

# Main loop
for j = 1: maxiter

    # select current subset of shots
    i = randperm(d_obs.nsrc)[1:batchsize]
    f, g = fwi_misfit(model0, q[i], d_obs[i])
    println("FWI iteration no: ", j, "; function value: ", f)
    fhistory_SGD[j] = f

    # linesearch
    step = backtracking_linesearch(model0, q[i], d_obs[i], f, g,proj, fwi_misfit; alpha=1f0)

    # Update model and bound projection
    model0.m = proj(model0.m + reshape(step, model0.n))

    if f <= fTerm || norm(g) <= gTerm
        break
    end
end

# Save results
h5open("result_2D_overthrust_sgd.h5", "w") do file
    write(file, "x", sqrt.(1f0 ./ reshape(model0.m, model0.n)), "fhistory", fhistory_SGD)
end

# Plot convergence and final result
figure(); plot(1:maxiter, fhistory_SGD/norm(fhistory_SGD, Inf));
xlabel("Iteration no."); ylabel("Normalized residual"); title("Convergence of FWI w/ SGD")
figure(); imshow(sqrt.(1f0./adjoint(model0.m)), extent=(0, 20.0, 5.15, 0)); title("FWI with SGD");
xlabel("Lateral position [km]"); ylabel("Depth [km]")
