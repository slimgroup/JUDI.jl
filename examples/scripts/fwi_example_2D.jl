# 2D FWI on Overthrust model using minConf library
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#

using Statistics, Random, LinearAlgebra
using JUDI, SlimOptim, HDF5, SegyIO, PyPlot, SlimPlotting

# Load starting model
n,d,o,m0 = read(h5open("$(JUDI.JUDI_DATA)/overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Slowness squared [s^2/km^2]
mmin = 6.5f0.^(-2)
mmax = 1.3f0.^(-2)

# Load data
block = segy_read("$(JUDI.JUDI_DATA)/overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)  # 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

############################### FWI ###########################################
F0 = judiModeling(deepcopy(model0), src_geometry, d_obs.geometry)
M = judiTopmute(model0; taperwidth=0)

# Optimization parameters
niterations = parse(Int, get(ENV, "NITER", "10"))
batchsize = 16
fhistory_SGD = zeros(Float32, niterations)

# Projection operator for bound constraints
function proj(x)
    out = 1 * x
    out[out .< mmin] .= mmin
    out[out .> mmax] .= mmax
    return out
end

ls = BackTracking(order=3, iterations=10, )

# Main loop
for j=1:niterations

    # get fwi objective function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0, q[i], d_obs[i])
    gradient = M * gradient
    p = -.05f0*gradient/norm(gradient, Inf)
    
    println("FWI iteration no: ",j,"; function value: ",fval)
    fhistory_SGD[j] = fval

    # linesearch
    function ϕ(α)
        misfit = .5*norm(F0[i](;m=proj(model0.m .+ α * p))*q[i] - d_obs[i])^2
        @show α, misfit
        return misfit
    end
    step, fval = ls(ϕ, 1f-1, fval, dot(gradient, p))

    # Update model and bound projection
    model0.m .= proj(model0.m .+ step .* p)
end

figure(); imshow(sqrt.(1f0./adjoint(model0.m))); title("FWI with SGD")
