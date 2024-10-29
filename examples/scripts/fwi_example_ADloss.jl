# 2D Envelope FWI on Overthrust model with SPG using minConf library and Zygote for the derivative of the Envelope misfit
# Author: Mathias Louboutin
# Date: September 2022
#

using Statistics, Random, LinearAlgebra, PythonPlot, Distributed
using JUDI, SlimOptim, HDF5, SegyIO
@everywhere using JUDI.FFTW, Zygote

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

@everywhere function H(x)
    n = size(x, 1)
    σ = ifftshift(sign.(-n/2+1:n/2))
    y = imag(ifft(σ.*fft(x, 1), 1))
    return y
end

@everywhere envelope(x, y) = sum(abs2.((x - y) .+ 1im .* H(x - y)))
@everywhere denvelope(x, y) = gradient(xs->envelope(xs, y), x)[1]
@everywhere myloss(x, y)= (envelope(x, y), denvelope(x, y))

@everywhere myloss(randn(Float32, 10, 10), randn(Float32, 10, 10))

# Optimization parameters
fevals = parse(Int, get(ENV, "NITER", "10"))
batchsize = 8

# Objective function for minConf library
count = 0
function objective_function(x)
    model0.m .= reshape(x,model0.n);

    # fwi function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, grad = fwi_objective(model0, q[i], d_obs[i]; misfit=myloss)
    grad = .125f0*grad/maximum(abs.(grad))  # scale for line search

    global count; count+= 1
    return fval, grad
end

# Bound projection
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)]; dims=2),model0.n)

# FWI with SPG
options = spg_options(verbose=3, maxIter=fevals, memory=3)
sol = spg(objective_function, model0.m, proj, options)

# Plot result
imshow(reshape(sqrt.(1f0 ./ sol.x), model0.n)', extent=[0, 10, 3, 0])
xlabel("Lateral position [km]")
ylabel("Depth [km]")
