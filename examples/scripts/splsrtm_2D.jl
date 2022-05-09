# LS-RTM of the 2D Marmousi model using linearized bregmann
# Author: mlouboutin3@gatech.edu
# Date: April 2022

using Statistics, Random, LinearAlgebra, JOLI
using JUDI, SegyIO, HDF5, PyPlot, SlimOptim


# Load migration velocity model
if ~isfile("$(JUDI.JUDI_DATA)/marmousi_model.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_model.h5")
end
n, d, o, m0 = read(h5open("$(JUDI.JUDI_DATA)/marmousi_model.h5", "r"), "n", "d", "o", "m0")

# Set up model structure
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Load data
if ~isfile("$(JUDI.JUDI_DATA)/marmousi_2D.segy")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_2D.segy")
end
block = segy_read("$(JUDI.JUDI_DATA)/marmousi_2D.segy")
d_lin = judiVector(block)   # linearized observed data

# Set up wavelet
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.03)    # 30 Hz wavelet
q = judiVector(src_geometry, wavelet)

###################################################################################################
# Infer subsampling based on free memory
mem = Sys.free_memory()/(1024^3)
t_sub = max(1, ceil(Int, 40/mem))
# Setup operators
opt = Options(subsampling_factor=t_sub, isic=true)  # ~40 GB of memory per source without subsampling
M = judiModeling(model0, q.geometry, d_lin.geometry; options=opt)
J = judiJacobian(M, q)

# Right-hand preconditioners (model topmute)
Mr = judiTopmute(model0.n, 52, 10)
# Sparsity
C = joEye(prod(model0.n); DDT=Float32, RDT=Float32)
# If available use curvelet instead for better result

# Setup linearized bregman
batchsize = 5 * parse(Int, get(ENV, "NITER", "$(q.nsrc ÷ 5)"))
niter = parse(Int, get(ENV, "NITER", "10"))
g_scale = 0

function obj(x)
    flush(stdout)
    dm = PhysicalParameter(x, model0.n, model0.d, model0.o)
    inds = randperm(q.nsrc)[1:batchsize]
    # Preconditionners
    Ml = judiMarineTopmute2D(30, d_lin[inds].geometry)

    residual = Ml*J[inds]*Mr*dm - Ml*d_lin[inds]
    # grad
    G = reshape(Mr'J[inds]'*Ml'*residual, model0.n)
    g_scale == 0 && (global g_scale = .05f0/maximum(G))
    G .*= g_scale
    return .5f0*norm(residual)^2, G[:]
end

# Bregman
bregopt = bregman_options(maxIter=niter, verbose=2, quantile=.9, alpha=.1, antichatter=false, spg=true)
solb = bregman(obj, zeros(Float32, prod(model0.n)), bregopt, C);


# Save final velocity model, function value and history
h5open("lsrtm_marmousi_breg_result.h5", "w") do file
    write(file, "x", reshape(solb.x, model0.n), "z", reshape(solb.z, model0.n), "fval", Float32.(solb.ϕ_trace))
end

# Plot final image
figure(); imshow(copy(adjoint(reshape(solb.x, model0.n))), extent = (0, 7.99, 3.19, 0), cmap = "gray", vmin = -3e-2, vmax = 3e-2)
title("LS-RTM with SGD"); xlabel("Lateral position [km]"); ylabel("Depth [km]")
