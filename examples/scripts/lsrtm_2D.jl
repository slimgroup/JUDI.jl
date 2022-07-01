# LS-RTM of the 2D Marmousi model using LSQR
# Author: ziyi.yin@gatech.edu
# Date: June 2022

using Statistics, Random, LinearAlgebra, JOLI
using JUDI, SegyIO, HDF5, PyPlot, IterativeSolvers


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

#' LSQR
niter = parse(Int, get(ENV, "NITER", "10"))
lsqr_sol = zeros(Float32, prod(n))

# only invert for the randomly picked indices so that this example can run a bit faster
# suggested to run with all indices in practice
idx = randperm(q.nsrc)[1:4]
Ml = judiMarineTopmute2D(30, d_lin[idx].geometry)
lsqr!(lsqr_sol, Ml*J[idx]*Mr, Ml*d_lin[idx]; maxiter=niter)

# Save final velocity model, function value and history
h5open("lsrtm_marmousi_lsqr_result.h5", "w") do file
    write(file, "x", reshape(lsqr_sol, model0.n))
end

# Plot final image
figure(); imshow(copy(adjoint(reshape(lsqr_sol, model0.n))), extent = (0, 7.99, 3.19, 0), cmap = "gray", vmin = -3e-2, vmax = 3e-2)
title("LS-RTM with LSQR"); xlabel("Lateral position [km]"); ylabel("Depth [km]")
