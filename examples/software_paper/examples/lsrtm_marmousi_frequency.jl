# Sparisty-promoting LS-RTM of the 2D Marmousi model with on-the-fly Fourier transforms
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using Statistics, Random, LinearAlgebra
using JUDI, SegyIO, HDF5, JOLI, PyPlot

# Load migration velocity model
if ~isfile("marmousi_model.h5")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_model.h5`)
end
n, d, o, m0 = read(h5open("marmousi_model.h5", "r"), "n", "d", "o", "m0")

# Set up model structure
model0 = Model((n[1], n[2]), (d[1], d[2]), (o[1], o[2]), m0)

# Load data
if ~isfile("marmousi_2D.segy")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/2DLSRTM/marmousi_2D.segy`)
end
block = segy_read("marmousi_2D.segy")
d_lin = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key = "source", segy_depth_key = "SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.03)    # 30 Hz wavelet
q = judiVector(src_geometry, wavelet)

# Set up info structure
ntComp = get_computational_nt(q.geometry, d_lin.geometry, model0)  # no. of computational time steps
info = Info(prod(model0.n), d_lin.nsrc, ntComp)

###################################################################################################

# Setup operators
opt = Options(optimal_checkpointing=false)
M = judiModeling(info, model0, q.geometry, d_lin.geometry; options=opt)
J = judiJacobian(M, q)

# Right-hand preconditioners (model topmute)
Mr = judiTopmute(model0.n, 52, 10)

# Generate distribution of frequecies from source wavelet
q_dist = generate_distribution(q)

# Linearized Bregman parameters
x = zeros(Float32, info.n)
z = zeros(Float32, info.n)
batchsize = 10
niter = 20
fval = zeros(Float32, niter)
J.options.frequencies = Array{Any}(undef, d_lin.nsrc)

# Soft thresholding functions and Curvelet transform
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64, lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32, lambda), 0f0)
C = joCurvelet2D(model0.n[1], model0.n[2]; zero_finest = true, DDT = Float32, RDT = Float64)
lambda = []
t = 1f-4

# Main loop
for j = 1: niter
    println("Iteration: ", j)

    # Select batch and set up left-hand preconditioner
    i = randperm(d_lin.nsrc)[1:batchsize]
    Ml = judiMarineTopmute2D(30, d_lin[i].geometry)    # data topmute

    # Select randomly selected batch of frequencies for each shot
    for k = 1: d_lin.nsrc
        J.options.frequencies[k] = select_frequencies(q_dist; fmin = 0.004f0, fmax = 0.05f0, nf=10)
    end

    # Compute residual and gradient
    r = Ml*J[i]*Mr*x - Ml*d_lin[i]
    g = adjoint(Mr)*adjoint(J[i])*adjoint(Ml)*r

    # Step size and update variable
    fval[j] = .5f0*norm(r)^2
    j == 1 && (global lambda = 0.5*norm(C*t*g, Inf))   # estimate thresholding parameter in 1st iteration

    # Update variables and save snapshot
    global z -= t*g
    global x = adjoint(C)*soft_thresholding(C*z, lambda)
end

# Save final velocity model, function value and history
h5open("lsrtm_marmousi_frequency_result.h5", "w") do file
    write(file, "x", reshape(x, model0.n), "fval", fval)
end

# Plot final image
figure(); imshow(copy(adjoint(reshape(x, model0.n))), extent = (0, 7.99, 3.19, 0), cmap = "gray", vmin = -3e-2, vmax = 3e-2)
title("SPLS-RTM with on on-the-fly DFTs"); xlabel("Lateral position [km]"); ylabel("Depth [km]")
