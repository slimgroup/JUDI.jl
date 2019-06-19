# Sparsity-promoting LS-RTM on the Sigsbee 2A model w/ on-the-fly Fourier transforms
# Author: Philipp Witte, pwitte.slim@gmail.com
# Date: May 2018
#

# TO DO:
# Replace w/ full path the observed data directory
#path_to_data = "/path/to/directory/"
path_to_data="/home/pwitte3/.julia/dev/JUDI/examples/compressive_splsrtm/Sigsbee2A/"
data_name = "sigsbee2A_marine"  # common base name of all shots

using Pkg; Pkg.activate("JUDI")
using JUDI.TimeModeling, PyPlot, JLD, SeisIO, JOLI, Random, LinearAlgebra

# Load Sigsbee model
M = load("sigsbee2A_model.jld")

# Setup info and model structure
model0 = Model(M["n"], M["d"], M["o"], M["m0"])
dm = vec(M["dm"])

# Load data
container = segy_scan(path_to_data, data_name, ["GroupX","GroupY","RecGroupElevation","SourceSurfaceElevation","dt"])
d_lin = judiVector(container)

# Set up source
src_geometry = Geometry(container; key="source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.015)  # 15 Hz peak frequency
q = judiVector(src_geometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(q.geometry,d_lin.geometry, model0)
info = Info(prod(model0.n), d_lin.nsrc, ntComp)


#################################################################################################

opt = Options(isic=true)

# Setup operators
Pr = judiProjection(info, d_lin.geometry)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, src_geometry)
J = judiJacobian(Pr*F0*Ps', q)

# Right-hand preconditioners (model topmute)
idx_wb = find_water_bottom(reshape(dm, model0.n))
Tm = judiTopmute(model0.n, idx_wb, 10)  # Mute water column
S = judiDepthScaling(model0)
Mr = S*Tm

# Linearized Bregman parameters
x = zeros(Float32, info.n)
z = zeros(Float32, info.n)
batchsize = 100
niter = 20
nfreq = 20
fval = zeros(Float32, niter)
q_dist = generate_distribution(q)
J.options.frequencies = Array{Any}(undef, d_lin.nsrc)
J.options.dft_subsampling_factor = 8

# Soft thresholding functions and Curvelet transform
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64, lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32, lambda), 0f0)
C = joCurvelet2D(model0.n[1], model0.n[2]; zero_finest=true, DDT=Float32, RDT=Float64)
lambda = []
t = 2f-5    # 4f-5 for nfreq=10

# Main loop
for j=1:niter
    println("Iteration: ", j)

    # Set randomized frequencies
    for k=1:d_lin.nsrc
        J.options.frequencies[k] = select_frequencies(q_dist; fmin=0.002, fmax=0.04, nf=nfreq)
    end

    # Compute residual and gradient
    i = randperm(d_lin.nsrc)[1:batchsize]
    d_sub = get_data(d_lin[i])    # load current shots into memory
    r = J[i]*Mr*x - d_sub
    g = Mr'*J[i]'*r

    # Step size and update variable
    fval[j] = .5*norm(r)^2
    j==1 && (global lambda = 0.05*norm(C*t*g, Inf))   # estimate thresholding parameter in 1st iteration

    # Update variables
    global z -= t*g
    global x = adjoint(C)*soft_thresholding(C*z, lambda)

    # Save snapshot
    save(join(["snapshot_splsrtm_fourier_iteration_", string(j),".jld"]), "x", reshape(x, model0.n), "z", reshape(z, model0.n))
end

# Save final result
save("sigsbee2A_splsrtm_frequency_domain.jld", "x", x, "fval", fval, "lambda", lambda)
