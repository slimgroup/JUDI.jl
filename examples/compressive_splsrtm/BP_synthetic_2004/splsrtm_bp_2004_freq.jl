# Least-squares RTM of the BP synthetic 2007 data set
# Author: Philipp Witte
# Date: March 2018
#

using JUDI.TimeModeling, SeisIO, JLD, PyPlot, JOLI

# Load velocity model
path = "/scratch/slim/shared/mathias-philipp/bp_synthetic_2004"
vp = load(join([path, "/model/vp_smooth_100.jld"]))["vp"] / 1f3
water_bottom = load(join([path, "/model/water_bottom_fine.jld"]))["wb"]

# Set up model structure
d = (6.25, 6.25)
o = (0., 0.)
m0 = (1f0 ./ vp).^2
n = size(m0)
model0 = Model(n, d, o, m0)

# Scan directory for segy files and create out-of-core data container
container = segy_scan(join([path, "/data_no_multiples/"]), "bp_observed_data", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = "SourceDepth")

# Set up source
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.020)  # 27 Hz peak frequency
q = judiVector(src_geometry, wavelet)

# Info structure for linear operators
ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), d_obs.nsrc, ntComp)


###################################################################################################

# Set options
opt = Options(limit_m = true,
              buffer_size = 3000f0,
              isic = true,
			  dft_subsampling_factor=8)

# Setup operators
F = judiModeling(info, model0, q.geometry, d_obs.geometry; options=opt)
J = judiJacobian(F, q)

# Right-hand preconditioners
D = judiDepthScaling(model0)
T = judiTopmute(model0.n, (1 - water_bottom), [])
Mr = D*T

# Linearized Bregman parameters
x = zeros(Float32, info.n)
z = zeros(Float32, info.n)
batchsize = 200
niter = 20
nfreq = 20
fval = zeros(Float32, niter)
q_dist = generate_distribution(q)
J.options.frequencies = Array{Any}(d_obs.nsrc)

# Soft thresholding functions and Curvelet transform
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) - convert(Float64,lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) - convert(Float32,lambda), 0f0)
C = joCurvelet2D(model0.n[1], model0.n[2]; zero_finest=true, DDT=Float32, RDT=Float64)
lambda = []
t = []  # t = 1f-4

# Main loop
for j=1:niter
    println("Iteration: ", j)

   # Set randomized frequencies
    for k=1:d_obs.nsrc
        J.options.frequencies[k] = select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=nfreq)
    end

    # Select batch and set up left-hand preconditioner
    i = randperm(d_obs.nsrc)[1:batchsize]
    d_sub = get_data(d_obs[i])
    Ml = judiMarineTopmute2D(35, d_sub.geometry)

    # Compute residual and estimate source
    if j > 1
        d_pred = J[i]*Mr*x
        r = Ml*d_pred - Ml*d_sub
    else
        r = Ml*d_sub*(-1f0)    # skip forward modeling in first iteration
    end
    
    # Residual and gradient
    g = Mr'*J[i]'*Ml'*r

    # Step size and update variable
    fval[j] = .5*norm(r)^2
    t = norm(r)^2/norm(g)^2 # divide by 10
    println("    Stepsize: ", t)

    j==1 && (lambda = 0.03*norm(C*t*g, Inf))   # estimate thresholding parameter in 1st iteration

    # Update variables
    z -= t*g
    x = C'*soft_thresholding(C*z, lambda)

    # Save snapshot
    save(join([path, "/results/splsrtm_freq_iteration_", string(j), ".jld"]), "x", reshape(x, model0.n), "z", reshape(z, model0.n), "t", t, "lambda", lambda, "fval", fval[j])
end



