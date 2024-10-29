# LS-RTM of the 2D Marmousi model using elastic average stochastic gradient descent
# Author: pwitte.slim@gmail.com
# Date: December 2018
#
# Warning: The examples requires ~40 GB of memory per shot if used without optimal checkpointing.
#

using Statistics, Random, LinearAlgebra, Distributed
using JUDI, SegyIO, HDF5, PythonPlot

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
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.03) # 30 Hz wavelet
q = judiVector(src_geometry, wavelet)

###################################################################################################

# Setup operators
opt = Options(optimal_checkpointing=true)  # ~40 GB of memory per source w/o checkpointing
M = judiModeling(model0, q.geometry, d_lin.geometry;options=opt)
J = judiJacobian(M, q)

# Right-hand preconditioners (model topmute)
Mr = judiTopmute(model0; taperwidth=10)

# Left preconditioner
Ml = judiDataMute(q.geometry, d_lin.geometry)    # data topmute

# Elastic average stochastic gradient descent
niter = 20
p = 10
batchsize = 1

eta = 0.03f0
rho = 1f0
alpha = eta*rho
beta = p*alpha

x = zeros(Float32, prod(model0.n), p)
xnew = zeros(Float32, prod(model0.n), p)
xav = zeros(Float32, prod(model0.n))

# Parallel gradient function
@everywhere function update_x(Ml, J, Mr, x, d, eta, alpha, xav)

    # gradient
    r = Ml*J*Mr*x - Ml*d
    g = adjoint(Mr)*adjoint(J)*adjoint(Ml)*r

    # Update variable
    return x - eta*g - alpha*(x - xav)
end
update_x_par = remote(update_x)     # parallel function instance

# Main loop
for j = 1: niter
    println("Iteration: ", j)

    @sync begin
        for k = 1: p
            # Select batch
            batchsize == 1 ? i = randperm(d_lin.nsrc)[1] : i = randperm(d_lin.nsrc)[1:batchsize]

            # Calculate x update
            @async xnew[:, k] = update_x_par(Ml[i], J[i], Mr, x[:,k], d_lin[i], eta, alpha, xav)
        end
    end

    global xav = (1-beta)*xav + beta*(1/p *sum(x, dims=2))
    global x = copy(xnew)
end

# Save final velocity model, function value and history
h5open("lsrtm_marmousi_easgd_result.h5", "w") do file
    write(file, "x", reshape(xav, model0.n))
end

# Plot final image
figure(); imshow(copy(adjoint(reshape(xav, model0.n))), extent = (0, 7.99, 3.19, 0), cmap = "gray", vmin = -3e-2, vmax = 3e-2)
title("LS-RTM with elastic average SGD"); xlabel("Lateral position [km]"); ylabel("Depth [km]")
