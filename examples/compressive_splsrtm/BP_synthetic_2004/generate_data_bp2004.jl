# Create BP Synthetic data w/ 20 Hz peak frequency and using
# the same geometry as the original data set, but w/o surface
# related multiples.
# Author: Philipp Witte, pwitte.slim@gmail.com
# Date: May 2018
#

# TO DO
# Set up path where data will be saved
data_path = "/path/to/data/"

using JUDI, JUDI.TimeModeling, SeisIO, JLD, PyPlot

# Load velocity
if !isfile("bp_synthetic_2004_true_velocity.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/bp_synthetic_2004_true_velocity.jld`)
end
vp = load(join([pwd(), "/bp_synthetic_2004_true_velocity.jld"]))["vp"] / 1f3

# Load density
if !isfile("bp_synthetic_2004_density.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/bp_synthetic_2004_density.jld`)
end
rho = load(join([pwd(), "/bp_synthetic_2004_density.jld"]))["rho"]

# Load geometry of original BP data set
if !isfile("bp_synthetic_2004_header_geometry.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/bp_synthetic_2004_header_geometry.jld`)
end
geometry = load(join([pwd(), "/bp_synthetic_2004_header_geometry.jld"]))

# Set up model structure
d = (6.25, 6.25)
o = (0., 0.)
m0 = (1f0 ./ vp).^2
n = size(m0)
model0 = Model(n, d, o, m0; rho=rho)

# Load header geometry from original data set
src_geometry = geometry["src_geometry"]
rec_geometry = geometry["rec_geometry"]
nsrc = geometry["nsrc"]

# Set up source
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.020)  # 27 Hz peak frequency
q = judiVector(src_geometry, wavelet)

# Info structure for linear operators
ntComp = get_computational_nt(src_geometry, rec_geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), nsrc, ntComp)

###################################################################################################

# Save data to disk
opt = Options(limit_m = true,
              space_order = 16,
              buffer_size = 4000f0,
              save_data_to_disk = true,
              file_path = data_path,
              file_name = "bp_observed_data_")

# Setup operators
F = judiModeling(info, model0, q.geometry, rec_geometry; options=opt)

# Model data (write to disk)
F*q
