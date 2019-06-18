using JUDI, JUDI.TimeModeling, SeisIO, JLD, PyPlot

# Path to model (insert correct path)
model_path = "/path/to/model/"
data_path = "/path/to/data/"
geometry_path = "/path/to/geometry/"

# Load velocity and density
vp = load(join([model_path, "bp_synthetic_2004_true_velocity.jld"]))["vp"] / 1f3
rho = load(join([model_path, "bp_synthetic_2004_density.jld"]))["rho"]

# Load geometry of original BP data set
geometry = load(join([geometry_path, "bp_synthetic_2004_header_geometry.jld"]))

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
