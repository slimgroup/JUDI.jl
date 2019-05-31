using JUDI.TimeModeling, SeisIO, JLD, PyPlot

# Load velocity model
path = "/scratch/slim/shared/mathias-philipp/bp_synthetic_2004"
vp = load(join([path, "/model/vp_fine.jld"]))["vp"] / 1f3
rho = load(join([path, "/model/rho_fine.jld"]))["rho"]

# Set up model structure
d = (6.25, 6.25)
o = (0., 0.)
m0 = (1f0 ./ vp).^2
n = size(m0)
model0 = Model(n, d, o, m0; rho=rho)

# Scan directory for segy files and create out-of-core data container
container = segy_scan(join([path, "/data/"]), "shots", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = "SourceDepth")

# Set up source
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.020)  # 27 Hz peak frequency
q = judiVector(src_geometry, wavelet)

# Info structure for linear operators
ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), d_obs.nsrc, ntComp)

###################################################################################################

# Enable optimal checkpointing
opt = Options(limit_m = true,
              space_order = 16,
              buffer_size = 4000f0,
              save_data_to_disk = true, 
              file_path = join([path, "/data_no_multiples"]),
              file_name = "bp_observed_data_")

# Setup operators
F = judiModeling(info, model0, q.geometry, d_obs.geometry; options=opt)

# Model data
d_container = F*q
save(join([path, "/data_no_multiples/observed_data_container.jld"]), "d_container", d_container)



