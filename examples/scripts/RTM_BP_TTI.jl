using JUDI.TimeModeling, SeisIO, JLD, PyPlot
# Load velocity model

vp = segy_read("/nethome/mlouboutin3/Research/datasets/BP2007/Vp_Model.sgy")  # IBM Float32 format [m/s]
epsilon = segy_read("/nethome/mlouboutin3/Research/datasets/BP2007/Epsilon_Model.sgy")
delta = segy_read("/nethome/mlouboutin3/Research/datasets/BP2007/Delta_Model.sgy")
theta = segy_read("/nethome/mlouboutin3/Research/datasets/BP2007/Theta_Model.sgy")

vp = convert(Array{Float32, 2}, vp.data)' / 1e3     # convert to IEEE Float32 and [km/s]
epsilon = convert(Array{Float32, 2}, epsilon.data)'
delta = convert(Array{Float32, 2}, delta.data)'
theta = - pi / 180.0f0 * convert(Array{Float32, 2}, theta.data)'

d = (6.25, 6.25)
o = (0., 0.)
m0 = (1f0 ./ vp).^2
n = size(m0)

# Set up model structure

model0 = Model_TTI(n, d, o, m0; epsilon=epsilon, delta=delta, theta=theta)

# Scan directory for segy files and create out-of-core data container

container = segy_scan("/nethome/mlouboutin3/Research/datasets/BP2007/shots/", "Ani", ["GroupX", "GroupY", "RecSourceScalar", "dt"])

d_obs = judiVector(container; segy_depth_key="ElevationScalar")

# Set up source
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.027)  # 8 Hz peak frequency
# Info structure for linear operators
ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), d_obs.nsrc, ntComp)

###################################################################################################
# Enable optimal checkpointing
opt = Options(optimal_checkpointing = true,
              limit_m = true,
              buffer_size = 2000f0,
              isic = true)

# Setup operators
Pr = judiProjection(info, d_obs.geometry)
F = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, src_geometry)
q = judiVector(src_geometry, wavelet)
J = judiJacobian(Pr*F*Ps', q)

rtm = J'*d_obs
save("rtm_bp_tti_synthetic_full.jld", "x", rtm)

# The above way requires to gather all 1600 gradients, we can split it

rtm = J[1:82]'*d_obs[1:82]
save(string("rtm_bp_tti_synthetic_flip_coarse",1,".jld"), "x", rtm)
for i=2:19
	println("Running shots ", (i-1)*82+1, " to ", i*82)
	rtm += J[(i-1)*82+1:i*82]'*d_obs[(i-1)*82+1:i*82]
	save(string("rtm_bp_tti_synthetic_flip_coarse",i,".jld"), "x", rtm)
end
