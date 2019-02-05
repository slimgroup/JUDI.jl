using Pkg; Pkg.activate("JUDI")
using PyCall, PyPlot, JUDI.TimeModeling, Images, SeisIO, JLD
# Load velocity model

vp = segy_read("/data/mlouboutin3/BP2007/Vp_Model.sgy")  # IBM Float32 format [m/s]
epsilon = segy_read("/data/mlouboutin3/BP2007/Epsilon_Model.sgy")
delta = segy_read("/data/mlouboutin3/BP2007/Delta_Model.sgy")
theta = segy_read("/data/mlouboutin3/BP2007/Theta_Model.sgy")

vp = convert(Array{Float32, 2}, vp.data)' / 1e3     # convert to IEEE Float32 and [km/s]
epsilon = convert(Array{Float32, 2}, epsilon.data)'
delta = convert(Array{Float32, 2}, delta.data)'
theta = - pi / 180.0f0 * convert(Array{Float32, 2}, theta.data)'

d = (6.25, 6.25)
o = (0., 0.)
m0 = (1f0 ./ vp).^2
n = size(m0)

# Smooth thomsen parameters


vp = Float32.(imfilter(vp, Kernel.gaussian(10)))
epsilon = Float32.(imfilter(epsilon, Kernel.gaussian(10)))
delta = Float32.(imfilter(delta, Kernel.gaussian(10)))
# theta = Float32.(imfilter(theta, Kernel.gaussian(5)))


wb = find_water_bottom(vp - minimum(vp))
k = 1
water_bottom = zeros(size(vp))
for i in wb
    water_bottom[k, 1:i]  =1.; k+=1;
end
water_bottom = Float32.(water_bottom)


# Set up model structure

model0 = Model_TTI(n, d, o, m0; epsilon=epsilon, delta=delta, theta=theta)

# Scan directory for segy files and create out-of-core data container

container = segy_scan("/data/mlouboutin3/BP2007/shots/", "Ani", ["GroupX", "GroupY", "RecSourceScalar", "dt"])

d_obs = judiVector(container; segy_depth_key="ElevationScalar")

# Set up source
src_geometry = Geometry(container; key = "source")
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.019)  # 8 Hz peak frequency
# Info structure for linear operators
ntComp = get_computational_nt(src_geometry, d_obs.geometry, model0)    # no. of computational time steps
info = Info(prod(model0.n), d_obs.nsrc, ntComp)

###################################################################################################
# Enable optimal checkpointing
opt = Options(optimal_checkpointing = true,
              limit_m = true,
              buffer_size = 2000f0,
              space_order=8,
              isic = true)

# Setup operators
Pr = judiProjection(info, d_obs.geometry)
F = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, src_geometry)
q = judiVector(src_geometry, [0;-diff(wavelet)])
J = judiJacobian(Pr*F*Ps', q)

# rtm = J'*d_obs
# save("rtm_bp_tti_synthetic_full.jld", "x", rtm)

# Right-hand preconditioners
D = judiDepthScaling(model0)
T = judiTopmute(model0.n, (1 - water_bottom), [])
Mr = D*T

d_sub = get_data(d_obs[50:50])#  get_data(d_obs[i])
Ml = judiMarineTopmute2D(40, d_sub.geometry)

# The above way requires to gather all 1600 gradients, we can split it

rtm = Mr*J[150:150]'*Ml*d_sub

# save(string("rtm_bp_tti_synthetic_flip_coarse",1,".jld"), "x", rtm)
# for i=2:19
# 	println("Running shots ", (i-1)*82+1, " to ", i*82)
# 	rtm += J[(i-1)*82+1:i*82]'*d_obs[(i-1)*82+1:i*82]
# 	save(string("rtm_bp_tti_synthetic_flip_coarse",i,".jld"), "x", rtm)
# end
