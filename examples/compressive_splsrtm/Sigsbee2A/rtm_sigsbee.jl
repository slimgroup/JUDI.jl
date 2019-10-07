# Reverse-time migration of the Sigsbee2A model: time-domain vs. frequency domain w/ on-the-fly Fourier transforms
# Author: Philipp Witte, pwitte.slim@gmail.com
# Date: May 2018
#

# TO DO:
# Replace w/ full path the observed data directory
path_to_data = "/path/to/directory/"
data_name = "sigsbee2A_marine"  # common base name of all shots

using JUDI.TimeModeling, PyPlot, JLD, SegyIO

# Load Sigsbee model
M = load("sigsbee2A_model.jld")

# Setup info and model structure
model0 = Model(M["n"], M["d"], M["o"], M["m0"])
dm = vec(M["dm"])

# Set up out-of-core data container
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

opt = Options(isic=true, optimal_checkpointing=true)    # use impedance imaging

# Setup operators
Pr = judiProjection(info, d_lin.geometry)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, q.geometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Time-domain RTM w/ optimal checkpointing
rtm_time = adjoint(J)*d_lin

# Save time-domain result
save("sigsbee2A_rtm_time_domain", "rtm", reshape(rtm_time, model0.n))

# Generate probability density function from source spectrum
q_dist = generate_distribution(q)

# Select 20 random frequencies per source location
nfreq = 20
J.options.optimal_checkpointing = false
J.options.frequencies = Array{Any}(undef, d_lin.nsrc)
for j=1:d_lin.nsrc
    J.options.frequencies[j] = select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=nfreq)
end
J.options.dft_subsampling_factor = 8    # perform dft every 8 time steps only

# Frequency-domain RTM w/ on-the-fly DFTs
rtm_freq = adjoint(J)*d_lin

# Save frequency-domain result
save("sigsbee2A_rtm_frequency_domain", "rtm", reshape(rtm_freq, model0.n))
