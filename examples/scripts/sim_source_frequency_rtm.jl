using Pkg; Pkg.activate("JUDI")
using Statistics, Random, LinearAlgebra
using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot

# Load background velocity model
n,d,o,m0 = read(h5open("overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)
n,d,o,m = read(h5open("overthrust_model.h5","r"), "n", "d", "o", "m")

# Load data
block = segy_read("overthrust_shot_records.segy")
d_obs = judiVector(block)

# Read original geometry (16 single sources)
src_Geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_Geometry.t[1],src_Geometry.dt[1],0.008f0)  # 8 Hz wavelet
num_sources = length(src_Geometry.xloc)

# Convert to single simultaneous source
xsrc = zeros(Float32, num_sources)
ysrc = zeros(Float32, num_sources)
zsrc = zeros(Float32, num_sources)
sim_source = zeros(Float32, src_Geometry.nt[1], num_sources)
for j=1:num_sources
    xsrc[j] = src_Geometry.xloc[j]
    zsrc[j] = src_Geometry.zloc[j]
    sim_source[:, j] = wavelet * randn(1)[1]    # wavelet w/ random weight
end

# simultaneous source geometry and JUDI vector
sim_geometry = Geometry(xsrc, ysrc, zsrc; dt=src_Geometry.dt[1], t=src_Geometry.t[1])
q = judiVector(sim_geometry, sim_source)

nsrc = 1    # one simultaneous source
ntComp = get_computational_nt(sim_geometry, d_obs[1].geometry, model0)

info = Info(prod(n), d_obs[1].nsrc, ntComp) # only need geometry of one shot record
opt = Options()

# Setup operators
Pr = judiProjection(info, d_obs[1].geometry)    # set up 1 simultaneous shot instead of 16
F = judiModeling(info, model0; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, sim_geometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Linearized modeling + migration
δm = m .- m0;
δm = (δm .-mean(δm)) ./ std(δm);

# Time domain image
δd = J*vec(δm);
δm̂_time = transpose(J)*δd;
δm̂_time = reshape(δm̂_time, model0.n)

# Linearized modeling with on-the-fly Fourier + migration

J.options.dft_subsampling_factor = 8
q_dist = generate_distribution(q)
J.options.frequencies = Array{Any}(undef, d_obs.nsrc)
for j=1:d_obs.nsrc
    J.options.frequencies[j] = select_frequencies(q_dist; fmin=0.002, fmax=0.04, nf=4)
end

#δd = J*vec(δm);
#δm̂ = transpose(J)*δd;
δm̂_freq_5 = transpose(J)*δd;

δm̂_freq_5 = reshape(δm̂_freq_5, model0.n)

figure(); imshow(δd.data[1], vmin=-5e2, vmax=5e2, cmap="gray")
title("Simultaneous shot")
figure();imshow(transpose(δm̂_time), vmin=-2e4, vmax=2e4);
title("Sim source RTM time-domain")
figure();imshow(transpose(δm̂_freq_5), vmin=-1e4, vmax=1e4)
title("Sim source RTM frequency-domain")
