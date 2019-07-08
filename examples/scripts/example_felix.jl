using Pkg; Pkg.activate("JUDI")
using Statistics, Random, LinearAlgebra
using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot

# Load background velocity model
n,d,o,m0 = read(h5open("../../data/overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)
n,d,o,m = read(h5open("../../data/overthrust_model.h5","r"), "n", "d", "o", "m")

# Set source & receiver geometry
nsrc = 369;
xsrc  = convertToCell(range(400f0, stop=9600f0, length=nsrc));
ysrc  = convertToCell(range(0f0, stop=0f0, length=nsrc));
zsrc  = convertToCell(range(50f0, stop=50f0, length=nsrc));
time  = 2000f0   # receiver recording time [ms]
dt    = 4.0f0    # receiver sampling interval [ms]

# Set up source structure
src_Geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time);
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
    sim_source[:, j] = wavelet * randn(1)[1]/sqrt(num_sources)    # wavelet w/ random weight
end

# simultaneous source geometry and JUDI vector
sim_geometry = Geometry(xsrc, ysrc, zsrc; dt=src_Geometry.dt[1], t=src_Geometry.t[1])
q = judiVector(sim_geometry, sim_source)

# Receiver
xrec  = range(400f0, stop=9600f0, length=nsrc);
yrec  = range(0f0, stop=0f0, length=nsrc);
zrec  = range(50f0, stop=50f0, length=nsrc);

# Set up source structure
rec_Geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time, nsrc=1);

nsrc = 1    # one simultaneous source
ntComp = get_computational_nt(sim_geometry, rec_Geometry, model0)

info = Info(prod(n), nsrc, ntComp) # only need geometry of one shot record

# Setup operators
opt = Options(return_array=true)
Pr = judiProjection(info, rec_Geometry)    # set up 1 simultaneous shot instead of 16
F = judiModeling(info, model0; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, sim_geometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Linearized modeling + migration
δm = m .- m0;
δm = (δm .-mean(δm)) ./ std(δm);

# Linearized modeling with on-the-fly Fourier + migration
J.options.dft_subsampling_factor = 8
q_dist = generate_distribution(q)
J.options.frequencies = Array{Any}(undef, nsrc)
for j=1:nsrc
    J.options.frequencies[j] = select_frequencies(q_dist; fmin=0.002, fmax=0.04, nf=4)
end

# Linearized modeling w/ single sim source
δd = J*vec(δm);

# RTM w/ single sim source
δm̂ = transpose(J)*vec(δd);

figure(); imshow(reshape(δd, rec_Geometry.nt[1], length(rec_Geometry.xloc[1])), vmin=-5e2, vmax=5e2, cmap="gray")
title("Simultaneous shot")
figure();imshow(transpose(reshape(δm̂,model0.n)), vmin=-1e4, vmax=1e4)
title("Sim source RTM frequency-domain")
