# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, PyPlot, JLD, SegyIO, LinearAlgebra

# Load Marmousi (4 km x 2 km)
if !isfile("marmousi_small.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/marmousi_small.jld`)
end
M = load("marmousi_small.jld")

# Setup info and model structure
model = Model(M["n"], M["d"], M["o"], M["m"])
model0 = Model(M["n"], M["d"], M["o"], M["m0"])
dm = vec(M["dm"])

# Setup receivers
nsrc = 10
nxrec = 761 # 10 m receiver spacing
xrec = range(100f0, stop=3900f0, length=nxrec)
yrec = 0f0
zrec = range(210f0, stop=210f0, length=nxrec)

# receiver sampling and recording time
timeR = 2500f0
dtR = 4f0

# Set up receiver structure
rec_geometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(200f0, stop=3800f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

# source sampling and number of time steps
timeS = 2500f0
dtS = 1f0

# Set up source structure
src_geometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# Set up source
wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.030)  # 25 Hz peak frequency
q = judiVector(src_geometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(q.geometry, rec_geometry, model0)
info = Info(prod(model0.n), nsrc, ntComp)


#################################################################################################

opt = Options(isic=true, optimal_checkpointing=true)

# Setup operators
Pr = judiProjection(info, rec_geometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, src_geometry)
J = judiJacobian(Pr*F0*Ps', q)

# Time-domain RTM
d_lin = J*dm
rtm_time_5 = adjoint(J[5])*d_lin[5]
rtm_time = adjoint(J)*d_lin

# Frequency-domain RTM
J.options.optimal_checkpointing = false
J.options.dft_subsampling_factor = 8
q_dist = generate_distribution(q)
J.options.frequencies = Array{Any}(undef, nsrc)
for j=1:nsrc
    J.options.frequencies[j] = select_frequencies(q_dist; fmin=0.002, fmax=0.05, nf=20)
end
rtm_freq = adjoint(J)*d_lin
rtm_freq_5 = adjoint(J[5])*d_lin[5]


######################################### Plots ####################################################


n = model0.n
rtm_time_5 = reshape(rtm_time_5, n)[:, 41:end]
rtm_time = reshape(rtm_time, n)[:, 41:end]
rtm_freq_5 = reshape(rtm_freq_5, n)[:, 41:end]
rtm_freq = reshape(rtm_freq, n)[:, 41:end]

rtm_time_5 /= norm(rtm_time_5, Inf)
rtm_time /= norm(rtm_time, Inf)
rtm_freq_5 /= norm(rtm_freq_5, Inf)
rtm_freq /= norm(rtm_freq, Inf)


figure(figsize=(6.66, 6))

s1=subplot(3, 2, 1)
im1 = imshow(adjoint(model0.m[:, 41:end]),cmap="gray", extent=(0.0, 4.0, 2.0, 0.2))
s1[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t1 = title("a)", fontweight="bold", size=10, loc="left")
t1["set_position"]([-.18, 1])

s2=subplot(3, 2, 2)
im1 = imshow(adjoint(reshape(dm, n)[:, 41:end]), vmin=-1e-1, vmax=1e-1,cmap="gray", extent=(0.0, 4.0, 2.0, 0.2))
s2[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t2 = title("b)", fontweight="bold", size=10, loc="left")
t2["set_position"]([-.18, 1])

s3=subplot(3, 2, 3)
im1 = imshow(adjoint(rtm_time_5), vmin=-2e-2, vmax=2e-2,cmap="gray", extent=(0.0, 4.0, 2.0, 0.2))
s3[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t3 = title("c)", fontweight="bold", size=10, loc="left")
t3["set_position"]([-.18, 1])

s4=subplot(3, 2, 4)
im1 = imshow(adjoint(rtm_time), vmin=-5e-2, vmax=5e-2,cmap="gray", extent=(0.0, 4.0, 2.0, 0.2))
s4[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t4 = title("d)", fontweight="bold", size=10, loc="left")
t4["set_position"]([-.18, 1])

s5=subplot(3, 2, 5)
im1 = imshow(adjoint(rtm_freq_5), vmin=-8e-2, vmax=8e-2,cmap="gray", extent=(0.0, 4.0, 2.0, 0.2))
s5[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t5 = title("e)", fontweight="bold", size=10, loc="left")
t5["set_position"]([-.18, 1])

s6=subplot(3, 2, 6)
im1 = imshow(adjoint(rtm_freq), vmin=-1.5e-1, vmax=1.5e-1,cmap="gray", extent=(0.0, 4.0, 2.0, 0.2))
s6[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t6 = title("f)", fontweight="bold", size=10, loc="left")
t6["set_position"]([-.18, 1])
tight_layout()

savefig("figure2", dpi=300, format="png")
run(`mogrify -trim figure2`)
