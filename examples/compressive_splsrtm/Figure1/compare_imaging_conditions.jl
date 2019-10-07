# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, PyPlot, JLD, SegyIO, LinearAlgebra

# Load Sigsbee model
if !isfile("sigsbee2A_model.jld")
    run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/Imaging.jl/CompressiveLSRTM/sigsbee2A_model.jld`)
end

M = load("sigsbee2A_model.jld")
m = M["m"][1200:2200, 150:800]
m0 = M["m0"][1200:2200, 150:800]
dm = M["dm"][1200:2200, 150:800]
n = size(m)

# Setup info and model structure
model = Model(n, M["d"], M["o"], m)
model0 = Model(n, M["d"], M["o"], m0)
dm = vec(dm)

## Set up receiver geometry
nsrc = 1    # number of sources
nxrec = 1200
xrec = range(700f0, stop=6700f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
timeR = 8000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([6800f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([20f0])

# source sampling and number of time steps
timeS = 8000f0
dtS = 2f0

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.015
wavelet = ricker_wavelet(timeS,dtS,f0)
q = judiVector(srcGeometry,wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry,recGeometry,model)
info = Info(prod(model.n),nsrc,ntComp)

#################################################################################################

opt = Options(isic=false,
              save_data_to_disk=false,
              optimal_checkpointing=false
              )

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*Ps', q)

# Set frequencies
q_dist = generate_distribution(q)
J.options.frequencies = Array{Any}(undef, nsrc)
J.options.frequencies[1] = select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=20)

# Velocity contrast
d_lin1 = J*dm
rtm1 = J'*d_lin1

# Impedance
J.options.isic = true
d_lin2 = J*dm
rtm2 = J'*d_lin2



############################################## Plots ###############################################

# Normalize images
dm1 = rtm1 / norm(rtm1, Inf)
dm2 = rtm2 / norm(rtm2, Inf)

# Plot migration velocity
fig=figure(figsize=(6.66, 5))
s1 = subplot(2, 2, 1)
im1 = imshow(adjoint(sqrt.(1f0 ./ m0)), vmin=1.5, vmax=4.511,cmap="gray", extent=(9.144, 16.764, 6.096, 1.143))
s1[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t1 = title("a)", fontweight="bold", size=10, loc="left")
t1["set_position"]([-.14, 1])

# Plot true image
s2 = subplot(2, 2, 2)
im2 = imshow(adjoint(reshape(dm, n)), vmin=-6e-2, vmax=6e-2,cmap="gray", extent=(9.144, 16.764, 6.096, 1.143))
s2[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t2 = title("b)", fontweight="bold", size=10, loc="left")
t2["set_position"]([-.14, 1])

# RTM image w/ zero-lag cross-correlation imaging condition
s3 = subplot(2, 2, 3)
im3 = imshow(adjoint(reshape(dm1,n)), vmin=-1e-1, vmax=1e-1,cmap="gray", extent=(9.144, 16.764, 6.096, 1.143))
s3[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t3 = title("c)", fontweight="bold", size=10, loc="left")
t3["set_position"]([-.14, 1])

# RTM image w/ inverse-scattering/impedance imaging condition
s4 = subplot(2, 2, 4)
im4 = imshow(adjoint(reshape(dm2, n)), vmin=-4e-2, vmax=4e-2,cmap="gray", extent=(9.144, 16.764, 6.096, 1.143))
s4[:tick_params]("both", labelsize=8)
xlabel("Lateral position [km]", size=8);
ylabel("Depth [km]", size=8)
t4 = title("d)", fontweight="bold", size=10, loc="left")
t4["set_position"]([-.14, 1])

tight_layout()

savefig("figure1", dpi=300, format="png")
run(`mogrify -trim figure1`)
