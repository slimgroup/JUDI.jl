# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, PyPlot, JLD, SeisIO

# Load Sigsbee model
M = load("/scratch/slim/shared/mathias-philipp/sigsbee2A/Sigsbee_LSRTM.jld")
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

opt = Options(isic=true,
              save_data_to_disk=false,
              optimal_checkpointing=false,
              dft_subsampling_factor=1
              )

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*Ps', q)

# Set frequencies
q_dist = generate_distribution(q)
J.options.frequencies = Array{Any}(nsrc)
J.options.frequencies[1] = [0.015776, 0.024124, 0.0199241, 0.023742]    #select_frequencies(q_dist; fmin=0.003, fmax=0.04, nf=4)

# Impedance
tic()
d_lin1 = J*dm
toc()
tic()
rtm1 = J'*d_lin1
toc()
figure(); imshow(reshape(rtm1, model0.n)')

J.options.dft_subsampling_factor = 8
tic()
d_lin2 = J*dm
toc()
tic()
rtm2 = J'*d_lin2
toc()
figure(); imshow(reshape(rtm2, model0.n)')
