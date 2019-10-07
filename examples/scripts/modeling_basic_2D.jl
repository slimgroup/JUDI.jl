# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, SegyIO, LinearAlgebra

## Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v0 = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 5f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 2	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

## Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
timeR = 1000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(400f0, stop=800f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

# source sampling and number of time steps
timeS = 1000f0  # ms
dtS = 4f0   # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

######################## WITH DENSITY ############################################

# Write shots as segy files to disk
opt = Options(save_data_to_disk=false, file_path=pwd(), file_name="observed_shot", optimal_checkpointing=false, isic=false, subsampling_factor=2)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q
qad = Ps*adjoint(F)*adjoint(Pr)*dobs

# Linearized modeling
#J.options.file_name = "linearized_shot"
dD = J*dm
rtm = adjoint(J)*dD

# evaluate FWI objective function
f, g = fwi_objective(model0, q, dobs; options=opt)
