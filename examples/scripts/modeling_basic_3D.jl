# Example for basic 3D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using Pkg; Pkg.activate("JUDI")
using JUDI.TimeModeling

## Set up model structure
n = (120, 100, 90)    # (x,y,z) or (x,z)
d = (10., 10., 10.)
o = (0., 0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .* 1.4f0
v0 = ones(Float32,n) .* 1.4f0
v[:, :, Int(round(end/2)):end] .= 4.0f0
rho = ones(Float32, n)

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 4
model = Model(n, d, o, m)  # to include density call Model(n,d,o,m,rho)
model0 = Model(n, d, o, m0)

## Set up 3D receiver geometry by defining one receiver vector in each x and y direction
nxrec = 120
nyrec = 100
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = range(100f0, stop=900f0, length=nyrec)
zrec = 50f0

# Construct 3D grid from basis vectors
(xrec, yrec, zrec) = setup_3D_grid(xrec, yrec, zrec)

# receiver sampling and recording time
timeR = 100f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([250f0, 500f0, 750f0, 1000f0])
ysrc = convertToCell([200f0, 400f0, 600f0, 800f0])
zsrc = convertToCell([50f0, 60f0, 70f0, 80f0])

# source sampling and number of time steps
timeS = 100f0   # source length in [ms]
dtS = 2f0    # source sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.01f0
wavelet = ricker_wavelet(timeS, dtS, f0)

# Info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)    # no. of computational time steps
info = Info(prod(n), nsrc, ntComp)

###################################################################################################

# Enable optimal checkpointing
opt = Options(optimal_checkpointing=true)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model)
Ps = judiProjection(info, srcGeometry)
q = judiVector(srcGeometry, wavelet)

# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q
qad = Ps*F*adjoint(Pr)*dobs

# Linearied modeling
F0 = judiModeling(info, model0) # modeling operator for background model
J = judiJacobian(Pr*F0*adjoint(Ps), q)
dD = J*dm
rtm = adjoint(J)*dD

# evaluate FWI objective function
f, g = fwi_objective(model0, q, dobs; options=opt)
