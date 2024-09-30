# Example for basic 3D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI

# Set up model structure
n = (120, 100)    # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ .5f0
v[:, Int(round(end/2)):end] .= 4.0f0
rho = ones(Float32, n)
vs = zeros(Float32,n)
vs[:, Int(round(end/2)):end] .= 2f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
# Setup model structure
nsrc = 1
model = Model(n, d, o, m; rho=rho, vs=vs)

# Set up 3D receiver geometry by defining one receiver vector in each x and y direction
nxrec = 120
nyrec = 100
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(10f0, stop=10f0, length=nxrec)

# receiver sampling and recording time
timeR = 1500f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = 600f0
ysrc = 0f0
zsrc = 10f0

# source sampling and number of time steps
timeS = 1500f0   # source length in [ms]
dtS = 2f0    # source sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.01f0
wavelet = ricker_wavelet(timeS, dtS, f0)

###################################################################################################

# Setup operators
F = judiModeling(model, srcGeometry, recGeometry; options=Options(space_order=8, free_surface=true))
q = judiVector(srcGeometry, wavelet)

# Nonlinear modeling
dobs = F*q
