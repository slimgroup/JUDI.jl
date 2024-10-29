# Example for modeling simultaneous sources and reverse-time migration
# Author: pwitte.slim@gmail.com
# Date: December 2018
#

using JUDI, PythonPlot

## Set up model structure
n = (120, 100)	# (x, y, z) or (x, z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32, n) .+ 0.4f0
v0 = ones(Float32, n) .+ 0.4f0
v[:, Int(round(end/2)):end] .= 4.0f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup model structure
nsrc = 1    # one simultaneous source experiment
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

## Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(100f0, stop=100f0, length=nxrec)
timeR = 2000f0
dtR = 4f0

# Setup receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt = dtR, t = timeR)

## Set up source geometry
xsrc = [250f0, 500f0, 750f0, 1000f0]	# four simultaneous sources
ysrc = 0f0
zsrc = [50f0, 50f0, 50f0, 50f0]

# Source sampling and number of time steps
timeS = 2000f0
dtS = 4f0

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt = dtS, t = timeS)

# setup wavelets
f0 = 0.01	# source peak frequencies
q = ricker_wavelet(500f0, dtS, f0)  # 500 ms wavelet

# Create four source functions with different time shifts of q
wavelet1 = zeros(srcGeometry.nt[1])
wavelet1[1:1+length(q)-1] = q
wavelet2 = zeros(srcGeometry.nt[1])
wavelet2[41:41+length(q)-1] = q
wavelet3 = zeros(srcGeometry.nt[1])
wavelet3[121:121+length(q)-1] = q
wavelet4 = zeros(srcGeometry.nt[1])
wavelet4[201:201+length(q)-1] = q
wavelet = [wavelet1 wavelet2 wavelet3 wavelet4]	# Collect all wavelets
###################################################################################################

# Setup operators
Pr = judiProjection(recGeometry)
A_inv = judiModeling(model)
A0_inv = judiModeling(model0)
Ps = judiProjection(srcGeometry)
q = judiVector(srcGeometry, wavelet)

# Linearized modeling
d = Pr*A_inv*adjoint(Ps)*q  # simultaneous shot
J = judiJacobian(Pr*A0_inv*adjoint(Ps), q)
dD = J*dm   # simultaneous reflection data
rtm = adjoint(J)*dD # RTM

# Plot results
figure(); imshow(d.data[1], vmin = -1e1, vmax = 1e1, cmap = "seismic"); title("Simultaneous shot w/ four sources")
figure(); imshow(copy(adjoint(reshape(rtm,model.n))), ColorMap("gray")); title("Migrated simultaneous shot")
