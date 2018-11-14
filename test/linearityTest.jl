# Test linearity of sources
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, JUDI.TimeModeling, Test, LinearAlgebra

## Set up model structure
n = (120,100)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 4f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2

# Setup info and model structure
nsrc = 1
ntComp = 250
info = Info(prod(n), nsrc, ntComp)	# number of gridpoints, number of experiments, number of computational time steps
model = Model(n,d,o,m)

## Set up receiver geometry
nxrec = 120
xrec = range(50f0,stop=1150f0,length=nxrec)
yrec = 0f0
zrec = range(50f0,stop=50f0,length=nxrec)

# receiver sampling and recording time
timeR = 1000f0	# receiver recording time [ms]
dtR = 4f0	# receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometries (cell array with source locations for each shot)
xsrc1 = 300f0
ysrc1 = 0f0
zsrc1 = 20f0

xsrc2 = 600f0
ysrc2 = 0f0
zsrc2 = 20f0

# source sampling and number of time steps
timeS = 1000f0
dtS = 4f0	# receiver sampling interval

# Set up source structure
srcGeometry1 = Geometry(xsrc1,ysrc1,zsrc1;dt=dtS,t=timeS)
srcGeometry2 = Geometry(xsrc2,ysrc2,zsrc2;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.01f0
wavelet = ricker_wavelet(timeS,dtS,f0)

###################################################################################################

# Modeling operators
Pr = judiProjection(info,recGeometry)
Ps1 = judiProjection(info,srcGeometry1)
Ps2 = judiProjection(info,srcGeometry2)
F = judiModeling(info,model)
q1 = judiVector(srcGeometry1,wavelet)
q2 = judiVector(srcGeometry2,wavelet)

d1 = Pr*F*adjoint(Ps1)*q1
d2 = Pr*F*adjoint(Ps2)*q2
d3 = Pr*F*(adjoint(Ps1)*q1 + adjoint(Ps2)*q2)
d4 = Pr*F*(adjoint(Ps1)*q1 - adjoint(Ps2)*q2)

@test isapprox(norm(d3), norm(d1 + d2))
@test isapprox(norm(d4), norm(d1 - d2))
