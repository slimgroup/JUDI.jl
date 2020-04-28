using JUDI.TimeModeling, SegyIO, LinearAlgebra, PyPlot

## Set up model structure
n = (121, 101)   # (x,y,z) or (x,z)
d = (2.5f-4, 2.5f-4)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.5f0
v0 = ones(Float32,n) .+ 0.5f0
v[:,Int(round(end/2)):end] .= 4f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

## Set up receiver geometry
nxrec = 120
xrec = range(d[1], stop=d[2]*(n[1]-1), length=nxrec)
yrec = 0f0
zrec = range(d[1], stop=d[1], length=nxrec)

# receiver sampling and recording time
timeR = 0.05f0   # receiver recording time [ms]
dtR = 0.0004f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([d[1]*61])
ysrc = convertToCell([0f0])
zsrc = convertToCell([d[1]])

# source sampling and number of time steps
timeS = 0.05f0  # ms
dtS = 0.0004f0   # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 250f0     # MHz
wavelet = ricker_wavelet(timeS, dtS, f0)/1f-4
q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

######################## WITH DENSITY ############################################

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model)
Ps = judiProjection(info, srcGeometry)

# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q

imshow(dobs.data[1], vmin=-4e-5, vmax=4e-5)