using JUDI, SegyIO, LinearAlgebra, PythonPlot

# Set up model structure
n = (121, 101)   # (x,y,z) or (x,z)
d = (2.5f0, 2.5f0) # in mm
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.5f0
v0 = ones(Float32,n) .+ 0.5f0
v[:,Int(round(end/2)):end] .= 4f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup model structure
nsrc = 1	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

## Set up receiver geometry
nxrec = 120
xrec = range(d[1], stop=d[2]*(n[1]-1), length=nxrec)
yrec = 0f0
zrec = range(d[1], stop=d[1], length=nxrec)

# receiver sampling and recording time
timeR = 250f0   # receiver recording time [ms]
dtR = 0.25f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = d[1]*61
ysrc = 0f0
zsrc = d[1]

# source sampling and number of time steps
timeS = 250f0     # ms
dtS =  0.25f0    # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = .05  # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

###################################################################################################

opt = Options(isic=true)
# Setup operators
Pr = judiProjection(recGeometry)
F = judiModeling(model)
F0 = judiModeling(model0; options=opt)
Ps = judiProjection(srcGeometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q


# With a transducer source pointing down so pi/2 angle and radius 5mm (1cm diameter)
q2 = transducer(q, model.d, 5, pi/2 .* ones(q.nsrc))
Ps2 = judiProjection(q2.geometry)

dobs2 = Pr*F*adjoint(Ps2)*q2

a = 1e-1
figure()
subplot(121)
imshow(dobs.data[1], vmin=-a, vmax=a, cmap="seismic", aspect=.25)
title("Point source")
subplot(122)
imshow(dobs2.data[1], vmin=-a, vmax=a, cmap="seismic", aspect=.25)
title("Transducer source")

dm = J'*dobs2
