using JUDI.TimeModeling, PyPlot

## Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) + 0.4f0
v0 = ones(Float32,n) + 0.4f0
v[:,Int(round(end/2)):end] = 3f0

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
xrec = linspace(50f0, 1150f0, nxrec)
yrec = 0f0
zrec = linspace(50f0, 50f0, nxrec)

# receiver sampling and recording time
timeR = 1000f0   # receiver recording time [ms]
dtR = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(600f0)
ysrc = convertToCell(0f0)
zsrc = convertToCell(20f0)

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

############################################################################

# Write shots as segy files to disk
opt = Options(save_data_to_disk=false, file_path=pwd(), file_name="observed_shot", optimal_checkpointing=true)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*Ps', q)

# Pr*F*Ps'*q
d_obs = Pr*F*Ps'*q
q_ad = Ps*F*Pr'*d_obs

# time_modeling(model, src_geometry, src_data, rec_geometry, rec_data, dm, nsrc, op, mode)

# Data in, wavefields out
u = time_modeling(model, q.geometry, q.data, nothing, nothing, nothing, 1, 'F', 1, Options())
v = time_modeling(model, nothing, nothing, d_obs.geometry, d_obs.data, nothing, 1, 'F', -1, Options())

# Wavefield in, data out
d_q = time_modeling(model, nothing, v.data, recGeometry, nothing, nothing, 1, 'F', 1, Options())
#qad_q = time_modeling(model, q.geometry, nothing, nothing, u, nothing, 1, 'F', 1, Options())


# Ps*F'*Pr'*d
#qad  = time_modeling(model, q.geometry, [], recGeometry, dobs.data, [], 1, 'F', -1)

# F*Ps'*q
#wf = time_modeling(mode, q.geometry, q.data, [], [], [], 1, 'F', 1)




