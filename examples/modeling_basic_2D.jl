# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, SeisIO

## Set up model structure
n = (120,100)   # (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) + 0.4f0
v0 = ones(Float32,n) + 0.4f0
v[:,Int(round(end/2)):end] = 3f0

# Slowness squared [s^2/km^2]
m = (1./v).^2
m0 = (1./v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 1	# number of sources
model = Model(n,d,o,m)
model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 5
xrec = linspace(312.5,312.5,5)
yrec = 312.5 # 0.
zrec = linspace(312.5,312.5,5)

# receiver sampling and recording time
timeR = 400.   # receiver recording time [ms]
dtR = 8.    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=1)

## Set up source geometry (cell array with source locations for each shot)
xsrc = 312.5 # convertToCell(linspace(400.,800.,nsrc))
ysrc = 312.5 # convertToCell(linspace(0.,0.,nsrc))
zsrc = 312.5 # convertToCell(linspace(20.,20.,nsrc))

# source sampling and number of time steps
timeS = 400.
dtS = 8.

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.015
wavelet = ricker_wavelet(timeS,dtS,f0)
q = judiVector(srcGeometry,[diff(wavelet);0])

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry,recGeometry,model00)
info = Info(prod(n),1,ntComp)

######################## WITH DENSITY ############################################

# Write shots as segy files to disk
opt = Options(save_data_to_disk=true, file_path=pwd(), file_name="observed_shot", optimal_checkpointing=true)

# Setup operators
Pr = judiProjection(info,recGeometry)
F = judiModeling(info,model; options=opt)
F0 = judiModeling(info,model0; options=opt)
Ps = judiProjection(info,srcGeometry)
J = judiJacobian(Pr*F0*Ps',q)

# Nonlinear modeling
dobs = Pr*F*Ps'*q
qad = Ps*F'*Pr'*dobs

# Linearized modeling
J.options.file_name="linearized_shot"
dD = J*dm
rtm1 = J'*dD

# evaluate FWI objective function
f,g = fwi_objective(model0, q, dobs; options=opt)
