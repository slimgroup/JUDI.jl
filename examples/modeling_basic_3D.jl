# Example for basic 3D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling

## Set up model structure
n = (120,100,90)    # (x,y,z) or (x,z)
d = (10.,10.,10.)
o = (0.,0.,0.)

# Velocity [km/s]
v = ones(Float32,n) + 0.4f0
v0 = ones(Float32,n) + 0.4f0
v[:,:,Int(round(end/2)):end] = 4.0f0
rho = ones(Float32,n)

# Slowness squared [s^2/km^2]
m = (1./v).^2
m0 = (1./v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 4
model = Model(n,d,o,m)  # to include density call Model(n,d,o,m,rho)
model0 = Model(n,d,o,m0)

## Set up 3D receiver geometry by defining one receiver vector in each x and y direction
nxrec = 120
nyrec = 100
xrec = linspace(50.,1150.,nxrec)
yrec = linspace(100., 900., nyrec)
zrec = 50

# Construct 3D grid from basis vectors
(xrec, yrec, zrec) = setup_3D_grid(xrec, yrec, zrec)

# receiver sampling and recording time
timeR = 1000.   # receiver recording time [ms]
dtR = 4.    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([250. 500. 750. 1000.])
ysrc = convertToCell([200. 400. 600. 800.])
zsrc = convertToCell([50. 60. 70. 80.])

# source sampling and number of time steps
timeS = 1000.   # source length in [ms]
dtS = 2.    # source sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.01
wavelet = ricker_wavelet(timeS,dtS,f0)

# Info structure for linear operators
ntComp = get_computational_nt(srcGeometry,recGeometry,model)    # no. of computational time steps
info = Info(prod(n),nsrc,ntComp)

###################################################################################################

# Enable optimal checkpointing
opt = Options(optimal_checkpointing=true)

# Setup operators
Pr = judiProjection(info,recGeometry)
F = judiModeling(info,model)
Ps = judiProjection(info,srcGeometry)
q = judiVector(srcGeometry,wavelet)

# Nonlinear modeling
dobs = Pr*F*Ps'*q
qad = Ps*F'*Pr'*dobs

# Linearied modeling
J = judiJacobian(Pr*F*Ps',q)
dD = J*dm
rtm = J'*dD

# evaluate FWI objective function 
f,g = fwi_objective(model0, q, dobs; options=opt)




