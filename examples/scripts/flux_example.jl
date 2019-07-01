# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

# Here is the hack for example below
module JOLI4Flux
    using JUDI.TimeModeling: judiJacobian
    using Tracker
    import Base.*
    import Tracker.@grad

    *(x::judiJacobian,y::TrackedVector) = Tracker.track(*, x, y)

    @grad a::judiJacobian * b::AbstractVecOrMat =
        Tracker.data(a)*Tracker.data(b), Δ -> (Δ * transpose(b), transpose(a) * Δ)
end
# end of hack

using Pkg; Pkg.activate("JUDI")
using JUDI.TimeModeling, SeisIO, LinearAlgebra, Flux, JOLI


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
nsrc = 1	# number of sources
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
xsrc = convertToCell([600f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([20f0])

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

# return linearized data as Julia array
opt = Options(return_array=true)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Linearized modeling
y = J*dm

##################################### FLUX CONVOLUTION ############################################

# Test demigration operator w/ Flux convolution
m = Conv((3, 3), 1=>1, pad=1, stride=1);
x = reshape(dm, model0.n[1], model0.n[2], 1, 1)

predict(x) = J * vec(m(x))
loss(x,y) = Flux.mse(predict(x), y)
f = loss(x, y)


##################################### FLUX DENSE LAYER #############################################

# Test demigration operator w/ Flux Dense Layer
W = randn(Float32, 100, length(y))
b = randn(Float32, 100)
x = dm
y = randn(Float32, 100)

predict(x) = W*(J*x) .+ b
loss(x,y) = Flux.mse(predict(x), y)

W = param(W)
b = param(b)

gs = Tracker.gradient(() -> loss(x, y), params(W, b))

gs[W]
