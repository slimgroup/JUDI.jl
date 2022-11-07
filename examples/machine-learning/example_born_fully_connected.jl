# Example for linearized modeling with Flux networks
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: June 2022
# Adapted from https://github.com/slimgroup/JUDI4Flux.jl/tree/master/examples

using JUDI, SegyIO, JOLI, Flux

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v0 = ones(Float32,n) .+ 0.4f0
v[:, Int(round(end/2)):end] .= 4f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup model structure
nsrc = 1	# number of sources
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 1f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([600f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([20f0])

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=time)

# setup wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)
q = judiVector(srcGeometry, wavelet)
# return linearized data as Julia array
opt = Options(return_array=true)

# Non-linear forward modeling operator
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
num_samples = recGeometry.nt[1] * nxrec

##################################################################################
# Fully connected neural network with linearized modeling operator
n_in = 100
n_out = 10

W1 = randn(Float32, prod(model0.n), n_in)
b1 = randn(Float32, prod(model0.n))

W2 = judiJacobian(F0, q)
b2 = randn(Float32, num_samples)

W3 = randn(Float32, n_out, num_samples)
b3 = randn(Float32, n_out)

function network(x)
    x = W1*x .+ b1
    x = vec(W2*x) .+ b2
    x = W3*x .+ b3
    return x
end

# Inputs and target
x = zeros(Float32, n_in)
y = randn(Float32, n_out)

# Evaluate MSE loss
loss(x, y) = Flux.mse(network(x), y)

# Compute gradient w.r.t. x and y
Δx, Δy = gradient(loss, x, y)

# Compute gradient for x, y and weights (except for W2)
p = Flux.params(x, y, W1, b1, b2, W3, b3)
gs = gradient(() -> loss(x, y), p)

# Access gradients
Δx = gs[x]
ΔW1 = gs[W1]
Δb1 = gs[b1]

# and so on...

