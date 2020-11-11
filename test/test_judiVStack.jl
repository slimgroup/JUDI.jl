# Unit test for judiVStack
# Ziyi Yin (ziyi.yin@gatech.edu)
# Nov 2020

using JUDI.TimeModeling, LinearAlgebra, PyPlot, IterativeSolvers, JOLI, Test, Printf

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 5f0

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2

# Setup info and model structure
nsrc = 1    # number of sources
model = Model(n, d, o, m)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 4f0    # receiver sampling interval [ms]
nt = Int(time/dt)+1

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# Source wavelet
f0 = 0.01f0     # MHz
wavelet = ricker_wavelet(time, dt, f0)

# Set up info structure for linear operators
ntComp = get_computational_nt(recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

###################################################################################################

# Write shots as segy files to disk
opt = Options()

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)

# Create operator for injecting the weights, multiplied by the provided wavelet(s)
Pw = judiLRWF(info, wavelet)

lambda = rand(Float32)
# Model observed data w/ extended source
I = joDirac(info.n, DDT=Float32, RDT=Float32)
F = Pr*F*adjoint(Pw)
F̄ = [F; lambda*I]

# Random weights (size of the model)
x = judiWeights(randn(Float32, model.n))

d_obs = F*x

y1 = deepcopy(d_obs)
y1.data[1] = randn(Float32,nt,nxrec)
y2 = judiWeights(randn(Float32, model.n))

y = [y1;y2]

a = dot(y,F̄*x)
b = dot(F̄'*y,x)

tol = 5f-3

@printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, a/b - 1)
@test isapprox(a, b, rtol=tol)
