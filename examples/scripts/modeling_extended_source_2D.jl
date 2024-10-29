using JUDI, SegyIO, LinearAlgebra, PythonPlot

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.4f0
v[:,Int(round(end/2)):end] .= 3f0
v0 = ones(Float32,n) .+ 0.4f0

# Slowness squared [s^2/km^2]
m0 = (1f0 ./ v0).^2
m = (1f0 ./ v).^2
dm = vec(m - m0)

# Setup model structure
nsrc = 2	# number of sources
model0 = Model(n, d, o, m0)
model = Model(n, d, o, m)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
time = 1000f0   # receiver recording time [ms]
dt = 4f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dt, t=time, nsrc=nsrc)

# Source wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(time, dt, f0)

###################################################################################################

# Write shots as segy files to disk
opt = Options(return_array=false, dt_comp=1.0, free_surface=true)

# Setup operators
Pr = judiProjection(recGeometry)
F = judiModeling(model; options=opt)

# Extended source weights
weights = Array{Array}(undef, nsrc)
for j=1:nsrc
    weights[j] = randn(Float32, model.n)
end
w = judiWeights(weights)

# Create operator for injecting the weights, multiplied by the provided wavelet(s)
Pw = judiLRWF(nsrc, dt, wavelet)

# Model observed data w/ extended source
F = Pr*F*adjoint(Pw)

# Simultaneous observed data
d_sim = F*w
dw = adjoint(F)*d_sim

# Jacobian
J = judiJacobian(F, w)
d_lin = J*dm
g = adjoint(J)*d_lin

# Plot results
figure()
subplot(1,2,1)
imshow(d_sim.data[1], vmin=-5e2, vmax=5e2, cmap="gray"); title("Non-linear shot record")
subplot(1,2,2)
imshow(d_lin.data[1], vmin=-5e3, vmax=5e3, cmap="gray"); title("Linearized shot record")

figure()
subplot(1,2,1)
imshow(adjoint(dw.weights[1]), vmin=-5e6, vmax=5e6, cmap="gray"); title("Weights 1")
subplot(1,2,2)
imshow(adjoint(reshape(g, model0.n)), vmin=-1e8, vmax=1e8, cmap="gray"); title("Gradient w.r.t. m")