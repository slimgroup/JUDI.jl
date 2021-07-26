# Example for 2D sparsity-promoting LS-RTM w/ anti chattering:
# The implementation of anti-chattering follows the journal article
# "Accelerating Sparse Recovery by Reducing Chatter
# Emmanouil Daskalakis, Felix J. Herrmann, Rachel Kuske
# SIAM Journal on Imaging Sciences, 2020"
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: July 2021
#

using JUDI, JOLI, LinearAlgebra, Images, Random, Statistics, Printf

# Set up model structure
n = (120, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.5f0
v[:,20:end] .= 2.5f0
v[:,50:end] .= 3.5f0
v0 = 1f0./convert(Array{Float32,2},imfilter(1f0./v, Kernel.gaussian(2))) # smooth 1/v

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 16	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = 120
xrec = range(50f0, stop=1150f0, length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
timeR = 1000f0   # receiver recording time [ms]
dtR = 2f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(50f0, stop=1150f0, length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(200f0, stop=200f0, length=nsrc))

# source sampling and number of time steps
timeS = 1000f0  # ms
dtS = 2f0   # ms

# Set up source structure
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

# setup wavelet
f0 = 0.01f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry, recGeometry, model)
info = Info(prod(n), nsrc, ntComp)

###################################################################################################

# Write shots as segy files to disk
opt = Options(isic=true)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q

# Preconditioner
idx_wb = find_water_bottom(m-m0)
Tm = judiTopmute(model0.n, idx_wb, 3)  # Mute water column
S = judiDepthScaling(model0)
Mr = Tm*S

# Linearized Bregman parameters

batchsize = 4
niter = 8 # 2 datapass

fval = zeros(Float32, niter)

# Soft thresholding functions and Curvelet transform
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64, lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32, lambda), 0f0)

C0 = joCurvelet2D(2*n[1], 2*n[2]; zero_finest = false, DDT = Float32, RDT = Float64)

function C_fwd(im, C, n)
	im = hcat(reshape(im, n), reshape(im, n)[:, end:-1:1])
    im = vcat(im, im[end:-1:1,:])
	coeffs = C*vec(im)/2f0
	return coeffs
end

function C_adj(coeffs, C, n)
	im = reshape(C'*coeffs, 2*n[1], 2*n[2])
	return vec(im[1:n[1], 1:n[2]] + im[1:n[1], end:-1:n[2]+1] + im[end:-1:n[1]+1, 1:n[2]] + im[end:-1:n[1]+1, end:-1:n[2]+1])/2f0
end

C = joLinearFunctionFwd_T(size(C0, 1), n[1]*n[2],
                          x -> C_fwd(x, C0, n),
                          b -> C_adj(b, C0, n),
                          Float32,Float64, name="Cmirrorext")

src_list = collect(1:dobs.nsrc)[randperm(dobs.nsrc)]
lambda = 0f0

x = zeros(Float32, size(C,2))
z = zeros(Float32, size(C,1))
tau = zeros(Float32,size(C,1))
tau1 = zeros(Float32,size(C,1))
t = zeros(Float32, niter)
flag = BitArray(undef,size(C,1))
sumsign = zeros(Float32,size(C,1))

# Main loop
for j = 1:niter
    # Select batch and set up left-hand preconditioner
    length(src_list) < batchsize && (global src_list = collect(1:dobs.nsrc)[randperm(dobs.nsrc)])
    global inds = [pop!(src_list) for b=1:batchsize]
    println("SPLS-RTM Iteration: $(j), imaging sources $(inds)")
    phi, g = lsrtm_objective(model0, q[inds], dobs[inds], Mr*x; nlind=true,options=opt)
    g = C*Mr'*vec(g)

    fval[j] = phi

    #global t[j] = 2f-5  # constant step
    global t[j] = 2*phi/norm(g)^2f0  # dynamic step

    # anti-chatter
    global sumsign = sumsign + sign.(g)
    global tau1 = t[j]*abs.(sumsign)/j
    global tau .= t[j]
    global tau[findall(flag)] = deepcopy(tau1[findall(flag)]) # use anti-chatter if pass the threshold

    # Update variables and save snapshot
    step_ = 1f-1/ceil(j/4)  # to tune
    global z -= step_*tau.*g
    (j==1) && (global lambda = quantile(abs.(z), .8))   # estimate thresholding parameter in 1st iteration
    global x = adjoint(C)*soft_thresholding(z, lambda)
    @printf("At iteration %d function value is %2.2e and step length is %2.2e \n", j, fval[j], t[j])
    @printf("Lambda is %2.2e \n", lambda)
    global flag = flag .| (abs.(z).>=lambda)  # check if pass the threshold
end