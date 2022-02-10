# Example for 2D sparsity-promoting LS-RTM via Linearized Bregman

# Couple of references:

# 1. sparsity-promoting LS-RTM
# Philipp A. Witte, Mathias Louboutin, Fabio Luporini, Gerard J. Gorman, and Felix J. Herrmann
# “Compressive least-squares migration with on-the-fly Fourier transforms”
# Geophysics, vol. 84, pp. R655-R672, 2019.


# 2. inverse-scattering imaging condition
# Philipp A. Witte, Mengmeng Yang, and Felix J. Herrmann
# “Sparsity-promoting least-squares migration with the linearized inverse scattering imaging condition”
# EAGE Annual Conference Proceedings, 2017.

# 3. anti chattering:
# Emmanouil Daskalakis, Felix J. Herrmann, Rachel Kuske
# "Accelerating Sparse Recovery by Reducing Chatter"
# SIAM Journal on Imaging Sciences, 2020.

# 4. Linearized Bregman
# Jian-Feng Cai, Stanley Osher and Zuowei Shen
# "Linearized Bregman iterations for compressed sensing"
# Mathematics of computation, 2009.

# 5. Computational simultaneous sources, a.k.a. super shots
# Xiang Li, Aleksandr Y. Aravkin, Tristan van Leeuwen, Felix J. Herrmann
# "Fast randomized full-waveform inversion with compressive sensing"
# Geophysics, 2012

# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: Feb, 2022
#

using JUDI, JOLI, LinearAlgebra, Random, Statistics, Images

# Set up model structure
n = (100, 100)   # (x,y,z) or (x,z)
d = (10., 10.)
o = (0., 0.)

# Velocity [km/s]
v = ones(Float32,n) .+ 0.5f0
v[:,20:end] .= 2.5f0
v[:,50:end] .= 3.5f0
v0 = 1f0./convert(Matrix{Float32}, imfilter(1f0./v, Kernel.gaussian(2))) # smooth 1/v

# Slowness squared [s^2/km^2]
m = (1f0 ./ v).^2
m0 = (1f0 ./ v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 4	# number of sources
model = Model(n, d, o, m)
model0 = Model(n, d, o, m0)

# Set up receiver geometry
nxrec = 120
xrec = range(d[1], stop=(n[1]-1)*d[1], length=nxrec)
yrec = 0f0
zrec = range(50f0, stop=50f0, length=nxrec)

# receiver sampling and recording time
timeR = 1000f0   # receiver recording time [ms]
dtR = 2f0    # receiver sampling interval [ms]

# Set up receiver structure
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(range(d[1], stop=(n[1]-1)*d[1], length=nsrc))
ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

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

# inverse-scattering imaging condition
opt = Options(isic=true)

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Generate data via nonlinear modeling
dobs = Pr*F*adjoint(Ps)*q
dobs0 = Pr*F0*adjoint(Ps)*q

# Get linearize data for LS-RTM
dlin = dobs-dobs0

# Preconditioner
idx_wb = find_water_bottom(m-m0)
Tm = judiTopmute(model0.n, idx_wb, 3)  # Mute water column
S = judiDepthScaling(model0)
Mr = Tm*S

# Linearized Bregman parameters

batchsize = 2       # num of sources in each iteration
niter = 4           # num of iterations (2 data pass)

fval = zeros(Float32, niter)

# Soft thresholding functions
soft_thresholding(x::Array{Float64}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float64, lambda), 0.0)
soft_thresholding(x::Array{Float32}, lambda) = sign.(x) .* max.(abs.(x) .- convert(Float32, lambda), 0f0)

# sparsifying transform (wavelet transform is used in this case)
C = joDWT(n[1], n[2]; DDT=Float32)


# set up linearized bregman variables
x = zeros(Float32, size(C,2))   # primal
z = zeros(Float32, size(C,1))   # dual
t = zeros(Float32, niter)       # step length

anti_chatter = true

# set up anti-chattering variables, please check the SIAM 2020 paper for details
if anti_chatter
    flag_ = BitArray(undef,size(C,1))
    sumsign = zeros(Float32,size(C,1))
end

# set up computational simultaneous source experiments
# set up source geometry
xsrc_sim = [[q.geometry.xloc[i][1] for i = 1:nsrc] for j = 1:batchsize]
ysrc_sim = [[0.0f0] for j = 1:batchsize]
zsrc_sim = [[q.geometry.zloc[i][1] for i = 1:nsrc] for j = 1:batchsize]
sim_src_geometry = Geometry(xsrc_sim, ysrc_sim, zsrc_sim; dt=q.geometry.dt[1], t=q.geometry.t[1])
ntComp_sim = get_computational_nt(sim_src_geometry, recGeometry, model)
info_sim = Info(prod(n), batchsize, ntComp_sim)
Ps_sim = judiProjection(info_sim, sim_src_geometry)

# set up receiver geometry
sim_rec_geometry = Geometry(dlin.geometry.xloc[1],dlin.geometry.yloc[1],dlin.geometry.zloc[1]; dt=dobs.geometry.dt[1], t=dlin.geometry.t[1], nsrc=batchsize)
Pr_sim = judiProjection(info_sim, sim_rec_geometry)

# set up modeling
F0_sim = judiModeling(info_sim, model0; options=opt)


# Main loop
for j = 1:niter
    # Set up random weights for current iteration
    weights = randn(Float32,batchsize,nsrc)

    # Create wavelet according to the random weights
    wavelet_sim = Vector{Matrix{Float32}}(undef, batchsize)
    for k = 1:batchsize
        wavelet_sim[k] = zeros(Float32, q.geometry.nt[1], nsrc)
        wavelet_sim[k][:,1:nsrc] = q.data[1]*weights[k:k,:]
    end
        
    # super shot for current iteration
    q_j = judiVector(sim_src_geometry, wavelet_sim)

    # set up linearized modeling operator
    J_j = judiJacobian(Pr_sim*F0_sim*adjoint(Ps_sim), q_j)

    # superposition of shot records
    data_j = Vector{Matrix{Float32}}(undef, batchsize)
    for k = 1:batchsize
        data_j[k] = sum(weights[k,:].*dlin.data)
    end

    # simultaneous shot record for current iteration
    dlin_j = judiVector(sim_rec_geometry, data_j)

    println("SPLS-RTM Iteration: $(j), use $(batchsize) super shots")

    # calculate residual and gradient with nice-looking LinearAlgebra
    r = J_j * Mr * x - dlin_j
    fval[j] = 0.5f0 * norm(r)^2f0
    g = C * Mr' * J_j' * r

    # step length
    t[j] = Float32(norm(r)^2f0/norm(g)^2f0)  # dynamic step

    if anti_chatter
        # anti-chattering, please check the SIAM 2020 paper for details
        global sumsign += sign.(g)
        tau1 = t[j]*abs.(sumsign)/j
        tau = t[j] * ones(Float32,size(C,1))
        @views tau[findall(flag_)] = tau1[findall(flag_)]

        # update dual variable
        @. global z -= t[j] * tau .* g
    else
        # update dual variable
        @. global z -= t[j] * g
    end

    # estimate thresholding parameter in 1st iteration
    (j==1) && (global lambda = quantile(abs.(z), .8))
    
    # update primal variable
    global x = adjoint(C)*soft_thresholding(z, lambda)

    # iterative log
    println("At iteration $j function value is $(fval[j]) and step length is $(t[j])")

    if anti_chatter
        global flag_ = flag_ .| (abs.(z).>=lambda)  # check if pass the threshold
    end
end 
