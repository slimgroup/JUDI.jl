# Adjoint test for F and J
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
#

using PyCall, PyPlot, JUDI.TimeModeling, Images, LinearAlgebra, Test, ArgParse

###
function smooth(v; sigma=5)
    return Float32.(imfilter(v, Kernel.gaussian(sigma)))
end
### Process command line args
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--tti"
            help = "TTI, default False"
            action = :store_true
        "--nlayer", "-n"
            help = "Number of layers"
            arg_type = Int
            default = 2
    end
    return parse_args(s)
end
parsed_args = parse_commandline()
### Model
nlayer = parsed_args["nlayer"]
n = (301, 151)
d = (10., 10.)
o = (0., 0.)
v = ones(Float32,n) .* 1.5f0
vp_i = range(1.5f0, 4.5f0, length=nlayer)
for i in range(2, nlayer, step=1)
    v[:, (i-1)*Int(floor(n[2] / nlayer)) + 1:end] .= vp_i[i]  # Bottom velocity
end

v0 = smooth(v, sigma=10)
v0[v .< 1.51] .= v[v .< 1.51]
v0 = smooth(v0, sigma=3)
rho0 = (v0 .+ .5f0) ./ 2

m0 = v0.^(-2f0)
m = v.^(-2f0)
dm = m0 .- m


# Setup info and model structure
nsrc = 1
if parsed_args["tti"]
    epsilon = smooth((v0[:, :] .- 1.5f0)/12f0, sigma=3)
    delta =  smooth((v0[:, :] .- 1.5f0)/14f0, sigma=3)
    theta =  smooth((v0[:, :] .- 1.5f0)/4, sigma=3)
    model0 = Model_TTI(n,d,o,m0; epsilon=epsilon, delta=delta, theta=theta)
    model = Model_TTI(n,d,o,m; epsilon=epsilon, delta=delta, theta=theta)
else
    model = Model(n,d,o,m,rho=rho0)
    model0 = Model(n,d,o,m0,rho=rho0)
end
## Set up receiver geometry
nxrec = 141
xrec = range(100f0,stop=900f0,length=nxrec)
yrec = 0f0
zrec = range(50f0,stop=50f0,length=nxrec)

# receiver sampling and recording time
timeR = 1400f0	# receiver recording time [ms]
dtR = calculate_dt(model)    # receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = 500f0
ysrc = 0f0
zsrc = 50f0

# source sampling and number of time steps
timeS = 1400f0
dtS = calculate_dt(model) # receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# Info structure
ntComp = get_computational_nt(srcGeometry,recGeometry,model0)
info = Info(prod(n), nsrc, ntComp)

# setup wavelet
f0 = 0.015f0
wavelet = ricker_wavelet(timeS,dtS,f0)
wave_rand = wavelet.*rand(Float32,size(wavelet))

###################################################################################################

# Modeling operators
opt = Options(sum_padding=true, isic=false, t_sub=1, h_sub=1, free_surface=false)
F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
q = judiVector(srcGeometry, wavelet)

# Nonlinear modeling
d_hat = F*q

# Generate random noise data vector with size of d_hat in the range of F
qr = judiVector(srcGeometry, wave_rand)
d1 = F*qr

# Adjoint computation
q_hat = adjoint(F)*d_hat

# Result F
a = dot(d1, d_hat)
b = dot(qr, q_hat)
println(a, ", ", b, " ", 1 - a/b)
# @test isapprox(a/b - 1, 0, atol=1f-4)

# Linearized modeling
J = judiJacobian(F0,q)

dD_hat = J*dm
dm_hat = adjoint(J)*dD_hat

c = dot(dD_hat, dD_hat)
d = dot(dm, dm_hat)

println(c, ", ", d, " ", 1 - c/d)
@test isapprox(c/d - 1, 0, atol=1f-4)
