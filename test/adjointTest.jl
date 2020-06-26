# Adjoint test for F and J
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#
#

using PyCall, PyPlot, JUDI.TimeModeling, Images, LinearAlgebra, Test, ArgParse, Printf

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
        "--fs"
            help = "Free surface, default False"
            action = :store_true
        "--nlayer", "-n"
            help = "Number of layers"
            arg_type = Int
            default = 2
    end
    return parse_args(s)
end

parsed_args = parse_commandline()


println("Adjoint test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )
### Model
nlayer = parsed_args["nlayer"]

n = (301, 151)
d = (10., 10.)
o = (0., 0.)

v = ones(Float32,n) .* 1.5f0
vp_i = range(1.5f0, 3.5f0, length=nlayer)
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
    @printf("TTI adjoint test")
    epsilon = smooth((v0[:, :] .- 1.5f0)/12f0, sigma=3)
    delta =  smooth((v0[:, :] .- 1.5f0)/14f0, sigma=3)
    theta =  smooth((v0[:, :] .- 1.5f0)/4, sigma=3)
    model0 = Model_TTI(n,d,o,m0; epsilon=epsilon, delta=delta, theta=theta)
    model = Model_TTI(n,d,o,m; epsilon=epsilon, delta=delta, theta=theta)
    jtol = 1f-2
else
    model = Model(n,d,o,m,rho=rho0)
    model0 = Model(n,d,o,m0,rho=rho0)
    jtol = 1f-3
end

## Set up receiver geometry
nxrec = 141
xrec = range(d[1],stop=(n[1] - 2) * d[1],length=nxrec)
yrec = 0f0
zrec = range(5*d[1],stop=5*d[1],length=nxrec)

# Sampling and recording time
time = 1400f0	# receiver recording time [ms]
dt = 1.0

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dt,t=time,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = (n[1] - 1) * d[1] / 2f0
ysrc = 0f0
zsrc = 5*d[1]

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dt,t=time)

# Info structure
ntComp = get_computational_nt(srcGeometry, recGeometry, model0; dt=dt)
info = Info(prod(n), nsrc, ntComp)

# setup wavelet
f0 = 0.015f0
wavelet = ricker_wavelet(time,dt,f0)
wave_rand = wavelet.*rand(Float32,size(wavelet))

###################################################################################################
# Modeling

# Modeling operators
opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"])

F = judiModeling(info,model0,srcGeometry,recGeometry; options=opt)
q = judiVector(srcGeometry,wavelet)

# Nonlinear modeling
y = F*q

# Generate random noise data vector with size of d_hat in the range of F
x = judiVector(srcGeometry, wave_rand)

# Forward-adjoint 
y_hat = F*x
x_hat = adjoint(F)*y

# Result F
a = dot(y, y_hat)
b = dot(x, x_hat)
@printf(" <F x, y> : %2.2e, <x, F' y> : %2.2e, relative error : %2.2e \n", a, b, a/b - 1)
@test isapprox(a/b - 1f0, 0, atol=1f-5)

# Linearized modeling
J = judiJacobian(F,q)
x = vec(dm)

y_hat = J*x
x_hat = adjoint(J)*y

c = dot(y, y_hat)
d = dot(x, x_hat)
@printf(" <J x, y> : %2.2e, <x, J' y> : %2.2e, relative error : %2.2e \n", c, d, c/d - 1)
@test isapprox(c/d - 1f0, 0, atol=jtol)


###################################################################################################
# Extended source modeling

println("Extended source adjoint test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )

opt = Options(return_array=true, sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"])

Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model0; options=opt)
Pw = judiLRWF(info, wavelet)
F = Pr*F*adjoint(Pw)

# Extended source weights
w = vec(randn(Float32, model0.n))
x = vec(randn(Float32, model0.n))

# Generate random noise data vector with size of d_hat in the range of F
y = F*w

# Forward-Adjoint computation
y_hat = F*x
x_hat = adjoint(F)*y

# Result F
a = dot(y, y_hat)
b = dot(x, x_hat)
@printf(" <F x, y> : %2.2e, <x, F' y> : %2.2e, relative error : %2.2e \n", a, b, a/b - 1)
@test isapprox(a/b - 1, 0, atol=1f-4)

# Linearized modeling
J = judiJacobian(F, w)
x = vec(dm)

y_hat = J*x
x_hat = adjoint(J)*y

c = dot(y, y_hat)
d = dot(x, x_hat)
@printf(" <J x, y> : %2.2e, <x, J' y> : %2.2e, relative error : %2.2e \n", c, d, c/d - 1)
@test isapprox(c/d - 1, 0, atol=1f-1)
