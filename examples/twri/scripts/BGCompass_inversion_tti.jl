################################################################################
#
# Run inversion for the BG Compass model
#
################################################################################



### Module loading


using LinearAlgebra, Dates, Distributed, Images, Random
using JUDI.TimeModeling, JUDI.SLIM_optim
@everywhere using JLD2
@everywhere push!(LOAD_PATH, string(pwd(), "/src/")); @everywhere using TWRIdual

### Load synthetic data
@everywhere @load "./data/BGCompass/BGCompass_data_tti.jld"

vvar = [20, 25, 30, 40]

for i=4:4
### Background model
idx_w = 17
var = vvar[i]
vare = 15
@everywhere model0 = deepcopy(model_true)
model0.m[:, idx_w+1:end] = R.(imfilter(model0.m[:, idx_w+1:end], Kernel.gaussian(var)))
model0.epsilon[:, idx_w+1:end] = R.(imfilter(model0.epsilon[:, idx_w+1:end], Kernel.gaussian(vare)))
model0.delta[:, idx_w+1:end] = R.(imfilter(model0.delta[:, idx_w+1:end], Kernel.gaussian(vare)))
model0.theta[:, idx_w+1:end] = R.(imfilter(model0.theta[:, idx_w+1:end], Kernel.gaussian(vare)))
m0 = model0.m

### Set objective functional
# Pre- and post-conditioning
@everywhere mask = BitArray(undef, model0.n)
mask .= false
mask[:, idx_w+1:end] .= true

mmin = 1f0/4.5f0^2
mmax = 1f0/1.4f0^2

# [TWRIdual]
inv_name = "TWRIdual"

ε0 = 0.0f0
ε = Array{Float32, 1}(undef, fsrc.nsrc)
ε .= ε0
grad_corr = true
objfact = 1f0
v_bg = sqrt(1/m0[1])

## Run

freq_peak = 0.003f0
δ = 1f0*R(sqrt(2)/2)*v_bg/freq_peak
weight_fun_pars = ("srcfocus", δ)

# Msg

DZ = judiDepthScaling(model0)
# Optimization parameters
#srand(1)    # set seed of random number generator
fevals = 20
batchsize = 25

# Objective function for minConf library
function objective_function(x, fun="fwi")
    flush(Base.stdout)
    x = reshape(x, model0.n)
    # fwi function value and gradient
    i = randperm(dat.nsrc)[1:batchsize]
    # weight_fun_pars=weight_fun_pars
    if fun == "wri"
        fval, grad = objTWRIdual(x, model0, nothing, fsrc[i], dat[i], ε;
 	                         mode = "grad", comp_alpha = true, grad_corr = false)
    else
        fval, grad = objFWI(x, model0, fsrc[i], dat[i]; mode="grad")
    end
    grad = reshape(DZ*vec(grad), model0.n)
    grad[:, 1:idx_w] .= 0.0f0
    grad = .1f0 .* grad ./ maximum(abs.(grad))  # scale for line search
    return fval, vec(grad)
end

wri_fun(x) = objective_function(x, "wri")
fwi_fun(x) = objective_function(x, "fwi")
# Bound projection
function proj_bounds(m, mmin, mmax)
    m[m[:] .< mmin] .= mmin
    m[m[:] .> mmax] .= mmax
    return m
end
ProjBound(x) = proj_bounds(x, mmin,  mmax)

# FWI with SPG
x0 = vec(m0)
options = spg_options(verbose=3, maxIter=fevals, memory=3, interp=0)


xwri, fsavewri, funEvals, p, hwri = minConf_SPG(wri_fun, x0, ProjBound, options)
xwri = reshape(xwri, model0.n)
@save "./results/tti/TTI_true_anis/wriw"*string(var)*".jld" xwri hwri

x0 = vec(m0)
xfwi, fsavefwi, funEvals, p, hfwi = minConf_SPG(fwi_fun, x0, ProjBound, options)
xfwi = reshape(xfwi, model0.n)
@save "./results/tti/TTI_true_anis/fwiw"*string(var)*".jld" xfwi hfwi
end
