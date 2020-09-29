################################################################################
#
# Run inversion for the BG Compass model
#
################################################################################



### Module loading


using LinearAlgebra, JLD2, PyPlot, Dates, Distributed, Images
using Optim, LineSearches
using JUDI.TimeModeling, JUDI.SLIM_optim
@everywhere push!(LOAD_PATH, string(pwd(), "/src/")); @everywhere using TWRIdual



### Load synthetic data


@load "./data/BGCompass/BGCompass_data_tti.jld"



### Background model


idx_w = 1
var = 20
m0 = deepcopy(model_true.m)
m0[:, idx_w+1:end] = R.(imfilter(m0[:, idx_w+1:end], Kernel.gaussian(var)))
# n = model_true.n
# d = model_true.d
# o = model_true.o
# model0 = Model(n, d, o, m0)
model0 = model_true
model0.m = m0

### Set objective functional


# Time sampling
dt_comp = dat.geometry.dt


# Pre- and post-conditioning
@everywhere mask = BitArray(undef, model0.n); mask .= false
mask[2:end-1, idx_w+1:end-1] .= true
function proj_bounds(m, mmin, mmax)
    m[m[:] .< mmin] .= mmin
    m[m[:] .> mmax] .= mmax
    return m
end
# function proj_bounds_contr(x, mask, m0, mmin, mmax)
#     m = contr2abs(x, mask, m0)
#     m[m[:] .< mmin] .= mmin
#     m[m[:] .> mmax] .= mmax
#     return ((m.-m0)./m0)[mask]
# end
mmin = 1f0/6f0^2
mmax = Inf
preproc(x) = proj_bounds(contr2abs(x, mask, m0), mmin, mmax)
# preproc(x) = contr2abs(x, mask, m0)
fun_proj(x) = proj_bounds_contr(x, mask, model0.m, mmin, mmax)
@everywhere postproc(g) = gradprec_contr2abs(g, mask, m0)


# [FWI]
inv_name = "FWI"
fun!(F, G, x, Filter) = objFWI!(F, G, preproc(x), n, d, o, fsrc, dat; dt_comp = dt_comp, Filter = Filter, gradmprec_fun = postproc)


# # [TWRIdual]
# inv_name = "TWRIdual"
# # ε0 = 0.01f0
# ε0 = 0.0f0
# ε = Array{Float32, 1}(undef, fsrc.nsrc); ε .= ε0
# grad_corr = true
# # grad_corr = false
# # objfact = 0.00017248749f0
# # objfact = 1f-4
# objfact = 1f0
# # objfact = 0.00012864293f0
# fun!(F, G, x, weight_fun_pars, objfact, Filter) = objTWRIdual!(F, G, preproc(x), n, d, o, fsrc, dat, ε; objfact = objfact, comp_alpha = true, grad_corr = grad_corr, weight_fun_pars = weight_fun_pars, dt_comp = dt_comp, Filter = Filter, gradmprec_fun = postproc)
#
# println(inv_name)
# # println(string(objfact))



### Optimization

## Optimization method

method = LBFGS()
println("\n", method)


## Optimizer options

# niter = 3
# niter = 20
niter = 10
optimopt = Optim.Options(iterations = niter, store_trace = true, show_trace = true, show_every = 1)


## Starting guess

# x0 = zeros(R, length(findall(mask .== true)))
# @load "./data/BGCompass/results/MultRun_TWRI_5Hz/curr_freq_3.jld"
# @load string("./data/BGCompass/results/MultRun_TWRI_5Hz/BGCompass_result_", string(Int64(floor(1f3*freq))), "Hz_3.jld") m_inv
@load "./data/BGCompass/results/MultRun_FWI_5Hz/curr_freq.jld"
@load string("./data/BGCompass/results/MultRun_FWI_5Hz/BGCompass_result_", string(Int64(floor(1f3*freq))), "Hz.jld") m_inv

x0 = abs2contr(m_inv, mask, m0)
m_inv = preproc(x0)


## Run

# Selecting freq filter
freq += 0.002f0
cfreqs = (freq-0.001f0, freq, freq, freq+0.001f0)
Filter = cfreqfilt(2*dat.geometry.nt[1]+1, dt_comp[1], cfreqs)

# Msg
println("\nInverting freqs: ", string(1f3*cfreqs[2]), " - ", string(1f3*cfreqs[3]), " Hz\n")

# Setting objective for current frequency bandwidth
# v_bg = sqrt(1/m0[1])
# δ = 0.1f0*R(sqrt(2)/2)*v_bg/freq
# weight_fun_pars = ("srcfocus", δ)
# fun_freq!(F, G, x) = fun!(F, G, x, weight_fun_pars, objfact, Filter)
fun_freq!(F, G, x) = fun!(F, G, x, Filter)

# Optimization
result = optimize(Optim.only_fg!(fun_freq!), x0, method, optimopt)
x0 = Optim.minimizer(result)

# # Save result
# m_inv = preproc(x0)
# # @save string("./data/BGCompass/results/MultRun_TWRI_5Hz/BGCompass_result_", string(Int64(floor(1f3*freq))), "Hz_3.jld") m_inv
# # @save "./data/BGCompass/results/MultRun_TWRI_5Hz/curr_freq_3.jld" freq
# @save string("./data/BGCompass/results/MultRun_FWI_5Hz/BGCompass_result_", string(Int64(floor(1f3*freq))), "Hz.jld") m_inv
# @save "./data/BGCompass/results/MultRun_FWI_5Hz/curr_freq.jld" freq
