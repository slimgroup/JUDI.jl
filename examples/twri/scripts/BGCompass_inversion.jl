################################################################################
#
# Run inversion for the BG Compass model
#
################################################################################

### Module loading
using LinearAlgebra, JLD2, PythonPlot, Dates, Distributed, Images
using Optim, LineSearches
using JUDI, SlimOptim

### Load synthetic data
base_path = dirname(pathof(JUDI))*"/../examples/twri/"
try
    @load string(base_path*"data/BGCompass_data_acou.jld") model_true dat fsrc
catch e
    @info "Data not found, modeling true data"
    include(base_path*"data/gen_data_bg_acou.jl")
    @load string(base_path*"data/BGCompass_data_acou.jld") model_true dat fsrc
end
### Background mode
idx_w = 1
var = 20
m0 = deepcopy(model_true.m)
m0[:, idx_w+1:end] = R.(imfilter(m0[:, idx_w+1:end], Kernel.gaussian(var)))
model0 = model_true
model0.m = m0
n = model0.n

### Set objective functional

# Pre- and post-conditioning
mask = zeros(Float32, model0.n)
mask[:, idx_w+1:end] .= 1
function proj_bounds(m, mmin, mmax)
    m[m[:] .< mmin] .= mmin
    m[m[:] .> mmax] .= mmax
    return m
end
mmin = 1f0/6f0^2
mmax = .5f0
preproc(x) = proj_bounds(x, mmin, mmax)

# [FWI]
function funFWI!(F, G, x, dat, fsrc)
    model0.m .= preproc(x)
    fun, g = fwi_objective(model0, fsrc, dat)
    F = fun
    isnothing(G) && return fun
    isnothing(g_const) && (global g_const = .1/maximum(abs.(g)))
    G .= g_const .* g[:] .* vec(mask)
    return fun
end

# # [TWRIdual]
ε0 = zeros(Float32, dat.nsrc)
weight_fun_pars = ("srcfocus", 0)
optwri = TWRIOptions(;grad_corr=true, comp_alpha=false, weight_fun=weight_fun_pars, eps=ε, params=:m)

function funWRI!(F, G, x, dat, fsrc, freq)
    # Frequency specific focusing
    δ = sqrt(2)*m0[1]^(-.5)/(2*freq)
    optwri.weight_fun = ("srcfocus", δ)
    # Compute objective and gradient
    model0.m .= preproc(x)
    fun, g = twri_objective(model0, fsrc, dat, nothing; optionswri=optwri)
    F = fun
    isnothing(g_const) && (global g_const = .1/maximum(abs.(g)))
    G .= g_const .* g[:].* vec(mask)
    return fun
end

### Optimization
linesearch = HagerZhang(display=1, linesearchmax=5)
method = LBFGS(;linesearch=linesearch)
niter = 10
optimopt = Optim.Options(iterations = niter, store_trace = true, show_trace = true, show_every = 1)

## Starting guess
x0_fwi = vec(m0)
x0_wri = vec(m0)

freq_list = 5:2:19

for f in freq_list
    println("\nInverting freqs: ", string(f-1), " - ", string(f+1), " Hz\n")
    dat_loc = low_filter(dat, 4; fmin=f-1, fmax=f+1)	
    fsrc_loc = low_filter(fsrc, 4; fmin=f-1, fmax=f+1)
    fwi!(F, G, x) = funFWI!(F, G, x, dat_loc, fsrc_loc)
    wri!(F, G, x) = funWRI!(F, G, x, dat_loc, fsrc_loc, f)

    # FWI
    result = optimize(Optim.only_fg!(fwi!), x0_fwi, method, optimopt)
    global x0_fwi = Optim.minimizer(result)
    m_inv_fwi = preproc(x0_fwi)

    # WRI
    result = optimize(Optim.only_fg!(wri!), x0_wri, method, optimopt)
    global x0_wri = Optim.minimizer(result)
    m_inv_wri = preproc(x0_wri)

    # Save iter
    @save "$(base_path)/data/BG_compass_acou_sweep_$(f)Hz.jld" f m_inv_fwi m_inv_wri n
end
