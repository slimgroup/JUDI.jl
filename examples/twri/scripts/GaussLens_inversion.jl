################################################################################
#
# Run inversion for the Gaussian lens model
#
################################################################################

R = Float32

### Module loading
using LinearAlgebra, JLD2, PythonPlot, Dates, Distributed
using Optim, LineSearches
using JUDI

### Load or model synthetic data
base_path = dirname(pathof(JUDI))*"/../examples/twri/"
try
    @load string(base_path*"data/GaussLens_data_acou.jld") model_true dat fsrc
catch e
    @info "Data not found, modeling true data"
    include(base_path*"data/gen_data_gausslens_acou.jl")
    @load string(base_path*"data/GaussLens_data_acou.jld") model_true dat fsrc
end

### Background model
m0 = 1f0./2f0.^2*ones(R, model_true.n)
n = model_true.n
d = model_true.d
o = model_true.o
model0 = Model(n, d, o, m0)

# Pre- and post-conditioning
mask = zeros(Float32, model0.n)
mask[6:end-5, 6:end-5] .= 1
function proj_bounds(m, mmin, mmax)
    m[m[:] .< mmin] .= mmin
    m[m[:] .> mmax] .= mmax
    return m
end
mmin = 1f0/4f0^2
mmax = 1f0/1f0^2
preproc(x) = proj_bounds(x, mmin, mmax)

# [FWI]
function funFWI!(F, G, x)
    i = rand(1:15, 4)
    model0.m .= preproc(x)
    fun, g = fwi_objective(model0, fsrc[i], dat[i])
    F = fun
    isnothing(G) && return fun
    isnothing(g_const) && (global g_const = .1/maximum(abs.(g)))
    G .= g_const .* g[:] .* vec(mask)
    return fun
end

# # [TWRIdual]
ε = zeros(Float32, fsrc.nsrc)
v_bg = sqrt(1/m0[1])
freq_peak = 0.006f0
δ = R(sqrt(2)/2)*v_bg/freq_peak
weight_fun_pars = ("srcfocus", δ)
optwri = TWRIOptions(;grad_corr=true, comp_alpha=true, weight_fun=weight_fun_pars, eps=ε, params=:m)

function funWRI(F, G, x, optwri)
    i = rand(1:15, 4)
    model0.m .= preproc(x)
    fun, g = twri_objective(model0, fsrc[i], dat[i], nothing; optionswri=optwri)
    F = fun
    isnothing(G) && return fun
    isnothing(g_const) && (global g_const = .1/maximum(abs.(g)))
    G .= g_const .* g[:].* vec(mask)
    return fun
end
funWRI!(F, G, x) = funWRI(F, G, x, optwri)

### Optimization
linesearch = HagerZhang(display=1, linesearchmax=10)
# Options
method = LBFGS(;linesearch=linesearch)
niter = 20
optimopt = Optim.Options(g_calls_limit=200, iterations = niter, store_trace = true, show_trace = true, show_every = 1)

# ## Run FWI
g_const = nothing
x0_fwi = vec(m0)
result = optimize(Optim.only_fg!(funFWI!), x0_fwi, method, optimopt)
x0_fwi .= Optim.minimizer(result)
m_inv_fwi = preproc(x0_fwi)
loss_log_fwi = Optim.f_trace(result)
@save "./GaussLens_result_fwi.jld" m_inv_fwi loss_log_fwi

## Run WRI
g_const = nothing
x0_wri = vec(m0)
result = optimize(Optim.only_fg!(funWRI!), x0_wri, method, optimopt)
x0_wri .= Optim.minimizer(result)
m_inv_wri = preproc(x0_wri)
loss_log_wri = Optim.f_trace(result)
@save "./GaussLens_result_wri.jld" m_inv_wri loss_log_wri
