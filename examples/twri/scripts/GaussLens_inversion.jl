################################################################################
#
# Run inversion for the Gaussian lens model
#
################################################################################

R = Float32

### Module loading


using LinearAlgebra, JLD2, PyPlot, Dates, Distributed
using Optim, LineSearches
using JUDI, JUDI.TimeModeling, JUDI.SLIM_optim

### Load synthetic data
base_path = dirname(pathof(JUDI))*"/../examples/twri/"
@load string(base_path*"data/GaussLens_data_acou.jld") model_true dat fsrc

### Background model
m0 = 1f0./2f0.^2*ones(R, model_true.n)
n = model_true.n
d = model_true.d
o = model_true.o
model0 = Model(n, d, o, m0)

# Pre- and post-conditioning
mask = zeros(Float32, model0.n)
mask[6:end-5, 6:end-5] .= 1
function proj_bounds(m, mmin, mmax, mask)
    m = m
    m[m[:] .< mmin] .= mmin
    m[m[:] .> mmax] .= mmax
    return m
end
mmin = 1f0/4f0^2
mmax = 1f0/1f0^2
preproc(x) = proj_bounds(x, mmin, mmax, mask)

g_const = nothing

# [FWI]
function funFWI!(F, G, x)
    model0.m .= preproc(x)
    fun, g = fwi_objective(model0, fsrc, dat)
    F = fun
    isnothing(G) && return fun
    isnothing(g_const) && (global g_const = .1/maximum(abs.(g)))
    G .= g_const .* g[:] .* vec(mask)
    return fun
end

# # [TWRIdual]
ε0 = 0.01f0
ε = Array{Float32, 1}(undef, fsrc.nsrc)
for i = 1:fsrc.nsrc
    ε[i] = ε0*norm(dat.data[i])*sqrt(dat.geometry.dt[i])
end
v_bg = sqrt(1/m0[1])
freq_peak = 0.006f0
δ = 1f0*R(sqrt(2)/2)*v_bg/freq_peak
weight_fun_pars = ("srcfocus", δ)
optwri = TWRIOptions(;grad_corr=false, comp_alpha=true, weight_fun=weight_fun_pars, eps=ε, params=:m)

function funWRI!(F, G, x)
    model0.m .= preproc(x)
    isnothing(G) ? optwri.params = nothing : optwri.params = :m
    fun, g = twri_objective(model0, fsrc, dat, nothing; optionswri=optwri)
    F = fun
    isnothing(G) && return fun
    isnothing(g_const) && (global g_const = .1/maximum(abs.(g)))
    G .= g_const .* g[:].* vec(mask)
    return fun
end

# Gradients

# x0 = vec(m0)
# g0fwi = zeros(R, length(model0.m))
# g0fwri = zeros(R, length(model0.m))
# F = 0f0

# funWRI!(F, g0fwri, x0)
# funFWI!(F, g0fwi, x0)

### Optimization

linesearch = HagerZhang(display=1, linesearchmax=5)
# Options
method = LBFGS(;linesearch=linesearch)
niter = 20
optimopt = Optim.Options(iterations = niter, store_trace = true, show_trace = true, show_every = 1)

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
