################################################################################
#
# Run inversion for the BG Compass model
#
################################################################################

### Module loading
using LinearAlgebra, Dates, Distributed, Images, Random
using JUDI, JUDI.SLIM_optim, JLD2

R = Float32
### Load synthetic data
base_path = dirname(pathof(JUDI))*"/../examples/twri/"
try
    @load string(base_path*"data/BGCompass_data_tti.jld") model_true dat fsrc
catch e
    @info "Data not found, modeling true data"
    include(base_path*"data/gen_data_bg_tti.jl")
    @load string(base_path*"data/BGCompass_data_tti.jld") model_true dat fsrc
end
# Different sigmas for smoothing
vvar = [20, 25, 30, 40]
# Global parameters
idx_w = 17
vare = 15
fevals = 20
batchsize = 25

# Bound projection
function proj_bounds(m, mmin, mmax)
    m[m[:] .< mmin] .= mmin
    m[m[:] .> mmax] .= mmax
    return m
end
mmin = 1f0/4.5f0^2
mmax = 1f0/1.4f0^2
ProjBound(x) = proj_bounds(x, mmin,  mmax)

for var=vvar
    for anis=["tt", "st"]
        ### Background model
        model0 = deepcopy(model_true)
        model0.m[:, idx_w+1:end] = R.(imfilter(model0.m[:, idx_w+1:end], Kernel.gaussian(var)))
        if anis == "st"
            model0.epsilon[:, idx_w+1:end] = R.(imfilter(model0.epsilon[:, idx_w+1:end], Kernel.gaussian(vare)))
            model0.delta[:, idx_w+1:end] = R.(imfilter(model0.delta[:, idx_w+1:end], Kernel.gaussian(vare)))
            model0.theta[:, idx_w+1:end] = R.(imfilter(model0.theta[:, idx_w+1:end], Kernel.gaussian(vare)))
        end
        m0 = model0.m

        ### Set objective functional
        mask = zeros(Float32, model0.n)
        mask[:, idx_w+1:end] .= 1

        # [TWRIdual]
        ε = zeros(Float32, fsrc.nsrc)

        ## Run
        # Very basic approximate inverse Hessian
        DZ = judiDepthScaling(model0)

        # Optimization parameters
        optwri = TWRIOptions(;grad_corr=false, comp_alpha=false, weight_fun=nothing, eps=ε, params=:m)

        # Objective function for minConf library
        function objective_function(x, fun="fwi")
            flush(Base.stdout)
            x = reshape(x, model0.n)
            # fwi function value and gradient
            i = randperm(dat.nsrc)[1:batchsize]
            # weight_fun_pars=weight_fun_pars
            if fun == "wri"
                fval, grad = twri_objective(model0, fsrc[i], dat[i], nothing; optionswri=optwri)
            else
                fval, grad = fwi_objective(model0, fsrc[i], dat[i])
            end
            grad = DZ*(vec(mask) .* vec(grad))
            grad = .1f0 .* grad ./ maximum(abs.(grad))  # scale for line search
            return fval, grad
        end

        wri_fun(x) = objective_function(x, "wri")
        fwi_fun(x) = objective_function(x, "fwi")

        # FWI with SPG
        x0 = vec(m0)
        options = spg_options(verbose=3, maxIter=fevals, memory=3, interp=0)

        sol = spg(wri_fun, x0, ProjBound, options)
        xwri, hwri = sol.x, sol.ϕ_trace
        @save string(base_path*"data/wriw_$(anis)_$(var).jld") xwri hwri

        x0 = vec(m0)
        sol = spg(fwi_fun, x0, ProjBound, options)
        xfwi, hfwi = sol.x, sol.ϕ_trace
        @save string(base_path*"data/fwiw_$(anis)_$(var).jld") xfwi hfwi
    end
end
