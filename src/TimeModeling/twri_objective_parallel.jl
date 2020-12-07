# Parallel instance of fwi_objective function # Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
export TWRIOptions

"""
    twri_objective(model, source, dobs; options=Options(), optionswri=TWRIOptions())

Evaluate the time domain Wavefield reconstruction inversion objective function. Returns a tuple with function value and \\
gradient(s) w.r.t to m and/or y. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======

    function_value, gradient = fwi_objective(model, source, dobs)

"""
function twri_objective(model::Model, source::judiVector, dObs::judiVector, y::Union{judiVector, Nothing};
                        options=Options(), optionswri=TWRIOptions())
# fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.

    p = default_worker_pool()
    twri_objective_par = remote(TimeModeling.twri_objective)
    twri_objective = retry(twri_objective_par)

    results = Array{Any}(undef, dObs.nsrc)
    @sync begin
        for j=1:dObs.nsrc
            opt_local = subsample(options,j)
            isnothing(y) ? yloc = y : yloc = y[j]
            @async results[j] = twri_objective(model, source[j], dObs[j], yloc, j;
                                               options=opt_local, optionswri=optionswri)
        end
    end

    # Collect and reduce gradients
    objective = results[1][1]
    gradientm = results[1][2]
    gradienty = results[1][3]

    for j=2:dObs.nsrc
        ~isnothing(gradientm) && (gradientm .+= results[j][2])
        ~isnothing(gradienty) && (gradienty = [gradienty; results[j][3]])
        objective += results[j][1]
        results[j] = []
    end

    # first value corresponds to function value, the rest to the gradient
    optionswri.params == :m && return objective, gradientm
    optionswri.params == :y && return objective, gradienty
    optionswri.params == :all && return objective, gradientm, gradienty
    return objective
end

# TWRI options
mutable struct TWRIOptions
    grad_corr::Bool
    comp_alpha::Bool
    weight_fun
    eps
    params::Symbol
    Invq::String
end

"""
    TWRIOptions
        grad_corr::Bool
        comp_alpha::Bool
        weight_fun
        eps
        params::Symbol
        Invq::String

Options structure for TWRI.

`grad_corr`: Whether to add the gradient correction J'(m0, q)*∇_y

`comp_alpha`: WHether to compute optimal alpha (alpha=1 if not)

`weight_fun`: Whether to apply focusing/weighting function to F(m0)'*y and its norm

`eps`: Epsilon (moise level) value (default=0)

`Invq`: How to compute F'Y, either as full field or as a rank 1 approximation `w(t)*q(x)` using the source wavelet for w

`param`: WHich gradient to compute. Choices are `nothing` (objective only), `:m`, `:y` or `:all`

Constructor
==========

All arguments are optional keyword arguments with the following default values:

TWRIOptions(;grad_corr=false, comp_alpha=true, weight_fun=nothing, eps=0, params=:m)
"""

TWRIOptions(;grad_corr=false, comp_alpha=true,
            weight_fun=nothing, eps=0, params=:m, Invq="standard")=
            TWRIOptions(grad_corr, comp_alpha, weight_fun, eps, params, Invq)
