# Parallel instance of fwi_objective function # Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
export TWRIOptions

red_funcs = Dict(:m => (sum!, sum!), :y => (vcat!, sum!), :all => (sum!, vcat!, sum!), nothing => (sum!,))
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
    red_op! = red_funcs(optionswri.params)

    results = judipmap(j -> twri_objective(model, source[j], dObs[j], subsample(y, j), subsample(options,j), subsample(optionswri,j)), 1:dObs.nsrc, red_op!...)

    return results
end

# TWRI options
mutable struct TWRIOptions
    grad_corr::Bool
    comp_alpha::Bool
    weight_fun
    eps
    params
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


function subsample(opt::TWRIOptions, srcnum::Int)
    eloc = length(opt.eps) == 1 ? opt.eps : opt.eps[srcnum]
    return TWRIOptions(opt.grad_corr, opt.comp_alpha, opt.weight_fun, eloc, opt.params, opt.Invq)
end