# Parallel instance of lsrtm_objective function
# Author: Mathias Louboutin, mloubutin3@gatech.edu
# Date: September 202-
#

"""
    lsrtm_objective(model, source, dobs, dm; options=Options())

Evaluate the least-square migration objective function. Returns a tuple with function value and \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======

    function_value, gradient = lsrtm_objective(model, source, dobs, dm)

"""
function lsrtm_objective(model::Model, source::judiVector, dObs::judiVector, dm; options=Options(), nlind=false)
# lsrtm_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.

    obj, gradient = judipmap(j -> lsrtm_objective(model, source[j], dObs[j], dm, subsample(options, j); nlind=nlind), 1:dObs.nsrc, sum!)

    return obj[1], gradient
end

function lsrtm_objective(model::Array{Model,1}, source::Array{judiVector{T,Array{T,2}},1}, dObs::Array{judiVector{T,Array{T,2}},1}, dm::Union{Array{Array{T,1},1}, Array{PhysicalParameter{T},1}}; options=Options(), nlind=false) where T
# lsrtm_objective function for multiple sources and multiple vintages. The function distributes the sources and the input data amongst the available workers.

    results = judipmap((m, q, d, d_m) -> lsrtm_objective(m, q, d, d_m; options=options, nlind=nlind), model, source, dObs, dm)

    obj = sum([results[i][1] for i = 1:length(results)])
    gradient = [results[i][2] for i = 1:length(results)]

    # first value corresponds to function value, the rest to the gradient
    return obj, gradient
end
