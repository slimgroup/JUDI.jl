# Parallel instance of lsrtm_objective function
# Author: Mathias Louboutin, mloubutin3@gatech.edu
# Date: September 202-
#

"""
    lsrtm_objective(model, source, dobs, dm; options=Options())

Evaluate the least-square migration objective function. Returns a tuple with function value and vectorized \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======

    function_value, gradient = lsrtm_objective(model, source, dobs, dm)

"""
function lsrtm_objective(model::Model, source::judiVector, dObs::judiVector, dm; options=Options(), nlind=false)
# lsrtm_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.

    p = default_worker_pool()
    lsrtm_objective_par = remote(TimeModeling.lsrtm_objective)
    lsrtm_objective = retry(lsrtm_objective_par)

    results = Array{Any}(undef, dObs.nsrc)

    @sync begin
        for j=1:dObs.nsrc
            opt_local = subsample(options,j)
            @async results[j] = lsrtm_objective(model, source[j], dObs[j], j, dm; options=opt_local, nlind=nlind)
        end
    end

    # Collect and reduce gradients
    objective =results[1][1]
    gradient = results[1][2]

    for j=2:dObs.nsrc
        gradient .+= results[j][2]
        objective += results[j][1]
        results[j] = []
    end

    # first value corresponds to function value, the rest to the gradient
    return objective, gradient
end
