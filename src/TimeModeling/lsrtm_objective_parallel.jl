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

    p = default_worker_pool()
    results = pmap(j -> lsrtm_objective(model, source[j], dObs[j], dm, options=subsample(options, j); nlind=nlind).
                   p, 1:dObs.nsrc)
    # Collect and reduce gradients
    objective = 0f0
    gradient = PhysicalParameter(zeros(Float32, model.n), model.d, model.o)

    for j=1:dObs.nsrc
        gradient .+= results[j][2]
        objective += results[j][1]
        results[j] = []
    end

    # first value corresponds to function value, the rest to the gradient
    return objective, gradient
end
