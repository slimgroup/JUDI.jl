# Parallel instance of fwi_objective function # Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

"""
    fwi_objective(model, source, dobs; options=Options())

Evaluate the full-waveform-inversion (reduced state) objective function. Returns a tuple with function value and \\
gradient. `model` is a `Model` structure with the current velocity model and `source` and `dobs` are the wavelets and \\
observed data of type `judiVector`.

Example
=======

    function_value, gradient = fwi_objective(model, source, dobs)

"""
function fwi_objective(model::Model, source::judiVector, dObs::judiVector; options=Options())
# fwi_objective function for multiple sources. The function distributes the sources and the input data amongst the available workers.

    results = judipmap(j -> fwi_objective(model, source[j], dObs[j], subsample(options, j)), 1:dObs.nsrc)

    # Collect and reduce gradients
    obj, gradient = reduce((x, y) -> x .+ y, results)

    # first value corresponds to function value, the rest to the gradient
    return obj, gradient
end
