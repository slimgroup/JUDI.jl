extended_source_modeling(model::Model, srcData, recGeometry, recData, weights, perturbation, srcnum::UnitRange{Int64}, op::Char, mode::Int64) =
    extended_source_modeling(model, srcData, recGeometry, recData, weights, perturbation, srcnum, op, mode, Options())

function extended_source_modeling(model::Model, srcData, recGeometry, recData, weights, dm, srcnum::UnitRange{Int64}, op::Char, mode::Int64, options)
# extended_source_modeling function for multiple sources. Depending on the operator and mode, this function distributes the sources
# and if applicable the input data amongst the available workers.

    p = default_worker_pool()
    results = pmap(j ->extended_source_modeling(model, subsample(srcData, j), subsample(recGeometry, j),
                                                subsample(recData, j), weights, dm, op, mode, subsample(options, j)),
                   p, srcnum)

    if op=='F' || (op=='J' && mode==1)
        argout1 = vcat(results...)
    elseif op=='J' && mode==-1
        argout1 = sum(results)
    else
        error("operation no defined")
    end
    return argout1
end
