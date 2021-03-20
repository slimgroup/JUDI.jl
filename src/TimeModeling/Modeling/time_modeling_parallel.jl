
time_modeling(model::Model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum::UnitRange{Int64}, op::Char, mode::Int64) =
    time_modeling(model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum, op, mode, Options())

function time_modeling(model::Model, srcGeometry, srcData, recGeometry, recData, dm, srcnum::UnitRange{Int64}, op::Char, mode::Int64, options)
# time_modeling function for multiple sources. Depending on the operator and mode, this function distributes the sources
# and if applicable the input data amongst the available workers.

    p = default_worker_pool()
    results = pmap(j -> time_modeling(model, subsample(srcGeometry,j), subsample(srcData, j),
                                      subsample(recGeometry,j), subsample(recData, j), dm,
                                      op, mode, subsample(options, j)),
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
