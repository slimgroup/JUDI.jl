extended_source_modeling(model::Model, srcData, recGeometry, recData, weights, perturbation, srcnum::UnitRange{Int64}, op::Char, mode::Int64) =
    extended_source_modeling(model, srcData, recGeometry, recData, weights, perturbation, srcnum, op, mode, Options())

function extended_source_modeling(model::Model, srcData, recGeometry, recData, weights, dm, srcnum::UnitRange{Int64}, op::Char, mode::Int64, options)
# extended_source_modeling function for multiple sources. Depending on the operator and mode, this function distributes the sources
# and if applicable the input data amongst the available workers.
    red_op! = (op=='F' || (op=='J' && mode==1)) ? vcat! : sum!
    results = judipmap(j ->extended_source_modeling(model, subsample(srcData, j), subsample(recGeometry, j),
                                                    subsample(recData, j), weights, dm, op, mode, subsample(options, j)), srcnum, red_op!)

    return results
end
