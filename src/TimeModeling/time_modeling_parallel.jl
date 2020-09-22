
time_modeling(model::Model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum::UnitRange{Int64}, op::Char, mode::Int64) =
    time_modeling(model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum, op, mode, Options())

function time_modeling(model::Model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum::UnitRange{Int64}, op::Char, mode::Int64, options)
# time_modeling function for multiple sources. Depending on the operator and mode, this function distributes the sources
# and if applicable the input data amongst the available workers.

    p = default_worker_pool()
    time_modeling_par = remote(TimeModeling.time_modeling)
    time_modeling = retry(time_modeling_par)

    numSources = length(srcnum)
    results = Array{Any}(undef, numSources)

    # Process shots from source channel asynchronously
    @sync begin
        for j=1:numSources

            # local geometry for current position
            if srcGeometry == nothing
                srcGeometryLocal = nothing
            else
                srcGeometryLocal = subsample(srcGeometry,j)
            end
            if recGeometry == nothing
                recGeometryLocal = nothing
            else
                recGeometryLocal = subsample(recGeometry,j)
            end
            opt_local = subsample(options,j)

            # Parallelization
            if op=='F' && mode==1
                @async results[j] = time_modeling(model, srcGeometryLocal, copy(srcData[j:j]), recGeometryLocal, nothing, nothing, j, op, mode, opt_local)
            elseif op=='F' && mode==-1
                @async results[j] = time_modeling(model, srcGeometryLocal, nothing, recGeometryLocal, copy(recData[j:j]), nothing, j, op, mode, opt_local)
            elseif op=='J' && mode==1
                @async results[j] = time_modeling(model, srcGeometryLocal, copy(srcData[j:j]), recGeometryLocal, nothing, perturbation, j, op, mode, opt_local)
            elseif op=='J' && mode==-1
                @async results[j] = time_modeling(model, srcGeometryLocal, copy(srcData[j:j]), recGeometryLocal, copy(recData[j:j]), nothing, j, op, mode, opt_local)
            end
        end
    end

    if op=='F' || (op=='J' && mode==1)
        argout1 = results[1]
        for j=2:numSources
            argout1 = [argout1; results[j]]
        end
    elseif op=='J' && mode==-1
        argout1 = results[1]
        for j=2:numSources
            argout1 += results[j]
        end
    else
        error("operation no defined")
    end
    return argout1
end
