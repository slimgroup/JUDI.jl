
time_modeling(model::Model, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, perturbation, srcnum::UnitRange{Int64}, op::Char, mode::Int64) = 
    time_modeling(model, srcGeometry, srcData, recGeometry, recData, perturbation, srcnum, op, mode, Options())

function time_modeling(model::Model, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, perturbation, srcnum::UnitRange{Int64}, op::Char, mode::Int64, options)
# time_modeling function for multiple sources. Depending on the operator and mode, this function distributes the sources
# and if applicable the input data amongst the available workers.

    p = default_worker_pool()
    time_modeling = remote(TimeModeling.time_modeling)
    # time_modeling = wrap_retry(time_modeling, options.retry_n)

    numSources = length(srcnum)
    results = Array{Any}(numSources)
    
    # Process shots from source channel asynchronously
    @sync begin
        for j=1:numSources
            
            # local geometry for current position
            srcGeometryLocal = subsample(srcGeometry,j)
            recGeometryLocal = subsample(recGeometry,j)

            # Parallelization
            if op=='F' && mode==1
                srcDataLocal = Array{Any}(1)
                srcDataLocal[1] = srcData[j]
                @async results[j] = time_modeling(model, srcGeometryLocal, srcDataLocal, recGeometryLocal, [], [], j, op, mode, options)
            elseif op=='F' && mode==-1
                recDataLocal = Array{Any}(1)
                recDataLocal[1] = recData[j]
                @async results[j] = time_modeling(model, srcGeometryLocal, [], recGeometryLocal, recDataLocal, [], j, op, mode, options)
            elseif op=='J' && mode==1
                srcDataLocal = Array{Any}(1)
                srcDataLocal[1] = srcData[j]
                @async results[j] = time_modeling(model, srcGeometryLocal, srcDataLocal, recGeometryLocal, [], perturbation, j, op, mode, options)  
            elseif op=='J' && mode==-1
                srcDataLocal = Array{Any}(1)
                srcDataLocal[1] = srcData[j]
                recDataLocal = Array{Any}(1)
                recDataLocal[1] = recData[j]
                @async results[j] = time_modeling(model, srcGeometryLocal, srcDataLocal, recGeometryLocal, recDataLocal, [], j, op, mode, options)  
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


