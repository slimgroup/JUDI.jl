export extended_source_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
function extended_source_modeling(model_full::Model, srcData, recGeometry, recData, weights, dm, srcnum::Int64, op::Char, mode::Int64, options)

    # Load full geometry for out-of-core geometry containers
    typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))

    # TO DO: limit model to area with sources/receivers
    model = model_full

    # Set up Python model structure
    modelPy = devito_model(model, options; dm=dm)
    if op=='J' && mode == 1
        if modelPy.dm == 0 && options.return_array == false
            return judiVector(recGeometry, zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
        elseif modelPy.dm == 0 && options.return_array == true
            return vec(zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
        end
    end

    # Load shot record if stored on disk
    if recData != nothing
        if typeof(recData) == SegyIO.SeisCon
            recData = convert(Array{Float32,2}, recData[1].data)
        elseif typeof(recData[1]) == String
            recData = load(recData)."d".data
        end
    end

    # Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
    if mode==1 && recGeometry != nothing
        recGeometry = remove_out_of_bounds_receivers(recGeometry, model)
    elseif mode==-1 && recGeometry != nothing
        recGeometry, recData = remove_out_of_bounds_receivers(recGeometry, recData, model)
    end

    isnothing(weights) ? nothing : weights = pad_array(weights[1], pad_sizes(model, options; so=0); mode=:zeros)
    # Devito interface
    argout = devito_interface(modelPy, model, srcData, recGeometry, recData, weights, dm, options)

    # Extend gradient back to original model size
    if op=='J' && mode==-1 && options.limit_m==true
        argout = extend_gradient(model_full, model, argout)
    end

    return argout
end

# Function instance without options
extended_source_modeling(model::Model, srcData, recGeometry, recData,  weights, perturbation, srcnum::Int64, op::Char, mode::Int64) =
    extended_source_modeling(model, srcData, recGeometry, recData, weights, perturbation, srcnum, op, mode, Options())
