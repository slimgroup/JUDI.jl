export extended_source_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
function extended_source_modeling(model_full::Model, srcData, recGeometry, recData, weights, dm, op::Char, mode::Int64, options)

    # Load full geometry for out-of-core geometry containers
    recGeometry = Geometry(recGeometry)

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
    typeof(recData) == SegyIO.SeisCon && (recData = convert(Array{Float32,2}, recData[1].data))

    # Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
    recGeometry, recData = remove_out_of_bounds_receivers(recGeometry, recData, model)

    isnothing(weights) ? nothing : weights = pad_array(weights, pad_sizes(model, options; so=0); mode=:zeros)
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
