export initial_value_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
function initial_value_modeling(model_full::Model, recGeometry, recData, weights, dm, srcnum::Int64, op::Char, mode::Int64, options)

    # Load full geometry for out-of-core geometry containers
    typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))

    # TO DO: limit model to area with sources/receivers
    model = model_full

    # Set up Python model structure
    modelPy = devito_model(model, options)

    # Load shot record if stored on disk
    if recData != nothing
        if typeof(recData[1]) == SegyIO.SeisCon
            recDataCell = Array{Any}(undef, 1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
        elseif typeof(recData[1]) == String
            recData = load(recData[1])."d".data
        end
    end

    # Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
    if mode==1 && recGeometry != nothing
        recGeometry = remove_out_of_bounds_receivers(recGeometry, model)
    elseif mode==-1 && recGeometry != nothing
        recGeometry, recData = remove_out_of_bounds_receivers(recGeometry, recData[1], model)
    end

    # Pad both values with PML 
    isnothing(weights) ? nothing : weights[1] = pad_array(weights[1], pad_sizes(model, options; so=0); mode=:zeros)
    isnothing(weights) ? nothing : weights[2] = pad_array(weights[2], pad_sizes(model, options; so=0); mode=:zeros)
    
    # Devito interface
    argout = devito_interface(modelPy, model, recGeometry, recData, weights, dm, options)

    return argout
end

# Function instance without options
initial_value_modeling(model::Model, srcData, recGeometry, recData,  weights, perturbation, srcnum::Int64, op::Char, mode::Int64) =
    initial_value_modeling(model, srcData, recGeometry, recData, weights, perturbation, srcnum, op, mode, Options())
