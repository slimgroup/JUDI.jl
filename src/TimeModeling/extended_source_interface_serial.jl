export extended_source_modeling

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito
function extended_source_modeling(model_full::Model, srcData, recGeometry, recData, weights, dm, srcnum::Int64, op::Char, mode::Int64, options)
    pm = load_pymodel()

    # Load full geometry for out-of-core geometry containers
    typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))
    length(model_full.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)

    # TO DO: limit model to area with sources/receivers
    model = model_full

    # Set up Python model structure
    modelPy = devito_model(model, options)
    update_m(modelPy, model.m, dims)

    if op=='J' && mode == 1
        update_dm(modelPy, reshape(dm,model.n), dims)
    end
    # Load shot record if stored on disk
    if recData != nothing
        if typeof(recData[1]) == SegyIO.SeisCon
            recDataCell = Array{Any}(undef, 1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
        elseif typeof(recData[1]) == String
            recData = load(recData[1])."d".data
        end
    end

    # To DO: Remove receivers outside the modeling domain

    # Devito interface
    argout = devito_interface(modelPy, model, srcData, recGeometry, recData, weights, dm, options)

    # Extend gradient back to original model size
    if op=='J' && mode==-1 && options.limit_m==true
        argout = vec(extend_gradient(model_full, model, reshape(argout, model.n)))
    end

    return argout
end

# Function instance without options
extended_source_modeling(model::Model, srcData, recGeometry, recData,  weights, perturbation, srcnum::Int64, op::Char, mode::Int64) =
    extended_source_modeling(model, srcData, recGeometry, recData, weights, perturbation, srcnum, op, mode, Options())
