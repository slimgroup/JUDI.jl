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
    if op=='J' && mode == 1
        modelPy = pm."Model"(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
            rho=process_physical_parameter(model.rho, dims), dm=process_physical_parameter(reshape(dm,model.n), dims), space_order=options.space_order)
    else
        modelPy = pm."Model"(origin=(0.,0.,0.), spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
            rho=process_physical_parameter(model.rho, dims), space_order=options.space_order)
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
    argout = devito_interface(modelPy, model.o, srcData, recGeometry, recData, weights, dm, options)

    # Extend gradient back to original model size
    if op=='J' && mode==-1 && options.limit_m==true
        argout = vec(extend_gradient(model_full, model, reshape(argout, model.n)))
    end

    return argout
end

# Function instance without options
extended_source_modeling(model::Model, srcData, recGeometry, recData,  weights, perturbation, srcnum::Int64, op::Char, mode::Int64) =
    extended_source_modeling(model, srcData, recGeometry, recData, weights, perturbation, srcnum, op, mode, Options())

######################################################################################################################################################


# d_obs = Pr*F*Pw'*w - modeling w/ extended source
function devito_interface(modelPy::PyCall.PyObject, origin, srcData::Array, recGeometry::Geometry, recData::Nothing, weights::Array, dm::Nothing, options::Options)
    ac = load_acoustic_codegen()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape, origin)

    # Devito call
    dOut = get(pycall(ac."forward_modeling", PyObject, modelPy, nothing, qIn, rec_coords,
                  space_order=options.space_order, nb=modelPy.nbpml, free_surface=options.free_surface, weight=weights[1]), 0)
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output shot record as judiVector
    if options.return_array == true
        return vec(dOut)
    else
        return judiVector(recGeometry,dOut)
    end
end

# dw = Pw*F'*Pr'*d_obs - adjoint modeling w/ extended source
function devito_interface(modelPy::PyCall.PyObject, origin, srcData::Array, recGeometry::Geometry, recData::Array, weights::Nothing, dm::Nothing, options::Options)
    ac = load_acoustic_codegen()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)
    ntSrc = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape, origin)

    # Devito call
    wOut = pycall(ac."adjoint_modeling", Array{Float32, length(modelPy.shape)}, modelPy, nothing, rec_coords, dIn,
                  space_order=options.space_order, nb=modelPy.nbpml, free_surface=options.free_surface, wavelet=qIn)
    ntSrc > ntComp && (qOut = [qOut zeros(size(qOut), ntSrc - ntComp)])

    # Output adjoint data as judiVector
    if options.return_array == true
        return vec(wOut)
    else
        return judiWeights(wOut)
    end
end

# Jacobian of extended source modeling: d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, origin, srcData::Array, recGeometry::Geometry, recData::Nothing, weights:: Array, dm::Array, options::Options)
    ac = load_acoustic_codegen()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    tmaxRec = recGeometry.t[1]
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(tmaxRec/dtComp + 1))

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    rec_coords = setup_grid(recGeometry, modelPy.shape, origin)

    # Devito call
    dOut = pycall(ac."forward_born", Array{Float32,2}, modelPy, nothing, qIn, rec_coords,
                  space_order=options.space_order, nb=modelPy.nbpml, isic=options.isic, weight=weights[1])
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output linearized shot records as judiVector
    if options.return_array == true
        return vec(dOut)
    else
        return judiVector(recGeometry,dOut)
    end
end

# Adjoint Jacobian of extended source modeling: dm = J'*d_lin
function devito_interface(modelPy::PyCall.PyObject, origin, srcData::Array, recGeometry::Geometry, recData::Array, weights:: Array, dm::Nothing, options::Options)
    ac = load_acoustic_codegen()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    if typeof(recData) == Array{Array, 1} || typeof(recData) == Array{Any, 1}
        dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    elseif typeof(recData) == Array{Array{Float32, 2}, 1}
        dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    else
        dIn = time_resample(recData[:,:,1],recGeometry,dtComp)[1]
    end

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape, origin)

    u0 = get(pycall(ac."forward_modeling", PyObject, modelPy, nothing, qIn, rec_coords, space_order=options.space_order, nb=modelPy.nbpml, save=true, 
        tsub_factor=options.subsampling_factor, weight=weights[1], return_devito_obj=true), 1)

    grad = pycall(ac."adjoint_born", Array{Float32, length(modelPy.shape)}, modelPy, rec_coords, dIn, u=u0, 
                      space_order=options.space_order, tsub_factor=options.subsampling_factor, nb=modelPy.nbpml, isic=options.isic)

    # Remove PML and return gradient as Array
    grad = remove_padding(grad, modelPy.nbpml, true_adjoint=options.sum_padding)
    return vec(grad)
end