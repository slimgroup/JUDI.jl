export devito_interface

# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = pycall(ac."forward_rec", Array{Float32,2}, modelPy, src_coords, qIn, rec_coords, space_order=options.space_order)
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output shot record as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry, srcData, recGeometry, dOut, options)
        return judiVector(container)
    elseif options.return_array == true
        return vec(dOut)
    else
        return judiVector(recGeometry,dOut)
    end
end

# q_ad = Ps*F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)
    ntSrc = Int(trunc(srcGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    qOut = pycall(ac."adjoint_rec", Array{Float32,2}, modelPy, src_coords, rec_coords, dIn, space_order=options.space_order)
    ntSrc > ntComp && (qOut = [qOut zeros(size(qOut), ntSrc - ntComp)])
    qOut = time_resample(qOut,dtComp,srcGeometry)

    # Output adjoint data as judiVector
    if options.return_array == true
        return vec(qOut)
    else
        return judiVector(srcGeometry,qOut)
    end
end

# u = F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    u = pycall(ac."forward_no_rec", Array{Float32,3}, modelPy, src_coords, qIn, space_order=options.space_order)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, u)
end

# v = F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    v = pycall(ac."adjoint_no_rec", Array{Float32,3}, modelPy, rec_coords, dIn, space_order=options.space_order)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, v)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = pycall(ac."forward_wf_src", Array{Float32,2}, modelPy, srcData[1], rec_coords, space_order=options.space_order)
    #ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    if options.return_array == true
        return vec(dOut)
    else
        return judiVector(recGeometry,dOut)
    end
end

# q_ad = Ps*F'*v
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntSrc = Int(trunc(srcGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    qOut = pycall(ac."adjoint_wf_src", Array{Float32,2}, modelPy, recData[1], src_coords, space_order=options.space_order)
    qOut = time_resample(qOut,dtComp,srcGeometry)

    # Output adjoint data as judiVector
    if options.return_array == true
        return vec(qOut)
    else
        return judiVector(srcGeometry,qOut)
    end
end

# u_out = F*u_in
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntComp = size(srcData[1], 1)#.shape[1]

    # Devito call
    u = pycall(ac."forward_wf_src_norec", Array{Float32,3}, modelPy, srcData[1], space_order=options.space_order)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, u)
end

# v_out = F'*v_in
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntComp = size(recData[1], 1)

    # Devito call
    v = pycall(ac."adjoint_wf_src_norec", Array{Float32,3}, modelPy, recData[1], space_order=options.space_order)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, v)
end

# d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Nothing, dm::Union{PhysicalParameter, Array}, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    tmaxRec = recGeometry.t[1]
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(tmaxRec/dtComp + 1))

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = pycall(ac."born_rec", Array{Float32,2}, modelPy, src_coords, qIn, rec_coords,
                  space_order=options.space_order, isic=options.isic)
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output linearized shot records as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
        return judiVector(container)
    elseif options.return_array == true
        return vec(dOut)
    else
        return judiVector(recGeometry,dOut)
    end
end

# dm = J'*d_lin
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    grad = pycall(ac."J_adjoint", Array{Float32, modelPy.dim}, modelPy,
                  src_coords, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, isic=options.isic,
                  dft_sub=options.dft_subsampling_factor[1])

    # Remove PML and return gradient as Array
    grad = remove_padding(grad, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(grad, model.d, model.o)
end


######################################################################################################################################################


# d_obs = Pr*F*Pw'*w - modeling w/ extended source
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          weights::Array, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)


    # Devito call
    dOut = pycall(ac."forward_rec_w", Array{Float32,2}, modelPy, weights,
                 qIn, rec_coords, space_order=options.space_order)
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
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Array, weights::Nothing, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)
    ntSrc = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    wOut = pycall(ac."adjoint_w", Array{Float32, modelPy.dim}, modelPy, rec_coords, dIn,
                  qIn, space_order=options.space_order)

    # Output adjoint data as judiVector
    wOut = remove_padding(wOut, modelPy.padsizes; true_adjoint=false)
    if options.free_surface
        selectdim(wOut, modelPy.dim, 1) .= 0f0
    end
    if options.return_array == true
        return vec(wOut)
    else
        return judiWeights(wOut)
    end
end

# Jacobian of extended source modeling: d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Nothing, weights::Array,
                          dm::Union{PhysicalParameter, Array}, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    tmaxRec = recGeometry.t[1]
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(tmaxRec/dtComp + 1))

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = pycall(ac."born_rec_w", Array{Float32,2}, modelPy, weights, qIn, rec_coords,
                  space_order=options.space_order, isic=options.isic)
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
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Array, weights:: Array, dm::Nothing, options::Options)
    ac = load_devito_jit()

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],recGeometry,dtComp)[1]
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    grad = pycall(ac."J_adjoint", Array{Float32, modelPy.dim}, modelPy,
                  nothing, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, isic=options.isic, ws=weights,
                  dft_sub=options.dft_subsampling_factor[1])
    # Remove PML and return gradient as Array
    grad = remove_padding(grad, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(grad, model.d, model.o)
end
