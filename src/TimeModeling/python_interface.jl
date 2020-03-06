export devito_interface

# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = get(pycall(ac."forward", PyObject, modelPy, src_coords, qIn, rec_coords, space_order=options.space_order,
        nb=modelPy.nbpml, free_surface=options.free_surface), 0)
    ntRec > ntComp && (dOut = [dOut zeros(size(dOut,1), ntRec - ntComp)])
    dOut = time_resample(dOut,dtComp,recGeometry)

    # Output shot record as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
        return judiVector(container)
    elseif options.return_array == true
        return vec(dOut)
    else
        return judiVector(recGeometry,dOut)
    end
end

# q_ad = Ps*F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)
    ntSrc = Int(trunc(srcGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    qOut = pycall(ac."adjoint", Array{Float32,2}, modelPy, src_coords, rec_coords, dIn, space_order=options.space_order, nb=modelPy.nbpml, free_surface=options.free_surface)
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
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    ntComp = size(qIn,1)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    u = pycall(ac."forward_no_rec", Array{Float32,3}, modelPy, src_coords, qIn, space_order=options.space_order, free_surface=options.free_surface)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, u)
end

# v = F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]
    ntComp = size(dIn,1)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    v = pycall(ac."adjoint_no_rec", Array{Float32,3}, modelPy, rec_coords, dIn, space_order=options.space_order, nb=modelPy.nbpml, free_surface=options.free_surface)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, v)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntRec = Int(trunc(recGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = pycall(ac."forward_wf_src", PyObject, modelPy, srcData[1], rec_coords, space_order=options.space_order, nb=modelPy.nbpml,free_surface=options.free_surface)
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
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntSrc = Int(trunc(srcGeometry.t[1]/dtComp + 1))

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    qOut = pycall(ac."adjoint_wf_src", Array{Float32,2}, modelPy, src_coords, recData[1], space_order=options.space_order, nb=modelPy.nbpml, free_surface=options.free_surface)
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
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntComp = size(srcData[1], 1)#.shape[1]

    # Devito call
    u = pycall(ac."forward_wf_src_norec", Array{Float32,3}, modelPy, srcData[1], space_order=options.space_order, nb=modelPy.nbpml, free_surface=options.free_surface)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, u)
end

# v_out = F'*v_in
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    ntComp = size(recData[1], 1)

    # Devito call
    v = pycall(ac."adjoint_wf_src_norec", Array{Float32,3}, modelPy, nothing, nothing, recData[1], space_order=options.space_order, nb=modelPy.nbpml, free_surface=options.free_surface)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, ntComp), dtComp, v)
end

# d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Array, options::Options)
    ac = load_devito_jit(model)

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
    dOut = pycall(ac."born", Array{Float32,2}, modelPy, src_coords, qIn, rec_coords, space_order=options.space_order, nb=modelPy.nbpml, isic=options.isic)
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
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)
    ac = load_devito_jit(model)

    # Interpolate input data to computational grid
    dtComp = modelPy.critical_dt
    qIn = time_resample(srcData[1],srcGeometry,dtComp)[1]
    dIn = time_resample(recData[1],recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    if options.optimal_checkpointing == true
        op_F = pycall(ac."forward_modeling", PyObject, modelPy, src_coords, qIn, rec_coords, op_return=true, space_order=options.space_order, nb=modelPy.nbpml)
        grad = pycall(ac."adjoint_born", Array{Float32, length(modelPy.shape)}, modelPy, rec_coords, dIn, op_forward=op_F, space_order=options.space_order,
            nb=modelPy.nbpml, is_residual=true, isic=options.isic, n_checkpoints=options.num_checkpoints, maxmem=options.checkpoints_maxmem)
    elseif ~isempty(options.frequencies)    # gradient in frequency domain
        typeof(options.frequencies) == Array{Any,1} && (options.frequencies = options.frequencies[1])
        d_pred, uf_real, uf_imag = pycall(ac."forward_freq_modeling", PyObject, modelPy, src_coords, qIn, rec_coords, options.frequencies, space_order=options.space_order, nb=modelPy.nbpml, factor=options.dft_subsampling_factor)
        grad = pycall(ac."adjoint_freq_born", Array{Float32, length(modelPy.shape)}, modelPy, rec_coords, dIn, options.frequencies, uf_real, uf_imag, space_order=options.space_order, nb=modelPy.nbpml, isic=options.isic, factor=options.dft_subsampling_factor)
    else
        u0 = get(pycall(ac."forward_modeling", PyObject, modelPy, src_coords, qIn, rec_coords, space_order=options.space_order, nb=modelPy.nbpml, save=true, tsub_factor=options.t_sub, return_devito_obj=true), 1)
        grad = pycall(ac."adjoint_born", Array{Float32, length(modelPy.shape)}, modelPy, rec_coords, dIn, u=u0, space_order=options.space_order, tsub_factor=options.t_sub, nb=modelPy.nbpml, isic=options.isic)
    end

    # Remove PML and return gradient as Array
    grad = remove_padding(grad,modelPy.nbpml, true_adjoint=options.sum_padding)
    return vec(grad)
end
