export devito_interface

########################################### Logging utility ########################################################
_op_str(fw::Bool) = fw ? "F" : "F'"

########################################## Python pycall wrappers with lock #########################################


function wrapcall_data(func, args...;kw...)
    argout = func(args...;kw...)
    res = []
    for a in argout
        if Bool(PythonCall.pybuiltins.isinstance(a, np.ndarray))
            push!(res, PyArray(a))
        elseif Bool(a == Py(PythonCall.pybuiltins.None))
            continue
        else
            push!(res, pyconvert(Float32, a))
        end
    end

    if length(res) == 1
        return res[1]
    else
        return tuple(res...)
    end
end

# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::Py, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Pr*$(_op_str(fw))*Ps'*q")
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    qIn = time_resample(srcData, srcGeometry, dtComp)
    qIn = _maybe_pad_t0(qIn, srcGeometry, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac.forward_rec, modelPy, src_coords, qIn, rec_coords, fw=fw, f0=options.f0, illum=illum)
end

# u = F*Ps'*q
function devito_interface(modelPy::Py, srcGeometry::Geometry, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("$(_op_str(fw))*Ps'*q")
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    qIn = time_resample(srcData, srcGeometry, dtComp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac.forward_no_rec, modelPy, src_coords, qIn, fw=fw, illum=illum)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::Py, srcGeometry::Nothing, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Pr*$(_op_str(fw))*u")
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    srcData = Py(srcData).to_numpy()
    return wrapcall_data(ac.forward_wf_src, modelPy, srcData, rec_coords, fw=fw, f0=options.f0, illum=illum)
end

# u_out = F*u_in
function devito_interface(modelPy::Py, srcGeometry::Nothing, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("$(_op_str(fw))*u_in")
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)

    # Devito call
    srcData = Py(srcData).to_numpy()
    return wrapcall_data(ac.forward_wf_src_norec, modelPy, srcData, fw=fw, illum=illum)
end

# d_lin = J*dm
function devito_interface(modelPy::Py, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Nothing, dm::Union{PhysicalParameter, Array}, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("J($(_op_str(fw)), q)*dm")
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    qIn = time_resample(srcData, srcGeometry, dtComp)
    qIn = _maybe_pad_t0(qIn, srcGeometry, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac.born_rec, modelPy, src_coords, qIn, rec_coords, fw=fw,
                         ic=options.IC, f0=options.f0, illum=illum)
end

# dm = J'*d_lin
function devito_interface(modelPy::Py, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("J($(_op_str(fw)), q)'*d_lin")
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    qIn = time_resample(srcData, srcGeometry, dtComp)
    dIn = time_resample(recData, recGeometry, dtComp)
    qIn, dIn = _maybe_pad_t0(qIn, srcGeometry, dIn, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies

    # Make dIn numpy to avoid indexing issues
    dIn = Py(dIn).to_numpy()

    return wrapcall_data(ac.J_adjoint, modelPy,
                  src_coords, qIn, rec_coords, dIn, fw=fw, t_sub=options.subsampling_factor,
                  checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, ic=options.IC, is_residual=true,
                  dft_sub=options.dft_subsampling_factor[1], f0=options.f0, illum=illum)
end

######################################################################################################################################################

# d_obs = Pr*F*Pw'*w - modeling w/ extended source
function devito_interface(modelPy::Py, weights::Array, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Pr*$(_op_str(fw))*Pw'*w")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    qIn = time_resample(srcData, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    weights = Py(weights).to_numpy()
    return wrapcall_data(ac.forward_rec_w, modelPy, weights, qIn, rec_coords,
                         fw=fw, f0=options.f0, illum=illum)
end

# dw = Pw*F'*Pr'*d_obs - adjoint modeling w/ extended source
function devito_interface(modelPy::Py, recGeometry::Geometry, recData::Array, srcData::Array, ::Nothing, ::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Pw*$(_op_str(fw))*Pr'*d_obs")
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    dIn = time_resample(recData, recGeometry, dtComp)
    qIn = time_resample(srcData, recGeometry, dtComp)
    qIn, dIn = _maybe_pad_t0(qIn, recGeometry, dIn, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac.adjoint_w, modelPy, rec_coords, dIn, qIn,
                          fw=fw, f0=options.f0, illum=illum)
end

# Jacobian of extended source modeling: d_lin = J*dm
function devito_interface(modelPy::Py, weights::Array, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          dm::Union{PhysicalParameter, Array}, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Jw($(_op_str(fw)), q)*dm")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    qIn = time_resample(srcData, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac.born_rec_w, modelPy, weights, qIn, rec_coords,
                         fw=fw, ic=options.IC, f0=options.f0, illum=illum)
end

# Adjoint Jacobian of extended source modeling: dm = J'*d_lin
function devito_interface(modelPy::Py, weights::Array, srcData::Array, recGeometry::Geometry, recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool, fw::Bool)
    judilog("Jw($(_op_str(fw)), q)'*d_lin")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = pyconvert(Float32, modelPy.critical_dt)
    qIn = time_resample(srcData, recGeometry, dtComp)
    dIn = time_resample(recData, recGeometry, dtComp)
    qIn, dIn = _maybe_pad_t0(qIn, recGeometry, dIn, recGeometry, dtComp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies

    # Make dIn numpy to avoid indexing issues
    dIn = Py(dIn).to_numpy()

    return wrapcall_data(ac.J_adjoint, modelPy,
                  nothing, qIn, rec_coords, dIn, fw=fw, t_sub=options.subsampling_factor,
                  checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, ic=options.IC, ws=weights, is_residual=true,
                  dft_sub=options.dft_subsampling_factor[1], f0=options.f0, illum=illum)
end
