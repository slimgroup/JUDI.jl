export devito_interface

function wrapcall_data(func, args...;kw...)
    IT = kw[:illum] ? PyArray : PyObject
    out = pylock() do
        pycall(func, Tuple{PyArray, IT}, args...;kw...)
    end
    # The returned array `out` is a Python Row-Major array with dimension (time, rec).
    # Unlike standard array we want to keep this ordering in julia (time first) so we need to
    # make a wrapper around the pointer, to flip the dimension the re-permute the dimensions.
    return PermutedDimsArray(unsafe_wrap(Array, out[1].data, reverse(size(out[1]))), length(size(out[1])):-1:1), out[2]
end

function wrapcall_weights(func, args...;kw...)
    IT = kw[:illum] ? PyArray : PyObject
    out = pylock() do 
        pycall(func, Tuple{PyArray, IT}, args...;kw...)
    end
    return out
end

function wrapcall_wf(func, args...;kw...)
    IT = kw[:illum] ? PyArray : PyObject
    out = pylock() do
        pycall(func, Tuple{Array{Float32}, IT}, args...;kw...)
    end
    return out
end

function wrapcall_grad(func, args...;kw...)
    IT = kw[:illum] ? (PyArray, PyArray) : (PyObject, PyObject)
    out = pylock() do 
        pycall(func, Tuple{PyArray, IT...}, args...;kw...)
    end
    return out
end

# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("Pr*F*Ps'*q")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, srcGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."forward_rec", modelPy, src_coords, qIn, rec_coords, space_order=options.space_order, f0=options.f0, illum=illum)
end

# q_ad = Ps*F'*Pr'*d_obs
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("Ps*F'*Pr'*d_obs")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."adjoint_rec", modelPy, src_coords, rec_coords, dIn, space_order=options.space_order, f0=options.f0, illum=illum)
end

# u = F*Ps'*q
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("F*Ps'*q")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    return wrapcall_wf(ac."forward_no_rec", modelPy, src_coords, qIn, space_order=options.space_order, illum=illum)
end

# v = F'*Pr'*d_obs
function devito_interface(modelPy::PyObject, srcGeometry::Nothing, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("F'*Pr'*d_obs")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_wf(ac."adjoint_no_rec", modelPy, rec_coords, dIn, space_order=options.space_order, illum=illum)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::PyObject, srcGeometry::Nothing, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("Pr*F*u")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."forward_wf_src", modelPy, srcData, rec_coords, space_order=options.space_order, f0=options.f0, illum=illum)
end

# q_ad = Ps*F'*v
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("Ps*F'*v")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."adjoint_wf_src", modelPy, recData, src_coords, space_order=options.space_order, f0=options.f0, illum=illum)
end

# u_out = F*u_in
function devito_interface(modelPy::PyObject, srcGeometry::Nothing, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("F*u_in")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Devito call
    return wrapcall_wf(ac."forward_wf_src_norec", modelPy, srcData, space_order=options.space_order, illum=illum)
end

# v_out = F'*v_in
function devito_interface(modelPy::PyObject, srcGeometry::Nothing, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("F'*v_in")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Devito call
    return wrapcall_wf(ac."adjoint_wf_src_norec", modelPy, recData, space_order=options.space_order, illum=illum)
end

# d_lin = J*dm
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Nothing, dm::Union{PhysicalParameter, Array}, options::JUDIOptions, illum::Bool)
    judilog("J*dm")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."born_rec", modelPy, src_coords, qIn, rec_coords,
                  space_order=options.space_order, ic=options.IC, f0=options.f0, illum=illum)
end

# dm = J'*d_lin
function devito_interface(modelPy::PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("J'*d_lin")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    return wrapcall_grad(ac."J_adjoint", modelPy,
                  src_coords, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, ic=options.IC, is_residual=true,
                  dft_sub=options.dft_subsampling_factor[1], f0=options.f0, illum=illum)
end

######################################################################################################################################################

# d_obs = Pr*F*Pw'*w - modeling w/ extended source
function devito_interface(modelPy::PyObject, weights::Array, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("Pr*F*Pw'*w")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."forward_rec_w", modelPy, weights,
                 qIn, rec_coords, space_order=options.space_order, f0=options.f0, illum=illum)
end

# dw = Pw*F'*Pr'*d_obs - adjoint modeling w/ extended source
function devito_interface(modelPy::PyObject, srcData::Array, ::Nothing, recGeometry::Geometry, recData::Array, ::Nothing, options::JUDIOptions, illum::Bool)
    judilog("Pw*F'*Pr'*d_obs")
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    dIn = time_resample(recData, recGeometry, dtComp)[1]
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_weights(ac."adjoint_w", modelPy, rec_coords, dIn,
                             qIn, space_order=options.space_order, f0=options.f0, illum=illum)
end

# Jacobian of extended source modeling: d_lin = J*dm
function devito_interface(modelPy::PyObject, weights::Array, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          dm::Union{PhysicalParameter, Array}, options::JUDIOptions, illum::Bool)
    judilog("Jw*dm")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    return wrapcall_data(ac."born_rec_w", modelPy, weights, qIn, rec_coords,
                  space_order=options.space_order, ic=options.IC, f0=options.f0, illum=illum)
end

# Adjoint Jacobian of extended source modeling: dm = J'*d_lin
function devito_interface(modelPy::PyObject, weights::Array, srcData::Array, recGeometry::Geometry, recData::Array, dm::Nothing, options::JUDIOptions, illum::Bool)
    judilog("Jw'*d_lin")
    weights = pad_array(reshape(weights, modelPy.shape), modelPy.padsizes; mode=:zeros)
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)[1]
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    return wrapcall_grad(ac."J_adjoint", modelPy,
                  nothing, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, ic=options.IC, ws=weights, is_residual=true,
                  dft_sub=options.dft_subsampling_factor[1], f0=options.f0, illum=illum)
end
