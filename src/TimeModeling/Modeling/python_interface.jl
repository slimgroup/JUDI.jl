export devito_interface

function wrapcall_data(func, args...;kw...)
    out = pycall(func, PyArray, args...;kw...)
    # The returned array `out` is a Python Row-Major array with dimension (time, rec).
    # Unlike standard array we want to keep this ordering in julia (time first) so we need to
    # make a wrapper around the pointer, to flip the dimension the re-permute the dimensions.
    return PermutedDimsArray(unsafe_wrap(Array, out.data, reverse(size(out))), length(size(out)):-1:1)
end

wrapcall_function(func, args...;kw...) = pycall(func, PyArray, args...;kw...)

# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)
    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, srcGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = wrapcall_data(ac."forward_rec", modelPy, src_coords, qIn, rec_coords, space_order=options.space_order)
    dOut = time_resample(dOut, dtComp, recGeometry)

    # Output shot record as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry, srcData, recGeometry, dOut, options)
        return judiVector(container)
    else
        return judiVector{Float32, Array{Float32, 2}}("F*q", prod(size(dOut)), 1, 1, recGeometry, [dOut])
    end
end

# q_ad = Ps*F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    qOut = wrapcall_data(ac."adjoint_rec", modelPy, src_coords, rec_coords, dIn, space_order=options.space_order)
    qOut = time_resample(qOut, dtComp, srcGeometry)

    # Output adjoint data as judiVector
    return judiVector{Float32, Array{Float32, 2}}("F'*d", prod(size(qOut)), 1, 1, srcGeometry, [qOut])
end

# u = F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    u = wrapcall_function(ac."forward_no_rec", modelPy, src_coords, qIn, space_order=options.space_order)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(u, 1)), dtComp, u)
end

# v = F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Nothing, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    v = wrapcall_function(ac."adjoint_no_rec", modelPy, rec_coords, dIn, space_order=options.space_order)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(v, 1)), dtComp, v)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Nothing, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = wrapcall_data(ac."forward_wf_src", modelPy, srcData, rec_coords, space_order=options.space_order)
    dOut = time_resample(dOut, dtComp, recGeometry)

    return judiVector{Float32, Array{Float32, 2}}("F*u", prod(size(dOut)), 1, 1, recGeometry, [dOut])
end

# q_ad = Ps*F'*v
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    qOut = wrapcall_data(ac."adjoint_wf_src", modelPy, recData, src_coords, space_order=options.space_order)
    qOut = time_resample(qOut ,dtComp, srcGeometry)

    # Output adjoint data as judiVector
    return judiVector{Float32, Array{Float32, 2}}("F'*d", prod(size(qOut)), 1, 1, srcGeometry, [qOut])
end

# u_out = F*u_in
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Nothing, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Devito call
    u = wrapcall_function(ac."forward_wf_src_norec", modelPy, srcData, space_order=options.space_order)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(u, 1)), dtComp, u)
end

# v_out = F'*v_in
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Nothing, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")

    # Devito call
    v = wrapcall_function(ac."adjoint_wf_src_norec", modelPy, recData, space_order=options.space_order)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(v, 1)), dtComp, v)
end

# d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Nothing, dm::Union{PhysicalParameter, Array}, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = wrapcall_data(ac."born_rec", modelPy, src_coords, qIn, rec_coords,
                         space_order=options.space_order, isic=options.isic)
    dOut = time_resample(dOut, dtComp, recGeometry)

    # Output linearized shot records as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
        return judiVector(container)
    else
        return judiVector{Float32, Array{Float32, 2}}("J*dm", prod(size(dOut)), 1, 1, recGeometry, [dOut])
    end
end

# dm = J'*d_lin
function devito_interface(modelPy::PyCall.PyObject, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    grad = wrapcall_function(ac."J_adjoint", modelPy,
                  src_coords, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, isic=options.isic,
                  dft_sub=options.dft_subsampling_factor[1])

    # Remove PML and return gradient as Array
    grad = remove_padding(grad, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(grad, modelPy.spacing, modelPy.origin)
end

######################################################################################################################################################

# d_obs = Pr*F*Pw'*w - modeling w/ extended source
function devito_interface(modelPy::PyCall.PyObject, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          weights::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = wrapcall_data(ac."forward_rec_w", modelPy, weights,
                         qIn, rec_coords, space_order=options.space_order)
    dOut = time_resample(dOut, dtComp, recGeometry)
    # Output shot record as judiVector
    return judiVector{Float32, Array{Float32, 2}}("F*w", prod(size(dOut)), 1, 1, recGeometry, [dOut])
end

# dw = Pw*F'*Pr'*d_obs - adjoint modeling w/ extended source
function devito_interface(modelPy::PyCall.PyObject, srcData::Array, recGeometry::Geometry, recData::Array, weights::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    dIn = time_resample(recData, recGeometry, dtComp)[1]
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    wOut = wrapcall_function(ac."adjoint_w", modelPy, rec_coords, dIn,
                         qIn, space_order=options.space_order)

    # Output adjoint data as judiVector
    wOut = remove_padding(wOut, modelPy.padsizes; true_adjoint=false)
    if options.free_surface
        selectdim(wOut, modelPy.dim, 1) .= 0f0
    end
    return judiWeights{Float32}("Pw*F'*d",prod(size(wOut)), 1, 1, [wOut])
end

# Jacobian of extended source modeling: d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, srcData::Array, recGeometry::Geometry, recData::Nothing, weights::Array,
                          dm::Union{PhysicalParameter, Array}, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut = wrapcall_data(ac."born_rec_w", modelPy, weights, qIn, rec_coords,
                         space_order=options.space_order, isic=options.isic)
    dOut = time_resample(dOut, dtComp, recGeometry)

    # Output linearized shot records as judiVector
    if options.save_data_to_disk
        container = write_shot_record(srcGeometry,srcData,recGeometry,dOut,options)
        return judiVector(container)
    else
        return judiVector{Float32, Array{Float32, 2}}("J*dm", prod(size(dOut)), 1, 1, recGeometry, [dOut])
    end
end

# Adjoint Jacobian of extended source modeling: dm = J'*d_lin
function devito_interface(modelPy::PyCall.PyObject, srcData::Array, recGeometry::Geometry, recData::Array, weights:: Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = convert(Float32, modelPy."critical_dt")
    qIn = time_resample(srcData, recGeometry, dtComp)[1]
    dIn = time_resample(recData, recGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    grad = wrapcall_function(ac."J_adjoint", modelPy,
                  nothing, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                  space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                  freq_list=freqs, isic=options.isic, ws=weights,
                  dft_sub=options.dft_subsampling_factor[1])
    # Remove PML and return gradient as Array
    grad = remove_padding(grad, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(grad, modelPy.spacing, modelPy.origin)
end
