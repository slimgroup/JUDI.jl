export devito_interface

##### Output type utilities
u_and_I(i::Model) = Tuple{Array{Float32,ndims(i)+1}, Array{Float32, ndims(i)}}
rec_and_I(i::Model) = Tuple{Array{Float32,2}, Array{Float32, ndims(i)}}
p_and_I(i::Model) = Tuple{Array{Float32, ndims(i)}, Array{Float32, ndims(i)}}
g_I_I(i::Model) = Tuple{Array{Float32, ndims(i)}, Array{Float32, ndims(i)}, Array{Float32, ndims(i)}}
fg_I_I(i::Model) = Tuple{Float32, Array{Float32, ndims(i)}, Array{Float32, ndims(i)}, Array{Float32, ndims(i)}}

# Output processing utilities
rec_out(data::T, geom::Geometry, ::Any, ::Any, ::Options, ::Val{false}) where T = judiVector{Float32, T}("dsyn", prod(size(data)), 1, 1, geom, [data])
function rec_out(data::T, geom::Geometry, srcData::Array, srcGeometry::Geometry, options::Options, ::Val{true}) where T
    container = write_shot_record(srcGeometry, srcData, geom, data, options)
    return judiVector(container)
end

function phys_out(p::Array{T, N}, modelPy::PyObject, model::Model, options::Options) where {T, N}
    p = remove_padding(p, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(p, model.d, model.o)
end
phys_out(x::Nothing, args...) = x
illum_out(p::Array{T, N}, name, modelPy::PyObject, model::Model, options::Options) where {T, N} = Illum(phys_out(p, modelPy, model, options), name)
illum_out(p::Tuple{Matrix, Matrix}, args...) =(illum_out(p[1], "u", args...), illum_out(p[2], "v", args...))

post_process(x::Tuple, model_full, model) = tuple((post_process(xi, model_full, model) for xi in x)...)
post_process(x, model_full, model) = x
post_process(x::PhysicalParameter, mf, m) = mf == m ? x : extend_gradient(x, mf, m)
post_process(x::Illum, mf, m) = mf == m ? x : Illum(extend_gradient(x.p, mf, m), x.name)


# d_obs = Pr*F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)
    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    qIn = time_resample(srcData, srcGeometry, dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut, Iu = pycall(ac."forward_rec", rec_and_I(model), modelPy, src_coords, qIn, rec_coords, space_order=options.space_order)
    dOut = time_resample(dOut, dtComp, recGeometry)
    dOut = rec_out(dOut, recGeometry, srcData, srcGeometry, options, Val(options.save_data_to_disk))
    return dOut, illum_out(Iu, "u", modelPy, model, options)
end

# q_ad = Ps*F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    dIn = time_resample(recData,recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    qOut, Iv = pycall(ac."adjoint_rec", rec_and_I(model), modelPy, src_coords, rec_coords, dIn, space_order=options.space_order)
    qOut = time_resample(qOut,dtComp,srcGeometry)
    qOut = rec_out(qOut, srcGeometry, nothing, nothing, options, Val(false))
    return qOut, illum_out(Iv, "v", modelPy, model, options)
end

# u = F*Ps'*q
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    u, Iu = pycall(ac."forward_no_rec", u_and_I(model), modelPy, src_coords, qIn, space_order=options.space_order)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(u, 1)), dtComp, u), illum_out(Iu, "u", modelPy, model, options)
end

# v = F'*Pr'*d_obs
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Nothing, recGeometry::Geometry, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    dIn = time_resample(recData,recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    v, Iv = pycall(ac."adjoint_no_rec", u_and_I(model), modelPy, rec_coords, dIn, space_order=options.space_order)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(v, 1)), dtComp, v), illum_out(Iv, "v", modelPy, model, options)
end

# d_obs = Pr*F*u
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Array, recGeometry::Geometry, recData::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut, Iu = pycall(ac."forward_wf_src",rec_and_I(model), modelPy, srcData, rec_coords, space_order=options.space_order)
    dOut = time_resample(dOut,dtComp,recGeometry)
    dOut = rec_out(dOut, recGeometry, nothing, nothing, options, Val(false))

    return dOut, illum_out(Iu, "u", modelPy, model, options)
end

# q_ad = Ps*F'*v
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)

    # Devito call
    qOut, Iv = pycall(ac."adjoint_wf_src", rec_and_I(model), modelPy, recData, src_coords, space_order=options.space_order)
    qOut = time_resample(qOut,dtComp,srcGeometry)
    qOut = rec_out(qOut, srcGeometry, nothing, nothing, options, Val(false))
    # Output adjoint data as judiVector
    return qOut, illum_out(Iv, "v", modelPy, model, options)
end

# u_out = F*u_in
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Array, recGeometry::Nothing, recData::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)

    # Devito call
    u, Iu = pycall(ac."forward_wf_src_norec", u_and_I(model), modelPy, srcData, space_order=options.space_order)

    # Output forward wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(u, 1)), dtComp, u), illum_out(Iu, "u", modelPy, model, options)
end

# v_out = F'*v_in
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Nothing, srcData::Nothing, recGeometry::Nothing, recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)

    # Devito call
    v, Iv = pycall(ac."adjoint_wf_src_norec", u_and_I(model), modelPy, recData, space_order=options.space_order)

    # Output adjoint wavefield as judiWavefield
    return judiWavefield(Info(prod(modelPy.shape), 1, size(v, 1)), dtComp, v), illum_out(Iv, "v", modelPy, model, options)
end

# d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Nothing, dm::Union{PhysicalParameter, Array}, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut, Iu = pycall(ac."born_rec", rec_and_I(model), modelPy, src_coords, qIn, rec_coords,
                      space_order=options.space_order, isic=options.isic)
    dOut = time_resample(dOut,dtComp,recGeometry)
    dOut = rec_out(dOut, recGeometry, srcData, srcGeometry, options, Val(options.save_data_to_disk))
    return dOut, illum_out(Iu, "u", modelPy, model, options)
end

# dm = J'*d_lin
function devito_interface(modelPy::PyCall.PyObject, model, srcGeometry::Geometry, srcData::Array, recGeometry::Geometry,
                          recData::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    qIn = time_resample(srcData,srcGeometry,dtComp)[1]
    dIn = time_resample(recData,recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    src_coords = setup_grid(srcGeometry, modelPy.shape)
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    grad, Iu, Iv = pycall(ac."J_adjoint", g_I_I(model), modelPy,
                          src_coords, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                          space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                          freq_list=freqs, isic=options.isic,
                          dft_sub=options.dft_subsampling_factor[1])

    # Remove PML and return gradient as Array
    return phys_out(grad, modelPy, model, options), illum_out((Iu, Iv), modelPy, model, options)...
end


######################################################################################################################################################


# d_obs = Pr*F*Pw'*w - modeling w/ extended source
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Nothing,
                          weights::Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    qIn = time_resample(srcData,recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut, Iu = pycall(ac."forward_rec_w", rec_and_I(model), modelPy, weights,
                      qIn, rec_coords, space_order=options.space_order)
    dOut = time_resample(dOut,dtComp,recGeometry)
    dOut = rec_out(dOut, recGeometry, nothing, nothing, options, Val(false))
    # Output shot record as judiVector
    return dOut, phys_out(Iu, "u", modelPy, model, options)
end

# dw = Pw*F'*Pr'*d_obs - adjoint modeling w/ extended source
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Array, weights::Nothing, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    dIn = time_resample(recData,recGeometry,dtComp)[1]
    qIn = time_resample(srcData,recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    wOut, Iv = pycall(ac."adjoint_w", p_and_I(model), modelPy, rec_coords, dIn,
                      qIn, space_order=options.space_order)

    # Output adjoint data as judiVector
    wOut = remove_padding(wOut, modelPy.padsizes; true_adjoint=false)
    if options.free_surface
        selectdim(wOut, modelPy.dim, 1) .= 0f0
    end
    return judiWeights{Float32}("Pw*F'*d",prod(size(wOut)), 1, 1, [wOut]), phys_out(Iv, "v", modelPy, model, options)
end

# Jacobian of extended source modeling: d_lin = J*dm
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Nothing, weights::Array,
                          dm::Union{PhysicalParameter, Array}, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    qIn = time_resample(srcData,recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    #origin = get_origin(modelPy)
    rec_coords = setup_grid(recGeometry, modelPy.shape)

    # Devito call
    dOut, Iu = pycall(ac."born_rec_w", rec_and_I(model), modelPy, weights, qIn, rec_coords,
                      space_order=options.space_order, isic=options.isic)
    dOut = time_resample(dOut,dtComp,recGeometry)
    dOut = rec_out(dOut, recGeometry, nothing, nothing, options, Val(false))
    # Output linearized shot records as judiVector
    return dOut, phys_out(Iu, "u", modelPy, model, options)
end

# Adjoint Jacobian of extended source modeling: dm = J'*d_lin
function devito_interface(modelPy::PyCall.PyObject, model, srcData::Array, recGeometry::Geometry, recData::Array, weights:: Array, dm::Nothing, options::Options)

    # Interpolate input data to computational grid
    dtComp = get_dt(model; dt=options.dt_comp)
    qIn = time_resample(srcData,recGeometry,dtComp)[1]
    dIn = time_resample(recData,recGeometry,dtComp)[1]

    # Set up coordinates with devito dimensions
    rec_coords = setup_grid(recGeometry, modelPy.shape)
    length(options.frequencies) == 0 ? freqs = nothing : freqs = options.frequencies
    grad, Iu, Iv = pycall(ac."J_adjoint", g_I_I(model), modelPy,
                          nothing, qIn, rec_coords, dIn, t_sub=options.subsampling_factor,
                          space_order=options.space_order, checkpointing=options.optimal_checkpointing,
                          freq_list=freqs, isic=options.isic, ws=weights,
                          dft_sub=options.dft_subsampling_factor[1])
    return phys_out(grad, modelPy, model, options), illum_out((Iu, Iv), modelPy, model, options)...
end
