
export time_modeling

GeomOrNot = Union{Geometry, Array, Nothing}
ArrayOrNot = Union{Array, PyArray, PyObject, Nothing}
PhysOrNot = Union{PhysicalParameter, Array, Nothing}

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to devito
function time_modeling(model_full::Model, srcGeometry::GeomOrNot, srcData::ArrayOrNot,
                       recGeometry::GeomOrNot, recData::ArrayOrNot, dm::PhysOrNot, op::Symbol, options::JUDIOptions)
    # Load full geometry for out-of-core geometry containers
    recGeometry = Geometry(recGeometry)
    srcGeometry = Geometry(srcGeometry)

    # Reutrn directly for J*0
    if (op==:born && norm(dm) == 0)
        return judiVector(recGeometry, zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
    end

    # Compute illumination ?
    illum = compute_illum(model_full, op)

    # limit model to area with sources/receivers
    if options.limit_m == true
        model = deepcopy(model_full)
        model, dm = limit_model_to_receiver_area(srcGeometry, recGeometry, model, options.buffer_size; pert=dm)
    else
        model = model_full
    end

    # Set up Python model structure
    modelPy = devito_model(model, options, dm)

    # Remove receivers outside the modeling domain (otherwise leads to segmentation faults)
    recGeometry, recData = remove_out_of_bounds_receivers(recGeometry, recData, model)

    # Devito interface
    argout = devito_interface(modelPy, srcGeometry, srcData, recGeometry, recData, dm, options, illum)
    argout = filter_none(argout)
    argout = post_process(argout, modelPy, Val(op), geom(srcGeometry, recGeometry, Val(op)), options)
    argout = save_to_disk(argout, srcGeometry, srcData, options, Val(options.save_data_to_disk))
    return argout
end

# Post processing of output of devito based on parameters
geom(srcGeometry, recGeometry, ::Val{:forward}) = recGeometry
geom(srcGeometry, recGeometry, ::Val{:born}) = recGeometry
geom(srcGeometry, recGeometry, ::Val{:adjoint}) = srcGeometry
geom(srcGeometry, recGeometry, ::Val{:adjoint_born}) = srcGeometry

post_process(t::Tuple, modelPy::PyObject, op::Val, G, o::JUDIOptions) = (post_process(t[1], modelPy, op, G, o), post_process(Base.tail(t), modelPy, Val(:adjoint_born), G, Options(;sum_padding=false))...)
post_process(t::Tuple{}, modelPy::PyObject, op::Val, G, o::JUDIOptions) = t

post_process(v::AbstractArray, modelPy::PyObject, ::Val{:forward}, G::Geometry, options::JUDIOptions) = judiVector{Float32, Matrix{Float32}}(1, G, [time_resample(v, calculate_dt(modelPy), G)])
post_process(v::AbstractArray, modelPy::PyObject, ::Val{:forward}, G, options::JUDIOptions) = judiWavefield{Float32}(1, [calculate_dt(modelPy)], [v])

post_process(v::AbstractArray, modelPy::PyObject, ::Val{:adjoint}, G::Geometry, options::JUDIOptions) = judiVector{Float32, Matrix{Float32}}(1, G, [time_resample(v, calculate_dt(modelPy), G)])
function post_process(v::AbstractArray{T, N}, modelPy::PyObject, ::Val{:adjoint}, G, options::JUDIOptions) where {T, N}
    if N == modelPy.dim
        return judiWeights{Float32}(1, [remove_padding(v, modelPy.padsizes; true_adjoint=false)])
    else
        return judiWavefield{Float32}(1, [calculate_dt(modelPy)], [v])
    end
end

function post_process(v::AbstractArray, modelPy::PyObject, ::Val{:adjoint_born}, G::Geometry, options::JUDIOptions)
    grad = remove_padding(v, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(grad, modelPy.spacing, modelPy.origin)
end

post_process(v::AbstractArray, modelPy::PyObject, ::Val{:born}, G::Geometry, options::JUDIOptions) = judiVector{Float32, Matrix{Float32}}(1, G, [time_resample(v, calculate_dt(modelPy), G)])


# Saving to disk utilities
save_to_disk(shot, args...) = shot
save_to_disk(t::Tuple, args...) = save_to_disk(t[1], args...), Base.tail(t)...
save_to_disk(shot::judiVector, srcGeometry, srcData, options, ::Val{false}) = shot

function save_to_disk(shot::judiVector, srcGeometry::GeometryIC, srcData::Array, options::JUDIOptions, ::Val{true}) 
    container = write_shot_record(srcGeometry, srcData, shot.geometry[1], shot.data[1], options)
    return judiVector(container)
end
