
export time_modeling

GeomOrNot = Union{Geometry, Array, Nothing}
ArrayOrNot = Union{Array, PyArray, PyObject, Nothing}
PhysOrNot = Union{PhysicalParameter, Array, Nothing}

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to devito
function time_modeling(model_full::AbstractModel, srcGeometry::GeomOrNot, srcData::ArrayOrNot,
                       recGeometry::GeomOrNot, recData::ArrayOrNot, dm::PhysOrNot,
                       op::Symbol, options::JUDIOptions, fw::Bool)
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
        @juditime "Limit model to geometry" begin
            model = deepcopy(model_full)
            model, dm = limit_model_to_receiver_area(srcGeometry, recGeometry, model, options.buffer_size; pert=dm)
        end
    else
        model = model_full
    end

    # Set up Python model structure
    @juditime "Devito Model" begin
        modelPy = devito_model(model, options, dm)
    end

    # Devito interface
    @juditime "Propagation" begin
        argout = devito_interface(modelPy, srcGeometry, srcData, recGeometry, recData, dm, options, illum, fw)
    end

    @juditime "Filter empty output" begin
        argout = filter_none(argout)
    end
    argout = post_process(argout, modelPy, Val(op), recGeometry, options)
    argout = save_to_disk(argout, srcGeometry, srcData, options, Val(fw), Val(options.save_data_to_disk))
    return argout
end

# Post processing of output of devito based on parameters
post_process(t::Tuple, modelPy::PyObject, op::Val, G, o::JUDIOptions) = (post_process(t[1], modelPy, op, G, o), post_process(Base.tail(t), modelPy, Val(:adjoint_born), G, Options(;sum_padding=false))...)
post_process(t::Tuple{}, ::PyObject, ::Val, ::Any, ::JUDIOptions) = t

post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:forward}, G::Geometry{T}, options::JUDIOptions) where {T<:Number} = judiVector{T, Matrix{T}}(1, G, [time_resample(v, calculate_dt(modelPy), G)])
post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:forward}, ::Any, options::JUDIOptions) where {T<:Number} = judiWavefield{T}(1, [calculate_dt(modelPy)], [v])
post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:adjoint}, G::Geometry{T}, options::JUDIOptions) where {T<:Number} = judiVector{T, Matrix{T}}(1, G, [time_resample(v, calculate_dt(modelPy), G)])

function post_process(v::AbstractArray{T, N}, modelPy::PyObject, ::Val{:adjoint}, ::Any, options::JUDIOptions) where {T, N}
    if N == modelPy.dim
        return judiWeights{T}(1, [remove_padding(v, modelPy.padsizes; true_adjoint=false)])
    else
        return judiWavefield{T}(1, [calculate_dt(modelPy)], [v])
    end
end

function post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:adjoint_born}, G::Geometry{T}, options::JUDIOptions) where {T<:Number}
    grad = remove_padding(v, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(grad, modelPy.spacing, modelPy.origin)
end

post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:born}, G::Geometry{T}, options::JUDIOptions) where {T<:Number} = judiVector{T, Matrix{T}}(1, G, [time_resample(v, calculate_dt(modelPy), G)])

# Saving to disk utilities
save_to_disk(shot, args...) = shot
save_to_disk(t::Tuple, args...) = save_to_disk(t[1], args...), Base.tail(t)...
save_to_disk(shot::judiVector{T, Matrix{T}}, ::Any, ::Any, ::Any, ::Any, ::Val{false}) where {T<:Number} = shot

function save_to_disk(shot::judiVector{T}, srcGeometry::GeometryIC{T}, srcData::Array, options::JUDIOptions,
                      ::Val{true}, ::Val{true}) where {T<:Number}
    @juditime "Dump data to segy" begin
        container = write_shot_record(srcGeometry, srcData, shot.geometry[1], shot.data[1], options)
        dout = judiVector(container)
    end
    return dout
end
