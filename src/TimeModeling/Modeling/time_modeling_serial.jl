
export time_modeling

GeomOrNot = Union{Geometry, Array, Nothing}
ArrayOrNot = Union{Array, PyArray, PyObject, Nothing}
PhysOrNot = Union{PhysicalParameter, Array, Nothing}

# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to devito
function time_modeling(model_full::AbstractModel, srcGeometry::GeomOrNot, srcData::ArrayOrNot,
                       recGeometry::GeomOrNot, recData::ArrayOrNot, dm::PhysOrNot,
                       op::Symbol, options::JUDIOptions, fw::Bool, illum::Bool)
    GC.gc(true)
    devito.clear_cache()

    # Load full geometry for out-of-core geometry containers
    recGeometry = Geometry(recGeometry)
    srcGeometry = Geometry(srcGeometry)

    # Reutrn directly for J*0
    if (op==:born && norm(dm) == 0)
        return judiVector(recGeometry, zeros(Float32, recGeometry.nt[1], length(recGeometry.xloc[1])))
    end

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
    argout = post_process(argout, modelPy, Val(op), recGeometry, srcGeometry, options)
    argout = save_to_disk(argout, srcGeometry, srcData, options, Val(fw), Val(options.save_data_to_disk))
    return argout
end

# Backward compat for external packages
post_process(t::Tuple, modelPy::PyObject, op::Val, Gr, o::JUDIOptions) = post_process(t, modelPy, op, Gr, nothing, o)

# Post processing of output of devito based on parameters
post_process(t::Tuple, modelPy::PyObject, op::Val, Gr, Gs, o::JUDIOptions) = (post_process(t[1], modelPy, op, Gr, Gs, o), post_process(Base.tail(t), modelPy, Val(:adjoint_born), Gr, Gs, Options(;sum_padding=false))...)
post_process(t::Tuple{}, ::PyObject, ::Val, ::Any, ::Any, ::JUDIOptions) = t

function post_process(v::AbstractArray{T, N}, modelPy::PyObject, ::Val{:adjoint}, ::Any, ::Any, options::JUDIOptions) where {T, N}
    if N == modelPy.dim
        return judiWeights{T}(1, [remove_padding(v, modelPy.padsizes; true_adjoint=false)])
    else
        return judiWavefield{T}(1, [calculate_dt(modelPy)], [v])
    end
end

post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:forward}, ::Any, ::Any, options::JUDIOptions) where {T<:Number} = judiWavefield{T}(1, [calculate_dt(modelPy)], [v])

function post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:adjoint_born}, Gr::Geometry{T}, ::Any, options::JUDIOptions) where {T<:Number}
    grad = remove_padding(v, modelPy.padsizes; true_adjoint=options.sum_padding)
    return PhysicalParameter(grad, modelPy.spacing, modelPy.origin)
end

post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:forward}, G::Geometry{T}, Gs, options::JUDIOptions) where {T<:Number} = post_process_src(v, calculate_dt(modelPy), G, Gs)
post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:adjoint}, G::Geometry{T}, Gs, options::JUDIOptions) where {T<:Number} = post_process_src(v, calculate_dt(modelPy), G, Gs)
post_process(v::AbstractArray{T}, modelPy::PyObject, ::Val{:born}, G::Geometry{T}, Gs, options::JUDIOptions) where {T<:Number} = post_process_src(v, calculate_dt(modelPy), G, Gs)

post_process_src(v::AbstractArray{T}, dt::T, Gr::Geometry, Gs::Geometry) where {T<:Number} = judiVector{T, Matrix{T}}(1, Gr, [time_resample(v, dt, Gs, Gr)])
post_process_src(v::AbstractArray{T}, dt::T, Gr::Geometry, ::Any) where {T<:Number} = judiVector{T, Matrix{T}}(1, Gr, [time_resample(v, dt, Gr)])

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
