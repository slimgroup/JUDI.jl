export time_resample


"""
    time_resample(data, geometry_in, dt_new)

Resample the input data with sinc interpolation from the current time sampling (geometrty_in) to the
new time sampling `dt_new`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `geometry_in`: Geometry on which `data` is defined.
* `dt_new`: New time sampling rate to interpolate onto.
"""
function time_resample(data::AbstractArray{T, N}, G_in::Geometry, dt_new::Real) where {T<:Real, N}
    tend = step(G_in.taxis[1])*(size(data, 1) - 1) + first(G_in.taxis[1])
    new_t = first(G_in.taxis[1]):dt_new:tend
    return time_resample(data, G_in.taxis[1], new_t)
end

"""
    time_resample(data, dt_in, dt_new)

Resample the input data with sinc interpolation from the current time sampling dt_in to the 
new time sampling `dt_new`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `dt_in`: Time sampling of input
* `dt_new`: New time sampling rate to interpolate onto.
"""
function time_resample(data::AbstractArray{T, N}, t_in::StepRangeLen, t_new::StepRangeLen) where {T<:Real, N}
    dt_in, dt_new = step(t_in), step(t_new)
    if dt_new==dt_in
        return data
    elseif (dt_new % dt_in) == 0
        rate = Int64(div(dt_new, dt_in))
        return _time_resample(data, rate)
    else
        @juditime "Data time sinc-interpolation" begin
            dataInterp = Float32.(SincInterpolation(data, t_in, t_new))
        end
        return dataInterp
    end
end

time_resample(data::AbstractArray{T, N}, dt_in::Number, dt_new::Number, t::Number) where {T<:Real, N} =
    time_resample(data, 0:dt_in:(dt_in*ceil(t/dt_in)), 0:dt_new:(dt_new*ceil(t/dt_new)))


"""
    time_resample(data, dt_in, geometry_in)

Resample the input data with sinc interpolation from the current time sampling (dt_in) to the
new time sampling `geometry_out`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `geometry_out`: Geometry on which `data` is to be interpolated.
* `dt_in`: Time sampling rate of the `data.`
"""
function time_resample(data::AbstractArray{T, N}, dt_in::Real, G_out::Geometry{T}) where {T<:Real, N}
    currt = range(0f0, step=dt_in, length=size(data, 1))
    return time_resample(data, currt, G_out.taxis[1])
end

function time_resample(data::AbstractArray{T, N}, dt_in::Real, G_in::Geometry{T}, G_out::Geometry{T}) where {T<:Real, N}
    t0 = min(get_t0(G_in, 1), get_t0(G_out, 1))
    currt = range(t0, step=dt_in, length=size(data, 1))
    return time_resample(data, currt, G_out.taxis[1])
end

_time_resample(data::Matrix{T}, rate::Integer) where T = data[1:rate:end, :]
_time_resample(data::PermutedDimsArray{T, 2, (2, 1), (2, 1), Matrix{T}}, rate::Integer) where {T<:Real} = data.parent[:, 1:rate:end]'

SincInterpolation(Y::Matrix{T}, S::StepRangeLen{T}, Up::StepRangeLen{T}) where T<:Real = sinc.( (Up .- S') ./ (S[2] - S[1]) ) * Y
SincInterpolation(Y::PermutedDimsArray{T, 2, (2, 1), (2, 1), Matrix{T}}, S::StepRangeLen{T}, Up::StepRangeLen{T}) where T<:Real = (Y.parent * sinc.( (Up' .- S) ./ (S[2] - S[1]) ))'


"""
    _maybe_pad_t0(q, qGeom, data, dataGeom)

Pad zeros for data with non-zero t0, usually from a segy file so that time axis and array size match for the source and data.
"""
function _maybe_pad_t0(qIn::Matrix{T}, qGeom::Geometry, dObserved::Matrix{T}, dataGeom::Geometry) where T<:Number
    dt0 = get_t0(dataGeom, 1) - get_t0(qGeom, 1)
    Dt = get_t(dataGeom, 1) - get_t(qGeom, 1)
    dsize = size(qIn, 1) - size(dObserved, 1)
    # Same times, do nothing
    if dsize == 0 && dt0 == 0 && Dt == 0
        return qIn, dObserved
    # First case, same size, then it's a shift
    elseif dsize == 0 && dt0 != 0 && Dt != 0
        # Shift means both t0 and t same sign difference
        @assert sign(dt0) == sign(Dt)
        pad_size = Int(div(get_t0(dataGeom, 1), get_dt(dataGeom, 1)))
        if dt0 > 0
            # Data has larger t0, pad data left and q right
            dObserved = vcat(zeros(T, pad_size, size(dObserved, 2)), dObserved)
            qIn = vcat(qIn, zeros(T, pad_size, size(qIn, 2)))
        else
            # q has larger t0, pad data right and q left
            dObserved = vcat(dObserved, zeros(T, pad_size, size(dObserved, 2)))
            qIn = vcat(zeros(T, pad_size, size(qIn, 2)), qIn)
        end
    elseif dsize !=0
        # We might still have differnt t0 and t
        # Pad so that we go from smaller dt to largest t
        ts = min(get_t0(qGeom, 1), get_t0(dataGeom, 1))
        te = max(get_t(qGeom, 1), get_t(dataGeom, 1))

        pdatal = Int(div(get_t0(dataGeom, 1) - ts, get_dt(dataGeom, 1)))
        pdatar = Int(div(te - get_t(dataGeom, 1), get_dt(dataGeom, 1)))
        dObserved = vcat(zeros(T, pdatal, size(dObserved, 2)), dObserved, zeros(T, pdatar, size(dObserved, 2)))

        pql = Int(div(get_t0(qGeom, 1) - ts, get_dt(qGeom, 1)))
        pqr = Int(div(te - get_t(qGeom, 1), get_dt(qGeom, 1)))
        qIn = vcat(zeros(T, pql, size(qIn, 2)), qIn, zeros(T, pqr, size(qIn, 2)))
    else
        throw(judiMultiSourceException("""
            Data and source have different
            t0 : $((get_t0(dataGeom, 1), get_t0(qGeom, 1)))
            and t: $((get_t(dataGeom, 1), get_t(qGeom, 1)))
            and are not compatible in size for padding: $((size(qIn, 1), size(dObserved, 1)))"""))
    end
    return qIn, dObserved
end

pad_msg = """
    This is an internal method for single source propatation,
    only single-source judiVectors are supported
"""

function _maybe_pad_t0(qIn::judiVector{T, Matrix{T}}, dObserved::judiVector{T, Matrix{T}}) where{T<:Number}
    @assert qIn.nsrc == 1 || throw(judiMultiSourceException(pad_msg))
    return _maybe_pad_t0(qIn.data[1], qIn.geometry[1], dObserved.data[1], dObserved.geometry[1])
end

_maybe_pad_t0(qIn::judiVector{T, AT}, dObserved::judiVector{T, AT}) where{T<:Number, AT} =
    _maybe_pad_t0(get_data(qIn), get_data(dObserved))
