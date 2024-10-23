export mse, studentst

"""
    mse(x, y)

Mean square error

    `5f0 * norm(x - y, 2)^2`

and its derivative w.r.t `x`

    `x-y`

"""
function _mse(x::AT, y::AbstractArray{T}) where {T<:Number, AT<:AbstractArray{T}}
    f = .5f0 * norm(x - y, 2)^2
    r = x - y
    return f, r
end

function mse(x::PyArray{T}, y::AbstractArray{T}) where {T<:Number}
    f, r = _mse(x, y)
    return f, Py(r).to_numpy()
end

mse(x::Matrix{T}, y::Matrix{T}) where {T<:Number} = _mse(x, y)

"""
studentst(x, y)

Student's T misfit 

    `.5 * (k+1) * log(1 + (x-y)^2 / k)`

and its derivative w.r.t x

    `(k + 1) * (x - y) / (k + (x - y)^2)`

"""
function _studentst(x::AT, y::AbstractArray{T}; k=T(2)) where {T<:Number, AT<:AbstractArray{T}}
    k = convert(T, k)
    f = sum(_studentst_loss.(x, y, k))
    r = (k + 1) .* (x - y) ./ (k .+ (x - y).^2)
    return f::T, r::AT{T}
end

function studentst(x::PyArray{T}, y::AbstractArray{T}; k=T(2)) where {T<:Number}
    f, r = _studentst(x, y, k)
    return f, Py(r).to_numpy()
end

studentst(x::Matrix{T}, y::Matrix{T}; k=T(2)) where {T<:Number} = _studentst(x, y, k)

_studentst_loss(x::T, y::T, k::T) where {T<:Number} = T(1/2) * (k + 1) * log(1 + (x-y)^2 / k)