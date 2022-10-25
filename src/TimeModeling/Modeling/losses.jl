export mse, studentst

"""
    mse(x, y)

Mean square error

    `5f0 * norm(x - y, 2)^2`

and its derivative w.r.t `x`

    `x-y`

"""
function mse(x::Matrix{T}, y::Matrix{T}) where {T<:Number}
    f = .5f0 * norm(x - y, 2)^2
    r = x - y
    return f::T, r::Matrix{T}
end

_studentst_loss(x::T, y::T, k::T) where {T<:Number} = T(1/2) * (k + 1) * log(1 + (x-y)^2 / k)

"""
studentst(x, y)

Student's T misfit 

    `.5 * (k+1) * log(1 + (x-y)^2 / k)`

and its derivative w.r.t x

    `(k + 1) * (x - y) / (k + (x - y)^2)`

"""
function studentst(x::Matrix{T}, y::Matrix{T}; k=T(2)) where {T<:Number}
    k = convert(T, k)
    f = sum(_studentst_loss.(x, y, k))
    r = (k + 1) .* (x - y) ./ (k .+ (x - y).^2)
    return f::T, r::Matrix{T}
end