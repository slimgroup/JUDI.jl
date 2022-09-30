export mse, studentst

function mse(x, y)
    f = .5f0 * norm(x - y, 2)^2
    r = x - y
    return f, r
end

_studentst_loss(x::T, y::T, k::T2) where {T<:Number, T2<:Real} = T(1/2) * (k + 1) * log(1 + (x-y)^2 / k)

function studentst(x, y; k=2)
    f = sum(_studentst_loss.(x, y, k))
    r = (k + 1) .* (x - y) ./ (k .+ (x - y).^2)
    return f, r
end