module ZygoteJUDIExt

isdefined(Base, :get_extension) ? (using JUDI) : (using ..JUDI)
using Zygote

Zygote.unbroadcast(x::AbstractArray, x̄::LazyPropagation) = Zygote.unbroadcast(x, eval_prop(x̄))

function Zygote.accum(x::judiVector{T, AT}, y::DenseArray) where {T, AT}
    newd = [Zygote.accum(x.data[i], y[:, :, i, 1]) for i=1:x.nsrc]
    return judiVector{T, AT}(x.nsrc, x.geometry, newd)
end

end