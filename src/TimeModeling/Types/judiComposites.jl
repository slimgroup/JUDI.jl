"""
Vector of a judiVector and a judiWeight
"""
mutable struct judiVStack{vDT<:Number}
    m::Integer
    n::Integer
    components::Array{Any, 1}
end

concrete_msv = [judiVector, judiWeights, judiWavefield]

for c in concrete_msv
    for c2 in concrete_msv
        if c2 != c
            @eval function vcat(x::$(c){T}, y::$(c2){T}) where {T<:Number}
                components = Array{Any}(undef, 2)
                components[1] = x
                components[2] = y
                m = length(x)+length(y)
                n = 1
                return judiVStack{T}(m, n, components)
            end
        end
    end
end

function vcat(x::judiVStack{T}, y::judiMultiSourceVector{T}) where {T<:Number}
    components = Array{Any}(undef, length(x.components) + 1)
    for i=1:length(x.components)
        components[i] = x.components[i]
    end
    components[end] = y
    m = x.m+length(y)
    n = 1
    return judiVStack{T}(m, n, components)
end

function vcat(x::judiMultiSourceVector{T}, y::judiVStack{T}) where {T<:Number}
    components = Array{Any}(undef, length(y.components) + 1)
    components[1] = x
    for i=2:length(y.components)+1
        components[i] = y.components[i-1]
    end
    m = y.m + length(x)
    n = 1
    return judiVStack{T}(m, n, components)
end

function vcat(x::judiVStack{T}, y::judiVStack{T}) where {T<:Number}
    nx = length(x.components)
    ny = length(y.components)
    components = Array{Any}(undef, nx+ny)
    for i=1:nx
        components[i] = x.components[i]
    end
    for i=nx+1:nx+ny
        components[i] = y.components[i-nx]
    end
    m = x.m + y.m
    n = 1
    return judiVStack{T}(m, n, components)
end


function *(F::joAbstractLinearOperator, v::judiVStack{T}) where {T<:Number}
    return sum(F.fop[i]*v[i] for i=1:length(v.components))
end

(msv::judiMultiSourceVector{T})(x::judiVStack{T}) where {T} = x

############################################################
## overloaded Base functions

# conj(jo)
conj(a::judiVStack{vDT}) where vDT =
    judiVStack{vDT}(a.m,a.n,a.components)

# transpose(jo)
transpose(a::judiVStack{vDT}) where vDT =
    judiVStack{vDT}(a.n,a.m,a.components)

# adjoint(jo)
adjoint(a::judiVStack{vDT}) where vDT =
    judiVStack{vDT}(a.n,a.m,a.components)

##########################################################
# Utilities

size(x::judiVStack{T}) where {T<:Number} = (x.m, x.n)
size(x::judiVStack{T}, ind::Integer) where {T<:Number} = (x.m, x.n)[ind]
length(x::judiVStack{T}) where {T<:Number} = x.m

eltype(::judiVStack{vDT}) where {vDT} = vDT

similar(x::judiVStack{T}) where {T<:Number} = judiVStack{Float32}(x.m, x.n, 0f0 .* x.components)
similar(x::judiVStack{T},  ::DataType, ::Union{AbstractUnitRange, Integer}...) where {T<:Number} = similar(x)

getindex(x::judiVStack{T}, a) where {T<:Number} = x.components[a]
firstindex(x::judiVStack{T}) where {T<:Number} = 1
lastindex(x::judiVStack{T}) where {T<:Number} = length(x.components)

dot(x::judiVStack{T}, y::judiVStack{T}) where {T<:Number} = T(sum(dot(x[i],y[i]) for i=1:length(x.components)))

function norm(x::judiVStack{T}, order::Real=2) where {T<:Number} 
    if order == Inf
        return max([norm(x[i], Inf) for i=1:length(x.components)]...)
    end
    out = sum(norm(x[i], order)^order for i=1:length(x.components))^(1/order)
    return T(out)
end

iterate(S::judiVStack{T}, state::Integer=1) where {T<:Number} = state > length(S.components) ? nothing : (S.components[state], state+1)
isfinite(S::judiVStack{T}) where {T<:Number} = all(isfinite(c) for c in S)

##########################################################

# minus
-(a::judiVStack{T}) where {T<:Number} = -1*a
+(a::judiVStack{T}, b::judiVStack{T}) where {T<:Number} = judiVStack{T}(a.m, a.n, a.components + b.components)
-(a::judiVStack{T}, b::judiVStack{T}) where {T<:Number} = judiVStack{T}(a.m, a.n, a.components - b.components)
+(a::judiVStack{T}, b::Number) where {T<:Number} = judiVStack{T}(a.m, a.n,a.components .+ b)
-(a::judiVStack{T}, b::Number) where {T<:Number} = judiVStack{T}(a.m, a.n,a.components .- b)
-(a::Number, b::judiVStack{T}) where {T<:Number} = judiVStack{T}(b.m, b.n, a .- b.components)

*(a::judiVStack{T}, b::Number) where {T<:Number} = judiVStack{T}(a.m, a.n, b .* a.components)
*(a::Number, b::judiVStack{T}) where {T<:Number} = b * a
+(a::Number, b::judiVStack{T}) where {T<:Number} = b + a

/(a::judiVStack{T}, b::Number) where {T<:Number} = judiVStack{T}(a.m, a.n, a.components ./ b)

##########################################################

BroadcastStyle(::Type{judiVStack}) = Base.Broadcast.DefaultArrayStyle{1}()

broadcasted(::typeof(+), x::judiVStack{T}, y::judiVStack{T}) where {T<:Number} = x + y
broadcasted(::typeof(-), x::judiVStack{T}, y::judiVStack{T}) where {T<:Number} = x - y
broadcasted(::typeof(+), x::judiVStack{T}, y::Number) where {T<:Number} = x + y
broadcasted(::typeof(-), x::judiVStack{T}, y::Number) where {T<:Number} = x - y
broadcasted(::typeof(+), y::Number, x::judiVStack{T}) where {T<:Number} = x + y
broadcasted(::typeof(-), y::Number, x::judiVStack{T}) where {T<:Number} = x - y

function broadcasted(::typeof(*), x::judiVStack{T}, y::judiVStack{T}) where {T<:Number}
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] = x.components[j] .* y.components[j]
    end
    return z
end

function broadcasted!(::typeof(*), x::judiVStack{T}, y::judiVStack{T}) where {T<:Number}
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] = x.components[j] .* y.components[j]
    end
    return z
end

function broadcasted(::typeof(/), x::judiVStack{T}, y::judiVStack{T}) where {T<:Number}
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] = x.components[j] ./ y.components[j]
    end
    return z
end

function broadcasted(::typeof(*), x::judiVStack{T}, y::Number) where {T<:Number}
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] .*= y
    end
    return z
end

broadcasted(::typeof(*), y::Number, x::judiVStack{T}) where {T<:Number} = x .* y

function broadcasted(::typeof(/), x::judiVStack{T}, y::Number) where {T<:Number}
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] ./= y
    end
    return z
end

function materialize!(x::judiVStack{T}, y::judiVStack{T}) where {T<:Number}
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    for j=1:length(x.components)
        try
            x.components[j].data .= y.components[j].data
        catch e
            x.components[j].weights .= y.components[j].weights
        end
    end
    return x
end

function broadcast!(::typeof(identity), x::judiVStack{T}, y::judiVStack{T}) where {T<:Number}
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    copy!(x,y)
end

broadcasted(::typeof(identity), x::judiVStack{T}) where {T<:Number} = x

function copy!(x::judiVStack{T}, y::judiVStack{T}) where {T<:Number}
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    for j=1:length(x.components)
        try
            x.components[j].data .= y.components[j].data
        catch e
            x.components[j].weights .= y.components[j].weights
        end
    end
end

function isapprox(x::judiVStack{T}, y::judiVStack; rtol::AbstractFloat=sqrt(eps()), atol::AbstractFloat=0.0) where {T<:Number}
    x.m == y.m || throw("Shape error")
    all(isapprox(xx, yy; rtol=rtol, atol=atol) for (xx, yy)=zip(x.components, y.components))
end

############################################################

function A_mul_B!(x::judiMultiSourceVector{Ts}, F::joCoreBlock{T, Ts}, y::judiVStack{T}) where {T<:Number, Ts<:Number}
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    x.data .= z.data
end

function A_mul_B!(x::judiVStack{Tv}, F::joCoreBlock{T, Tv}, y::judiMultiSourceVector{T}) where {T<:Number, Tv<:Number}
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.components)
        x.components[j].data .= z.components[j].data
    end
end

mul!(x::judiMultiSourceVector{Ts}, J::joCoreBlock{T, Ts}, y::judiVStack{T}) where {T<:Number, Ts<:Number} = A_mul_B!(x, J, y)
mul!(x::judiVStack{Tv}, J::joCoreBlock{T, Tv}, y::judiMultiSourceVector{T}) where {T<:Number, Tv<:Number} = A_mul_B!(x, J, y)
