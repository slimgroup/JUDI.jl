"""
Vector of a judiVector and a judiWeight
"""
mutable struct judiVStack{vDT<:Number}
    m::Integer
    n::Integer
    components::Array{Any, 1}
end

function vcat(x::judiMultiSourceVector, y::judiMultiSourceVector)
    components = Array{Any}(undef, 2)
    components[1] = x
    components[2] = y
    m = length(x)+length(y)
    n = 1
    return judiVStack{Float32}(m, n, components)
end

function vcat(x::judiVStack, y::judiMultiSourceVector)
    components = Array{Any}(undef, length(x.components) + 1)
    for i=1:length(x.components)
        components[i] = x.components[i]
    end
    components[end] = y
    m = x.m+length(y)
    n = 1
    return judiVStack{Float32}(m, n, components)
end

function vcat(x::judiMultiSourceVector, y::judiVStack)
    components = Array{Any}(undef, length(y.components) + 1)
    components[1] = x
    for i=2:length(y.components)+1
        components[i] = y.components[i-1]
    end
    m = y.m + length(x)
    n = 1
    return judiVStack{Float32}(m, n, components)
end

function vcat(x::judiVStack, y::judiVStack)
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
    return judiVStack{Float32}(m, n, components)
end


for T ∈ [judiVector, judiWeights, judiWavefield]
    @eval begin
        function vcat(x::$T, y::$T)
            xn = deepcopy(x)
            push!(xn, y)
            return xn
        end
    end
end

function *(F::joAbstractLinearOperator, v::judiVStack)
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

size(x::judiVStack) = (x.m, x.n)
size(x::judiVStack, ind::Integer) = (x.m, x.n)[ind]

length(x::judiVStack) = x.m

eltype(v::judiVStack{vDT}) where {vDT} = vDT

similar(x::judiVStack) = judiVStack{Float32}(x.m, x.n, 0f0 .* x.components)

similar(x::judiVStack,  element_type::DataType, dims::Union{AbstractUnitRange, Integer}...) = similar(x)

getindex(x::judiVStack, a) = x.components[a]

firstindex(x::judiVStack) = 1

lastindex(x::judiVStack) = length(x.components)

dot(x::judiVStack, y::judiVStack) = sum(dot(x[i],y[i]) for i=1:length(x.components))

function norm(x::judiVStack, order::Real=2)
    if order == Inf
        return max([norm(x[i], Inf) for i=1:length(x.components)]...)
    end
    sum(norm(x[i], order)^order for i=1:length(x.components))^(1/order)
end

iterate(S::judiVStack, state::Integer=1) = state > length(S.components) ? nothing : (S.components[state], state+1)

isfinite(S::judiVStack) = all(isfinite(c) for c in S)

##########################################################

# minus
-(a::judiVStack) = -1*a
+(a::judiVStack, b::judiVStack) = judiVStack{Float32}(a.m, a.n, a.components + b.components)
-(a::judiVStack, b::judiVStack) = judiVStack{Float32}(a.m, a.n, a.components - b.components)
+(a::judiVStack, b::Number) = judiVStack{Float32}(a.m, a.n,a.components .+ b)
-(a::judiVStack, b::Number) = judiVStack{Float32}(a.m, a.n,a.components .- b)
-(a::Number, b::judiVStack) = judiVStack{Float32}(b.m, b.n, a .- b.components)

*(a::judiVStack, b::Number) = judiVStack{Float32}(a.m, a.n, b .* a.components)
*(a::Number, b::judiVStack) = b * a
+(a::Number, b::judiVStack) = b + a

/(a::judiVStack, b::Number) = judiVStack{Float32}(a.m, a.n, a.components ./ b)

##########################################################

BroadcastStyle(::Type{judiVStack}) = Base.Broadcast.DefaultArrayStyle{1}()

broadcasted(::typeof(+), x::judiVStack, y::judiVStack) = x + y
broadcasted(::typeof(-), x::judiVStack, y::judiVStack) = x - y
broadcasted(::typeof(+), x::judiVStack, y::Number) = x + y
broadcasted(::typeof(-), x::judiVStack, y::Number) = x - y
broadcasted(::typeof(+), y::Number, x::judiVStack) = x + y
broadcasted(::typeof(-), y::Number, x::judiVStack) = x - y

function broadcasted(::typeof(*), x::judiVStack, y::judiVStack)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] = x.components[j] .* y.components[j]
    end
    return z
end

function broadcasted!(::typeof(*), x::judiVStack, y::judiVStack)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] = x.components[j] .* y.components[j]
    end
    return z
end

function broadcasted(::typeof(/), x::judiVStack, y::judiVStack)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] = x.components[j] ./ y.components[j]
    end
    return z
end

function broadcasted(::typeof(*), x::judiVStack, y::Number)
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] .*= y
    end
    return z
end

broadcasted(::typeof(*), y::Number, x::judiVStack) = x .* y

function broadcasted(::typeof(/), x::judiVStack, y::Number)
    z = deepcopy(x)
    for j=1:length(x.components)
        z.components[j] ./= y
    end
    return z
end

function materialize!(x::judiVStack, y::judiVStack)
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

function broadcast!(::typeof(identity), x::judiVStack, y::judiVStack)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    copy!(x,y)
end

function broadcasted(::typeof(identity), x::judiVStack)
    return x
end

function copy!(x::judiVStack, y::judiVStack)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    for j=1:length(x.components)
        try
            x.components[j].data .= y.components[j].data
        catch e
            x.components[j].weights .= y.components[j].weights
        end
    end
end

function isapprox(x::judiVStack, y::judiVStack; rtol::AbstractFloat=sqrt(eps()), atol::AbstractFloat=0.0)
    x.m == y.m || throw("Shape error")
    all(isapprox(xx, yy; rtol=rtol, atol=atol) for (xx, yy)=zip(x.components, y.components))
end

############################################################

function A_mul_B!(x::judiMultiSourceVector, F::joCoreBlock, y::judiVStack)
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    x.data .= z.data
end

function A_mul_B!(x::judiVStack, F::joCoreBlock, y::judiMultiSourceVector)
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.components)
        x.components[j].data .= z.components[j].data
    end
end

mul!(x::judiMultiSourceVector, J::joCoreBlock, y::judiVStack) = A_mul_B!(x, J, y)
mul!(x::judiVStack, J::joCoreBlock, y::judiMultiSourceVector) = A_mul_B!(x, J, y)
