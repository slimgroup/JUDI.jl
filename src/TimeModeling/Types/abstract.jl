export subsample

############################################################################################################################
# Base multi source abstract type
abstract type judiMultiSourceVector{T} <: AbstractVector{T} end

mutable struct judiMultiSourceException <: Exception
    msg :: String
end

display(ms::judiMultiSourceVector) = println("$(string(typeof(ms))) with $(ms.nsrc) sources")
show(io::IO, ms::judiMultiSourceVector) = print(io, "$(string(typeof(ms))) with $(ms.nsrc) sources")
showarg(io::IO, ms::judiMultiSourceVector, toplevel) = print(io, "$(string(typeof(ms))) with $(ms.nsrc) sources")
summary(io::IO, ms::judiMultiSourceVector) = print(io, string(typeof(ms)))
show(io::IO, ::MIME{Symbol("text/plain")}, ms::judiMultiSourceVector) = println(io, "$(typeof(ms)) with $(ms.nsrc) sources")

# Size and type 
eltype(::judiMultiSourceVector{T}) where T = T
size(jv::judiMultiSourceVector) = (jv.nsrc,)
length(jv::judiMultiSourceVector{T}) where {T} = sum([length(jv.data[i]) for i=1:jv.nsrc])

# Comparison
isequal(ms1::judiMultiSourceVector, ms2::judiMultiSourceVector) = ms1 == ms2
==(ms1::judiMultiSourceVector, ms2::judiMultiSourceVector) = all(getfield(ms1, s) == getfield(ms2, s) for s in fieldnames(typeof(ms1)))

isapprox(x::judiMultiSourceVector, y::judiMultiSourceVector; rtol::AbstractFloat=sqrt(eps()), atol::AbstractFloat=0.0) =
    all(isapprox(getfield(x, f), getfield(y, f); rtol=rtol, atol=atol) for f in fieldnames(typeof(x)))

check_compat(ms::Vararg{judiMultiSourceVector, N}) where N = true
check_compat(x::Number, ms::judiMultiSourceVector) = true
check_compat(ms::judiMultiSourceVector, x::Number) = true

# Copy
copyto!(ms::judiMultiSourceVector, a::Vector{Array{T, N}}) where {T, N} = copyto!(ms.data, a)
copyto!(a::Vector{Array{T, N}}, ms::judiMultiSourceVector) where {T, N} = copyto!(a, ms.data)
copy(ms::judiMultiSourceVector{T}) where {T} = begin y = zero(T, ms); y.data = deepcopy(ms.data); y end
deepcopy(ms::judiMultiSourceVector{T}) where {T} = copy(ms)
unsafe_convert(::Type{Ptr{T}}, msv::judiMultiSourceVector{T}) where {T} = unsafe_convert(Ptr{T}, msv.data)

# indexing
IndexStyle(::Type{<:judiMultiSourceVector}) = IndexLinear()
setindex!(ms::judiMultiSourceVector{T}, v::Array{T, N}, i::Integer) where {T, N} = begin ms.data[i] = v; nothing end
getindex(ms::judiMultiSourceVector{T}, i::Integer) where {T} = i > ms.nsrc ? 0 : ms[i:i]
getindex(ms::judiMultiSourceVector{T}, ::Colon) where {T} = vec(ms)
firstindex(ms::judiMultiSourceVector) = 1
lastindex(ms::judiMultiSourceVector) = ms.nsrc
iterate(S::judiMultiSourceVector, state::Integer=1) = state > S.nsrc ? nothing : (S[state], state+1)
# Backward compat subsample
subsample(ms::judiMultiSourceVector, i) = getindex(ms, i)

zero(::Type{T}, x::judiMultiSourceVector) where T = throw(judiMultiSourceException("$(typeof(x)) does not implement zero copy zero(::Type{T}, x)"))
similar(x::judiMultiSourceVector{T}) where T = zero(T, x)
similar(x::judiMultiSourceVector, nsrc::Integer) = nsrc < x.nsrc ? zero(eltype(ET), x; nsrc=nsrc) : zero(eltype(ET), x)
similar(x::judiMultiSourceVector, ::Type{ET}) where ET = zero(eltype(ET), x)
similar(x::judiMultiSourceVector, ::Type{ET}, dims::AbstractUnitRange) where ET = similar(x, ET)
similar(x::judiMultiSourceVector, ::Type{ET}, nsrc::Integer) where ET = nsrc <= x.nsrc ? zero(eltype(ET), x; nsrc=nsrc) : similar(x, ET)

jo_convert(::Type{Array{T, N}}, v::judiMultiSourceVector, B::Bool) where {T, N} = jo_convert(T, v, B)

fill!(x::judiMultiSourceVector, val) = fill!.(x.data, val)
sum(x::judiMultiSourceVector) = sum(sum(x.data))

isfinite(v::judiMultiSourceVector) = all(all(isfinite.(v.data[i])) for i=1:v.nsrc)

conj(A::judiMultiSourceVector{vDT}) where vDT = A
transpose(A::judiMultiSourceVector{vDT}) where vDT = A
adjoint(A::judiMultiSourceVector{vDT}) where vDT = A

maximum(a::judiMultiSourceVector{avDT}) where avDT = max([maximum(a.data[i]) for i=1:a.nsrc]...)
minimum(a::judiMultiSourceVector{avDT}) where avDT = min([minimum(a.data[i]) for i=1:a.nsrc]...)

vec(x::judiMultiSourceVector) = vcat(vec.(x.data)...)

time_sampling(ms::judiMultiSourceVector) = [1 for i=1:ms.nsrc]

reshape(ms::judiMultiSourceVector, dims::Dims{1}) = ms ### during AD, size(ms::judiVector) = ms.nsrc
reshape(ms::judiMultiSourceVector, dims::Dims{N}) where N = reshape(vec(ms), dims)

############################################################################################################################
# Linear algebra `*`
(msv::judiMultiSourceVector{mT})(x::AbstractVector{T}) where {mT, T<:Number} = x
(msv::judiMultiSourceVector{T})(x::judiMultiSourceVector{T}) where {T<:Number} = x
(msv::judiMultiSourceVector{mT})(x::AbstractVector{T}) where {mT, T<:Array} = begin y = deepcopy(msv); y.data .= x; return y end

function *(J::Union{Matrix{vDT}, joAbstractLinearOperator}, x::judiMultiSourceVector{vDT}) where vDT
    outvec = try J.fop(x) catch; J*vec(x) end
    outdata = try reshape(outvec, size(x.data[1]), x.nsrc) catch; outvec end
    return x(outdata)
end

function *(J::joCoreBlock, x::judiMultiSourceVector{vDT}) where vDT
    outvec = vcat([J.fop[i]*x for i=1:J.l]...)
    outdata = try reshape(outvec, size(x.data[1]), J.l*x.nsrc); catch; outvec end
    return x(outdata)
end

# Propagation input
make_input(ms::judiMultiSourceVector) = throw(judiMultiSourceException("$(typeof(ms)) must implement `make_input(ms, dt)` for propagation"))
make_input(a::Array) = a

as_src(ms::judiMultiSourceVector{T}) where T = ms
as_src(p::AbstractVector{T}) where T = p
as_src(p) = vec(p)
############################################################################################################################
# Linear algebra norm/abs/cat...
function norm(a::judiMultiSourceVector{T}, p::Real=2) where T
    if p == Inf
        return maximum([norm(a.data[i], p) for i=1:a.nsrc])
    end
    x = 0.f0
    dt = time_sampling(a)
    for j=1:a.nsrc
        x += dt[j] * norm(a.data[j], p)^p
    end
    return T(x^(1.f0/p))
end

# inner product
function dot(a::judiMultiSourceVector{T}, b::judiMultiSourceVector{T}) where T
    # Dot product for data containers
    size(a) == size(b) || throw(judiMultiSourceException("dimension mismatch: $(size(a)) != $(size(b))"))
    dotprod = 0f0
    dt = time_sampling(a)
    for j=1:a.nsrc
        dotprod += dt[j] * dot(a.data[j], b.data[j])
    end
    return T(dotprod)
end

dot(a::judiMultiSourceVector{T}, b::Array{T}) where T = dot(vec(a), vec(b))
dot(a::Array{T}, b::judiMultiSourceVector{T}) where T = dot(b, a)

# abs
function abs(a::judiMultiSourceVector{T}) where T
    b = deepcopy(a)
    for j=1:a.nsrc
        b.data[j] = abs.(a.data[j])
    end
    return b
end

# vcat
vcat(a::Array{<:judiMultiSourceVector, 1}) = vcat(a...)

function vcat(ai::Vararg{<:judiMultiSourceVector{T}, N}) where {T, N}
    N == 1 && (return ai[1])
    N > 2 && (return vcat(ai[1], vcat(ai[2:end]...)))
    a, b = ai
    res = deepcopy(a)
    push!(res, b)
    return res
end

############################################################################################################################
# Diff/cumsum
function cumsum(x::judiMultiSourceVector;dims=1)
    y = deepcopy(x)
    cumsum!(y, x; dims=dims)
    return y
end

function cumsum!(y::judiMultiSourceVector, x::judiMultiSourceVector;dims=1)
    dims == 1 || dims == 2 || throw(judiMultiSourceException("Dimension $(dims) is out of range for a 2D array"))
    h = dims == 1 ? time_sampling(x) : [1f0 for i=1:x.nsrc] # receiver dimension is non-dimensional
    for i = 1:x.nsrc
        cumsum!(y.data[i], x.data[i], dims=dims)
        lmul!(h[i], y.data[i])
    end
    return y
end

function diff(x::judiMultiSourceVector; dims=1)
    # note that this is not the same as default diff in julia, the first entry stays in the diff result
    dims == 1 || dims == 2 || throw(judiMultiSourceException("Dimension $(dims) is out of range for a 2D array"))
    y = 1f0 * x        # receiver dimension is non-dimensional
    h = dims == 1 ? time_sampling(x) : [1f0 for i=1:x.nsrc]
    for i = 1:x.nsrc
        n = size(y.data[i],dims)
        selectdim(y.data[i], dims, 2:n) .-= selectdim(y.data[i], dims, 1:n-1)
        lmul!(1 / h[i], y.data[i])
    end
    return y
end


############################################################################################################################
# Type conversions
tof32(x::Number) = [Float32(x)]
tof32(x::Array{T, N}) where {N, T<:Real} = T==Float32 ? x : Float32.(x)
tof32(x::Array{Array{T, N}, 1}) where {N, T<:Real} = T==Float32 ? x : tof32.(x)
tof32(x::Array{Any, 1}) = try Float32.(x) catch e tof32.(x) end
tof32(x::StepRangeLen) = convert(Vector{Float32}, x)
tof32(x::Vector{<:StepRangeLen}) = tof32.(x)
