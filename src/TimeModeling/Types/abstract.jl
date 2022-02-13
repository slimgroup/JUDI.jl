# Base multi source abstract type
abstract type judiMultiSourceVector{T} <: DenseVector{T} end

mutable struct judiMultiSourceException <: Exception
    msg :: String
end

make_input(ms::judiMultiSourceVector, dt) = throw(judiMultiSourceException("$(typeof(ms)) must implement `make_input(ms, dt)` for propagation"))

isequal(ms1::judiMultiSourceVector, ms2::judiMultiSourceVector) = ms1 == ms2
==(ms1::judiMultiSourceVector, ms2::judiMultiSourceVector) = all(getfield(ms1, s) == getfield(ms2, s) for s in fieldnames(typeof(ms1)))

unsafe_convert(::Type{Ptr{T}}, msv::judiMultiSourceVector{T}) where {T} = unsafe_convert(Ptr{T}, msv.data)

display(ms::judiMultiSourceVector) = println("$(typeof(ms)) wiht $(ms.nsrc) sources")

IndexStyle(::Type{<:judiMultiSourceVector}) = IndexLinear()

copyto!(ms::judiMultiSourceVector, a::Vector{Array{T, N}}) where {T, N} = copyto!(ms.data, a)
copyto!(a::Vector{Array{T, N}}, ms::judiMultiSourceVector) where {T, N} = copyto!(a, ms.data)
copy(ms::judiMultiSourceVector{T}) where {T} = begin y = zero(T, ms); y.data = deepcopy(ms.data); y end
deepcopy(ms::judiMultiSourceVector{T}) where {T} = copy(ms)

setindex!(ms::judiMultiSourceVector{T}, v::Array{T, N}, i::Integer) where {T, N} = begin ms.data[i] = v; nothing end
getindex(ms::judiMultiSourceVector{T}, i) where {T} = subsample(ms, i)
iterate(S::judiMultiSourceVector, state::Integer=1) = state > S.nsrc ? nothing : (S[state], state+1)

eltype(::judiMultiSourceVector{T}) where T = T

size(jv::judiMultiSourceVector) = (length(jv),)
length(jv::judiMultiSourceVector{T}) where {T} = jv.nsrc

zero(::Type{T}, x::judiMultiSourceVector) where T = throw(judiMultiSourceException("$(typeof(x)) does not implement zero copy zero(::Type{T}, x)"))

similar(x::judiMultiSourceVector{T}) where T = zero(T, x)
similar(x::judiMultiSourceVector, ET::DataType) = zero(eltype(ET), x)
similar(x::judiMultiSourceVector, ET::DataType, dims::Union{AbstractUnitRange, Integer}...) = zero(eltype(ET), x)[dims...]

jo_convert(::Type{Array{T, N}}, v::judiMultiSourceVector, B::Bool) where {T, N} = jo_convert(T, v, B)

fill!(x::judiMultiSourceVector, val) = fill!.(x.data, val)
sum(x::judiMultiSourceVector) = sum(sum(x.data))

isfinite(v::judiMultiSourceVector) = all(all(isfinite.(v.data[i])) for i=1:v.nsrc)

subsample(v::judiMultiSourceVector, srcnum) = v[srcnum]

time_sampling(ms::judiMultiSourceVector) = [1 for i=1:ms.nsrc]

conj(A::judiMultiSourceVector{vDT}) where vDT = A
transpose(A::judiMultiSourceVector{vDT}) where vDT = A
adjoint(A::judiMultiSourceVector{vDT}) where vDT = A

isapprox(x::judiMultiSourceVector, y::judiMultiSourceVector; rtol::Real=sqrt(eps()), atol::Real=0) =
    all(isapprox(getfield(x, f), getfield(y, f); rtol=rtol, atol=atol) for f in fieldnames(typeof(x)))

maximum(a::judiMultiSourceVector{avDT}) where avDT = max([maximum(a.data[i]) for i=1:a.nsrc]...)
minimum(a::judiMultiSourceVector{avDT}) where avDT = min([minimum(a.data[i]) for i=1:a.nsrc]...)

vec(x::judiMultiSourceVector) = vcat(vec.(x.data)...)

(msv::judiMultiSourceVector{T})(x::Vector{T}) where {T<:Real} = x
(msv::judiMultiSourceVector{T})(x::judiMultiSourceVector{T}) where {T} = x
(msv::judiMultiSourceVector{mT})(x::Vector{T}) where {mT, T<:Array} = begin y = deepcopy(msv); y.data .= x; return y end

function *(J::Union{Matrix{vDT}, joAbstractLinearOperator}, x::judiMultiSourceVector{vDT}) where vDT
    outvec = try J.fop(x) catch; J*vec(x) end
    outdata = try reshape(outvec, size(x.data[1]), x.nsrc) catch; outvec end
    return x(outdata)
end

function *(J::joCoreBlock, x::judiMultiSourceVector{vDT}) where vDT
    outvec = vcat([J.fop[i]*vec(x) for i=1:J.l]...)
    outdata = try reshape(outvec, size(x.data[1]), J.l*x.nsrc); catch; outvec end
    return x(outdata)
end

function norm(a::judiMultiSourceVector{T}, p::Real=2) where T
    if p == Inf
        return max([maximum(abs.(a.data[i])) for i=1:a.nsrc]...)
    end
    x = 0.f0
    dt = time_sampling(a)
    for j=1:a.nsrc
        x += dt[j] * sum(abs.(vec(a.data[j])).^p)
    end
    return T(x^(1.f0/p))
end

# inner product
function dot(a::judiMultiSourceVector{T}, b::judiMultiSourceVector{T}) where T
	# Dot product for data containers
	size(a) == size(b) || throw(judiMultiSourceException("dimension mismatch"))
	dotprod = 0f0
    dt = time_sampling(a)
	for j=1:a.nsrc
		dotprod += dt[j] * dot(vec(a.data[j]),vec(b.data[j]))
	end
	return T(dotprod)
end

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

# Type conversions

function matmulT(a::AbstractArray{T, 2}, b) where T
    return a*vec(vcat(b.data...))
end

tof32(x::Number) = [Float32(x)]
tof32(x::Array{T, N}) where {N, T<:Real} = T==Float32 ? x : Float32.(x)
tof32(x::Array{Array{T, N}, 1}) where {N, T<:Real} = T==Float32 ? x : tof32.(x)
tof32(x::Array{Any, 1}) = try Float32.(x) catch e tof32.(x) end
tof32(x::StepRangeLen) = tof32.(x)
tof32(x::Array{StepRangeLen}) = tof32.(x)

##### Rebuild bad vector

function rebuild_maybe_jld(x::Vector{Any})
    try
        return tof32(x)
    catch e
        if hasproperty(x[1], :offset)
            return [Float32.(StepRangeLen(xi.ref, xi.step, xi.len, xi.offset)) for xi in x]
        end
        return x
    end
end

## Propagation
get_source(x::judiMultiSourceVector, dtComp) = x.data[1]