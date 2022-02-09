# Base multi source abstract type
abstract type judiMultiSourceVector{T} <: DenseVector{T} end


IndexStyle(::Type{<:judiMultiSourceVector}) = IndexLinear()

copyto!(ms::judiMultiSourceVector, a::Vector{Array{T, N}}) = copyto!(ms.data, a)
copyto!(a::Vector{Array{T, N}}, ms::judiMultiSourceVector) = copyto!(a, ms.data)

eltype(::judiMultiSourceVector{T}) where T = T

size(jv::judiMultiSourceVector) = (length(jv),)
length(jv::judiMultiSourceVector{T, AT}) where {T, AT} = jv.nsrc

similar(x::judiMultiSourceVector) = 0f0*x
similar(x::judiMultiSourceVector, element_type::DataType, dims::Union{AbstractUnitRange, Integer}...) = 0f0*x

fill!(x::judiMultiSourceVector, val) where {vDT, AT} = fill!.(x.data, val)
sum(x::judiMultiSourceVector) = sum(sum(x.data))

isfinite(v::judiMultiSourceVector) = all(all(isfinite.(v.data[i])) for i=1:v.nsrc)

subsample(v::judiMultiSourceVector, srcnum) = v[srcnum]

time_sampling(ms::judiMultiSourceVector) = (1 for i=1:ms.nsrc)

transpose(A::judiMultiSourceVector{vDT}) where vDT = A
adjoint(A::judiMultiSourceVector{vDT}) where vDT = conj(transpose(A))

isapprox(x::judiMultiSourceVector, y::judiMultiSourceVector; rtol::Real=sqrt(eps()), atol::Real=0) =
    all(isapprox(getfield(x, f), getfield(x, f); rtol=rtol, atol=atol) for f in fieldnames(tyepof(x)))

maximum(a::judiWeights{avDT}) where avDT = max([maximum(a.data[i]) for i=1:a.nsrc]...)
minimum(a::judiWeights{avDT}) where avDT = min([minimum(a.data[i]) for i=1:a.nsrc]...)

# norm
function norm(a::judiMultiSourceVector{T}, p::Real=2) where T
    if p == Inf
        return max([maximum(abs.(a.data[i])) for i=1:a.info.nsrc]...)
    end
    x = 0.f0
    dt = time_sampling(a)
    for j=1:a.info.nsrc
        x += dt[j] * sum(abs.(vec(a.data[j])).^p)
    end
    return T(x^(1.f0/p))
end

# inner product
function dot(a::judiMultiSourceVector{T}, b::judiMultiSourceVector{T}) where T
	# Dot product for data containers
	size(a) == size(b) || throw(judiWavefieldException("dimension mismatch"))
	dotprod = 0f0
    dt = time_sampling(a)
	for j=1:a.info.nsrc
		dotprod += dt[j] * dot(vec(a.data[j]),vec(b.data[j]))
	end
	return T(dotprod)
end

# abs
function abs(a::judiMultiSourceVector{T}) where T
	b = deepcopy(a)
	for j=1:a.info.nsrc
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

# Bypass mismatch in naming and fields
Base.getproperty(obj::judiWeights, sym::Symbol) = sym == :weights ? getfield(obj, :data) : getfield(obj, sym)

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