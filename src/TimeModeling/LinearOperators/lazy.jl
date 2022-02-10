export judiProjection

# Lazy adjoint like in LinearAlgebra
struct jAdjoint{T}
    op::T
end

function getproperty(jA::jAdjoint, s::Symbol)
    s == :m && (return jA.op.n)
    s == :n && (return jA.op.m)
    s == :op && (return getfield(jA, s))
    return getfield(jA.op, s)
end

# Base abstract type
abstract type judiNoopOperator{D} <: joAbstractLinearOperator{D, D} end

struct judiAbstractProjection{D, T} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
    data::T
end

const judiProjection{D} = judiAbstractProjection{D, Geometry}
const judiLRWF{D} = judiAbstractProjection{D, Vector{D}}
const AnyProjection{D} = Union{judiProjection{D}, judiLRWF{D}}


# Prettyfier for data access
getproperty(P::judiProjection, s::Symbol) = s == :geometry ? P.data : getfield(P, s)
getproperty(P::judiLRWF, s::Symbol) = s == :wavelet ? P.data : getfield(P, s)

# Constructors
judiProjection(G::Geometry) = judiProjection{Float32}(_rec_space, time_space_size(2), G)
judiLRWF(nsrc::Integer, w::Vector{T}) where T<:Real = judiLRWF{Float32}(_time_space, _space, [w for i=1:nsrc])
judiLRWF(w::Vector{T}) where T<:Real = judiLRWF(1, w)
judiLRWF(w::Vector{T}) where T<:Array = judiLRWF{Float32}(_time_space, _space, w)

adjoint(P::judiNoopOperator{D}) where D = jAdjoint(P)
transpose(P::judiNoopOperator{D}) where D = jAdjoint(P)
conj(P::judiNoopOperator{D}) where D = P

display(P::judiProjection{D}) where {D, O} = println("JUDI projection operator $(repr(P.n)) -> $(repr(P.m))")

getindex(P::judiAbstractProjection{D}, i) where D = judiProjection{Float32}(P.m, P.n, P.data[i])
subsample(P::judiAbstractProjection{D}, i) where D = getindex(P, i)

# Processing utilities
get_coords(P::judiProjection{D}) where D = hcat(P.geometry.xloc[1], P.geometry.zloc[1])
get_coords(P::judiLRWF{D}) where D = P.wavelet

out_type(::judiProjection{T}, ndim) where T = Array{Float32, 2}
out_type(::judiLRWF{T}, ndim) where T = Array{Float32, ndim}

process_out(dout, rI::judiProjection, dtComp) = judiVector{Float32, Array{Float32, 2}}(1, rI.gometry, [time_resample(dout, dtComp, rI.geometry.dt[1])])
process_out(dout, rI::judiLRWF, dtComp) = dout