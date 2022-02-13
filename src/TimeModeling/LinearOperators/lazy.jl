export judiProjection, judiLRWF

# Base abstract type
abstract type judiNoopOperator{D} <: joAbstractLinearOperator{D, D} end

# Lazy adjoint like in LinearAlgebra
struct jAdjoint{T}
    op::T
end

function getproperty(jA::jAdjoint, s::Symbol)
    s == :m && (return jA.op.n)
    s == :n && (return jA.op.m)
    s == :op && (return getfield(jA, s))
    return getproperty(jA.op, s)
end

size(jA::jAdjoint) = (jA.op.n, jA.op.m)
adjoint(jA::jAdjoint) = jA.op
transpose(jA::jAdjoint) = jA.op
conj(jA::jAdjoint) = jA
getindex(jA::jAdjoint{T}, i) where T = jAdjoint{T}(jA.op[i])
==(jA1::jAdjoint, jA2::jAdjoint) = jA1.op  == jA2.op

# Projection operator 
struct judiProjection{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
    geometry::Geometry
end


struct judiLRWF{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
    wavelet::Vector{Array{D, N}} where N
    dt::Vector{D}
end

const Projection{D} = Union{judiProjection{D}, judiLRWF{D}}
const AdjointProjection{D} = jAdjoint{<:Projection{D}}

==(P1::judiProjection, P2::judiProjection) = P1.data == P2.data
==(P1::judiLRWF, P2::judiLRWF) = (P1.wavelet == P2.wavelet && P1.dt == P2.dt)

# Prettyfier for data access
getproperty(P::judiProjection, s::Symbol) = s == :data ? P.geometry : getfield(P, s)
getproperty(P::judiLRWF, s::Symbol) = s == :data ? P.wavelet : getfield(P, s)

# Constructors
judiProjection(G::Geometry) = judiProjection{Float32}(_rec_space, time_space_size(2), G)
judiLRWF(nsrc::Integer, dt::T, w::Vector{T}) where T<:Real = judiLRWF{Float32}(_time_space, _space, [w for i=1:nsrc], [dt for i=1:nsrc])
judiLRWF(dt::T, w::Vector{T}) where T<:Real = judiLRWF(1, dt, w)
judiLRWF(dt::Vector{<:Number}, w::Vector{T}) where T<:Array = judiLRWF{Float32}(_time_space, _space, w, dt)
judiLRWF(dt::dtT, w::Vector{T}) where {dtT<:Number, T<:Array} = judiLRWF([dt for i=1:length(w)], w)

# Deprecation error
judiLRWF(nsrc::Integer, w::Vector{T}) where T<:Real = throw(ArgumentError("Time sampling of the wavelet need to be suplied `judiLRWF(nsrc, dt, wavelet)`"))

adjoint(P::judiNoopOperator{D}) where D = jAdjoint(P)
transpose(P::judiNoopOperator{D}) where D = jAdjoint(P)
conj(P::judiNoopOperator{D}) where D = P

display(P::judiProjection{D}) where {D, O} = println("JUDI projection operator $(repr(P.n)) -> $(repr(P.m))")

getindex(P::judiProjection{D}, i) where D = judiProjection{D}(P.m, P.n, P.geometry[i])
getindex(P::judiProjection{D}, i::Integer) where D = judiProjection{D}(P.m, P.n, P.geometry[i:i])

getindex(P::judiLRWF{D}, i) where D = judiLRWF{D}(P.m, P.n, P.wavelet[i], P.dt[i])
getindex(P::judiLRWF{D}, i::Integer) where D = judiLRWF{D}(P.m, P.n, P.wavelet[i:i], P.dt[i:i])

subsample(P::judiNoopOperator{D}, i) where D = getindex(P, i)

# Processing utilities
get_coords(P::judiProjection{D}) where D = hcat(P.geometry.xloc[1], P.geometry.zloc[1])
get_coords(P::judiLRWF{D}) where D = P.wavelet

out_type(::judiProjection{T}, ndim) where T = Array{Float32, 2}
out_type(::judiLRWF{T}, ndim) where T = Array{Float32, ndim}

process_out(rI::judiProjection{T}, dout, dtComp) where T = judiVector{T, Array{T, 2}}(1, rI.geometry, [time_resample(dout, dtComp, rI.geometry.dt[1])])
process_out(rI::judiLRWF{T}, dout, dtComp) where T = judiWeights{T}(1, [dout])

make_input(P::judiProjection, dtComp) = Dict(:src_coords=>get_coords(P))
make_input(P::jAdjoint{<:judiProjection}, dtComp) = Dict(:rec_coords=>get_coords(P.op))
make_input(P::judiLRWF, dtComp) = Dict(:wavelet=>time_resample(P.data[1], P.dt[1], dtComp))
make_input(P::jAdjoint{<:judiLRWF}, dtComp) = Dict(:ws=>time_resample(P.data[1], P.dt[1], dtComp))