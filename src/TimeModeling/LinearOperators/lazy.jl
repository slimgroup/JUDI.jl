export judiProjection, judiWavelet, judiLRWF, judiRHS

############################################################################################################################
# Abstract types
abstract type judiNoopOperator{D} <: joAbstractLinearOperator{D, D} end

############################################################################################################################
# Concrete types
# Lazy adjoint like in LinearAlgebra
struct jAdjoint{T}
    op::T
end

# Projection operator 
struct judiProjection{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
    geometry::Geometry
end

# Wavelet "projection"
struct judiWavelet{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
    wavelet::Vector{Array{D, N}} where N
    dt::Vector{D}
end

# Poorly named backward compat
const judiLRWF{T} = judiWavelet{T}
const Projection{D} = Union{judiProjection{D}, judiWavelet{D}}
const AdjointProjection{D} = jAdjoint{<:Projection{D}}

"""
    judiRHS
        dt::Vector{T}
        geometry::Geometry
        data
Abstract sparse vector for right-hand-sides of the modeling operators. The `judiRHS` vector has the\\
dimensions of the full time history of the wavefields, but contains only the data defined at the \\
source or receiver positions (i.e. wavelets or shot records).

Constructor
==========
    judiRHS(dt, geometry, data)

Examples
========
Assuming `Pr` and `Ps` are projection operators of type `judiProjection` and `dobs` and `q` are\\
seismic vectors of type `judiVector`, then a `judiRHS` vector can be created as follows:
    rhs = Pr'*dobs    # right-hand-side with injected observed data
    rhs = Ps'*q    # right-hand-side with injected wavelet
"""
struct judiRHS{D} <: judiMultiSourceVector{D}
    nsrc::Integer
    P::jAdjoint{judiProjection{D}}
    d::judiVector
end


############################################################################################################################
# Constructors
"""
    judiProjection(geometry)

Projection operator for sources/receivers to restrict or inject data at specified locations.

Examples
========
`F` is a modeling operator of type `judiModeling` and `q` is a seismic source of type `judiVector`:
    Pr = judiProjection(rec_geometry)
    Ps = judiProjection(q.geometry)
    dobs = Pr*F*Ps'*q
    qad = Ps*F'*Pr'*dobs
"""
judiProjection(G::Geometry) = judiProjection{Float32}(rec_space(G), time_space_src(get_nsrc(G), G.nt, 3), G)

"""
    judiWavelet(dt, wavelet)

Low-rank wavefield operator which injects a wavelet q at every point of the subsurface.

Examples
========
`F` is a modeling operator of type `judiModeling` and `w` is a weighting matrix of type `judiWeights`:
    Pr = judiProjection(rec_geometry)
    Pw = judiWavelet(rec_geometry.dt, q.data) # or judiLRWF(rec_geometry.dt, q.data)
    dobs = Pr*F*Pw'*w
    dw = Pw*F'*Pr'*dobs
"""
judiWavelet(nsrc::Integer, dt::T, w::Array{T, N}) where {T<:AbstractFloat, N} = judiWavelet{Float32}(space_src(nsrc), time_space_src(nsrc, length.(w)), [w for i=1:nsrc], [dt for i=1:nsrc])
judiWavelet(dt::T, w::Array{T, N}) where {T<:AbstractFloat, N} = judiWavelet(1, dt, w)
judiWavelet(dt::Vector{<:Number}, w::Vector{T}) where T<:Array = judiWavelet{Float32}(space_src(length(dt)), time_space_src(length(dt), length.(w)), w, dt)
judiWavelet(dt::Vector{<:Number}, w::Vector{<:Number}) = judiWavelet{Float32}(space_src(length(dt)), time_space_src(length(dt), [length(w) for i=1:length(dt)]), [w for i=1:length(dt)], dt)
judiWavelet(dt::dtT, w::Array{T, N}) where {dtT<:Number, T<:Array, N} = judiWavelet([dt for i=1:length(w)], w)

# Deprecation error
judiWavelet(::Integer, ::Array{T, N}) where {T<:AbstractFloat, N} = throw(ArgumentError("Time sampling of the wavelet need to be suplied `judiWavelet(nsrc, dt, wavelet)`"))

############################################################################################################################
# Base overload
getproperty(P::judiProjection, s::Symbol) = s == :data ? P.geometry : getfield(P, s)
getproperty(P::judiWavelet, s::Symbol) = s == :data ? P.wavelet : getfield(P, s)

function getproperty(jA::jAdjoint, s::Symbol)
    s == :m && (return jA.op.n)
    s == :n && (return jA.op.m)
    s == :op && (return getfield(jA, s))
    return getproperty(jA.op, s)
end

size(jA::jAdjoint) = (jA.op.n, jA.op.m)
display(P::jAdjoint) = println("Adjoint($(P.op))")
display(P::judiProjection{D}) where D = println("JUDI projection operator $(repr(P.n)) -> $(repr(P.m))")

############################################################################################################################
# Indexing
getindex(jA::jAdjoint{T}, i) where T = jAdjoint{T}(jA.op[i])
getindex(P::judiProjection{D}, i) where D = judiProjection{D}(P.m[i], P.n[i], P.geometry[i])
getindex(P::judiProjection{D}, i::Integer) where D = judiProjection{D}(P.m[i], P.n[i], P.geometry[i:i])
getindex(P::judiWavelet{D}, i) where D = judiWavelet{D}(P.m[i], P.n[i], P.wavelet[i], P.dt[i])
getindex(P::judiWavelet{D}, i::Integer) where D = judiWavelet{D}(P.m[i], P.n[i], P.wavelet[i:i], P.dt[i:i])
getindex(rhs::judiRHS{D}, i::Integer) where D = judiRHS{D}(length(i), rhs.P[i], rhs.d[i])
getindex(rhs::judiRHS{D}, i::RangeOrVec) where D = judiRHS{D}(length(i), rhs.P[i], rhs.d[i])

# Backward compatible subsample
subsample(P::judiNoopOperator{D}, i) where D = getindex(P, i)

############################################################################################################################
# Linear algebra
adjoint(jA::jAdjoint) = jA.op
adjoint(P::judiNoopOperator{D}) where D = jAdjoint(P)

transpose(jA::jAdjoint) = jA.op
transpose(P::judiNoopOperator{D}) where D = jAdjoint(P)

conj(jA::jAdjoint) = jA
conj(P::judiNoopOperator{D}) where D = P

*(P::jAdjoint{judiProjection{D}}, d::judiVector{D, AT}) where {D, AT} = judiRHS{D}(d.nsrc, P, d)

# Combinations of rhs
for (op, s) in zip([:+, :-], (1, -1))
    for T1 in [judiRHS, LazyAdd]
        for T2 in [judiRHS, LazyAdd]
            @eval function $(op)(r1::$(T1){D}, r2::$(T2){D}) where D
                r1.nsrc == r2.nsrc || throw(ArgumentError("Incompatible number of source experiment"))
                LazyAdd{D}(r1.nsrc, r1, r2, $s)
            end
        end
    end
end

function *(M::Matrix{T}, P::judiProjection{T}) where T
    @assert size(M, 2) == get_nsrc(P)
    g = super_shot_geometry(P.geometry)
    geom = Geometry(g.xloc[1], g.yloc[1], g.zloc[1]; dt=g.dt[1], t=g.t[1], nsrc=size(M, 1))
    return judiProjection(geom)
end

*(M::Matrix{T}, P::jAdjoint{judiProjection{T}}) where T = jAdjoint(M*P.op)
*(M::Matrix{T}, R::judiRHS{T}) where T = judiRHS{T}(size(M, 1), M*R.P, M*R.d)

############################################################################################################################
# Comparison
==(jA1::jAdjoint, jA2::jAdjoint) = jA1.op  == jA2.op
==(P1::judiProjection, P2::judiProjection) = P1.data == P2.data
==(P1::judiWavelet, P2::judiWavelet) = (P1.wavelet == P2.wavelet && P1.dt == P2.dt)

############################################################################################################################
###### Inpout processing for propagation
function make_src(q::judiMultiSourceVector, P::judiProjection{D}) where D
    P.data[1] ≈ q.geometry[1] || throw(ArgumentError("Incompatible geometry and source"))
    return P.data[1], make_input(q)
end

make_src(q, P::judiProjection{D}) where D = (P.data[1], reshape(make_input(q), P.data.nt[1], P.data.nrec[1]))
make_src(q, P::judiLRWF{D}) where D = (make_input(q), P.data[1])
make_src(rhs::judiWavelet) = make_src(rhs.d, rhs.P)
make_src(q, P::jAdjoint{<:judiNoopOperator}) = make_src(q, P.op)

get_nsrc(P::Projection) = P.m[:src]
get_nsrc(P::jAdjoint{<:Projection}) = P.op.m[:src]
get_nt(P::Projection) = P.n[:time]
get_nt(P::jAdjoint{<:Projection}) = P.op.n[:time]

"""
    reshape(x, P::judiProjection, m::AbstractModel; with_batch=false)

reshape the input `x` into an ML friendly format `HWCB` using the projection operator and model to infer dimensions sizes.
"""
function reshape(x::Vector{T}, P::judiProjection{T}, ::AbstractModel; with_batch=false) where T
    out = reshape(x, P.geometry)
    out = with_batch ? reshape(out, size(out, 1), size(out, 2), 1, size(out, 3)) : out
    return out
end

function reshape(x::Vector{T}, P::judiWavelet{T}, m::AbstractModel; with_batch=false) where T
    out = with_batch ? reshape(x, m.n..., 1,get_nsrc(P)) : reshape(x, m.n..., get_nsrc(P))
    return out
end

function _as_src(P::judiLRWF{T}, model::AbstractModel, q::Array{T}) where T
    qcell = process_input_data(q, model, get_nsrc(P))
    return judiWeights{T}(get_nsrc(P), qcell)
end

function _as_src(P::judiProjection{T}, ::AbstractModel, q::Array{T}) where T
    qcell = process_input_data(q, P.geometry)
    return judiVector{T, Matrix{T}}(get_nsrc(P), P.geometry, qcell)
end

_as_src(::judiNoopOperator, ::AbstractModel, q::judiMultiSourceVector) = q


############################################################################################################################
###### Evaluate lazy operation
eval(rhs::judiRHS) = rhs.d
