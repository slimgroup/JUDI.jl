export judiProjection, judiWavelet, judiLRWF, nsrc

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

display(P::jAdjoint) where {D, O} = println("Adjoint($(P.op))")

# Projection operator 
struct judiProjection{D} <: judiNoopOperator{D}
    m::AbstractSize
    n::AbstractSize
    geometry::Geometry
end

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

==(P1::judiProjection, P2::judiProjection) = P1.data == P2.data
==(P1::judiWavelet, P2::judiWavelet) = (P1.wavelet == P2.wavelet && P1.dt == P2.dt)

# Prettyfier for data access
getproperty(P::judiProjection, s::Symbol) = s == :data ? P.geometry : getfield(P, s)
getproperty(P::judiWavelet, s::Symbol) = s == :data ? P.wavelet : getfield(P, s)

# Constructors
judiProjection(G::Geometry) = judiProjection{Float32}(_rec_space, time_space_size(2), G)
judiWavelet(nsrc::Integer, dt::T, w::Array{T, N}) where {T<:Real, N} = judiWavelet{Float32}(_time_space, _space, [w for i=1:nsrc], [dt for i=1:nsrc])
judiWavelet(dt::T, w::Array{T, N}) where {T<:Real, N} = judiWavelet(1, dt, w)
judiWavelet(dt::Vector{<:Number}, w::Vector{T}) where T<:Array = judiWavelet{Float32}(_time_space, _space, w, dt)
judiWavelet(dt::Vector{<:Number}, w::Vector{<:Number}) = judiWavelet{Float32}(_time_space, _space, [w for i=1:length(dt)], dt)
judiWavelet(dt::dtT, w::Array{T, N}) where {dtT<:Number, T<:Array, N} = judiWavelet([dt for i=1:length(w)], w)

# Deprecation error
judiWavelet(::Integer, ::Array{T, N}) where {T<:Real, N} = throw(ArgumentError("Time sampling of the wavelet need to be suplied `judiWavelet(nsrc, dt, wavelet)`"))

adjoint(P::judiNoopOperator{D}) where D = jAdjoint(P)
transpose(P::judiNoopOperator{D}) where D = jAdjoint(P)
conj(P::judiNoopOperator{D}) where D = P

display(P::judiProjection{D}) where {D, O} = println("JUDI projection operator $(repr(P.n)) -> $(repr(P.m))")

getindex(P::judiProjection{D}, i) where D = judiProjection{D}(P.m, P.n, P.geometry[i])
getindex(P::judiProjection{D}, i::Integer) where D = judiProjection{D}(P.m, P.n, P.geometry[i:i])

getindex(P::judiWavelet{D}, i) where D = judiWavelet{D}(P.m, P.n, P.wavelet[i], P.dt[i])
getindex(P::judiWavelet{D}, i::Integer) where D = judiWavelet{D}(P.m, P.n, P.wavelet[i:i], P.dt[i:i])

subsample(P::judiNoopOperator{D}, i) where D = getindex(P, i)

# Processing utilities
get_coords(P::judiProjection{D}) where D = hcat(P.geometry.xloc[1], P.geometry.zloc[1])
get_coords(P::judiWavelet{D}) where D = P.wavelet

out_type(::judiProjection{T}, ndim) where T = Array{Float32, 2}
out_type(::judiWavelet{T}, ndim) where T = Array{Float32, ndim}

process_out(rI::judiProjection{T}, dout, dtComp, ::Symbol) where T = judiVector{T, Array{T, 2}}(1, rI.geometry, [time_resample(dout, dtComp, rI.geometry)])

function process_out(::judiWavelet{T}, dout, dtComp, solver) where T
    we = getfield(JUDI, solver)
    padsize = we."model".padsizes
    dout = remove_padding(dout, padsize)
    judiWeights{T}(1, [dout])
end

###### Lazy injection

struct judiRHS{D} <: judiMultiSourceVector{D}
    nsrc::Integer
    P::jAdjoint{judiProjection{D}}
    d::judiVector
end

*(P::jAdjoint{judiProjection{D}}, d::judiVector{D, AT}) where {D, AT} = judiRHS{D}(d.nsrc, P, d)
getindex(rhs::judiRHS{D}, i) where D = judiRHS{D}(length(i), rhs.P[i], rhs.d[i])
make_input(rhs::judiRHS) = make_input(rhs.d)
eval(rhs::judiRHS) = rhs.d

# Combination of lazy injections
struct LazyAdd{D} <: judiMultiSourceVector{D}
    nsrc::Integer
    A
    B
    sign
end

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

getindex(la::LazyAdd{D}, i::RangeOrVec) where D = LazyAdd{D}(length(i), la.A[i], la.B[i], la.sign)

function eval(ls::LazyAdd{D}) where D
    aloc = eval(ls.A)
    bloc = eval(ls.B)
    ga = aloc.geometry
    gb = bloc.geometry
    @assert (ga.nt == gb.nt && ga.dt == gb.dt && ga.t == gb.t)
    xloc = [vcat(ga.xloc[1], gb.xloc[1])]
    yloc = [vcat(ga.yloc[1], gb.yloc[1])]
    zloc = [vcat(ga.zloc[1], gb.zloc[1])]
    geom = GeometryIC{D}(xloc, yloc, zloc, ga.dt, ga.nt, ga.t)
    data = hcat(aloc.data[1], ls.sign*bloc.data[1])
    judiVector{D, Matrix{D}}(1, geom, [data])
end

make_input(ls::LazyAdd{D}) where D = make_input(eval(ls))
