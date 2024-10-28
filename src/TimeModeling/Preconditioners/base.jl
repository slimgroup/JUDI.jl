abstract type Preconditioner{D, R} <: joAbstractLinearOperator{D, R} end

"""
    ModelPreconditioner{T}

Base abstract type for model space preconditioners. Acta on [`PhysicalParameter{T}`](@ref) or `Vector{T}`
"""
abstract type ModelPreconditioner{D, R} <: Preconditioner{D, R} end

"""
    DataPreconditioner{T}

Base abstract for Data space preconditioner. As a data space operator, it must implement `getindex(M, i)` to be aplied on a single shot record.
Acts on `judiVector{T}` or `Vector{T}`.
"""
abstract type DataPreconditioner{D, R} <: Preconditioner{D, R} end

# JOLI compat
_get_property(J::Preconditioner, ::Val{:fop}) = x -> matvec(J, x)
_get_property(J::Preconditioner, ::Val{:fop_T}) = x -> matvec_T(J, x)
_get_property(J::Preconditioner, ::Val{:name}) = "$(typeof(J))"
_get_property(J::Preconditioner, ::Val{:n}) = getfield(J, :m)
_get_property(J::Preconditioner, ::Val{s}) where {s} = getfield(J, s)

getproperty(J::Preconditioner, s::Symbol) = _get_property(J, Val{s}())

# Base compat
*(J::Preconditioner, ms::judiMultiSourceVector) = matvec(J, ms)
*(J::Preconditioner, ms::PhysicalParameter) = matvec(J, ms)
*(J::Preconditioner, v::VecOrMat{T}) where T = matvec(J, v)
*(J::Preconditioner, v::PyArray{T}) where T = matvec(J, v)

mul!(out::judiMultiSourceVector, J::Preconditioner, ms::judiMultiSourceVector) = copyto!(out, matvec(J, ms))
mul!(out::PhysicalParameter, J::Preconditioner, ms::PhysicalParameter) = copyto!(out, matvec(J, ms))

# OOC judiVector
*(J::DataPreconditioner, v::judiVector{T, SegyIO.SeisCon}) where T = LazyMul(v.nsrc, J, v)
*(J::Preconditioner, v::LazyMul) = LazyMul(v.nsrc, J*v.P, v)

"""
    MultiPrcontinioner{TP, T}

Type for the combination of preconditioners. It is a linear operator that applies the preconditioners in sequence.
"""

struct MultiPreconditioner{TP, T} <: Preconditioner{T, T}
    precs::Vector{TP}
end

function matvec(J::MultiPreconditioner, x)
    y = J.precs[end] * x
    for P in J.precs[1:end-1]
        y = P * y
    end
    return y
end

function matvec_T(J::MultiPreconditioner, x)
    y = J.precs[1]' * x
    for P in J.precs[2:end]
        y = P' * y
    end
    return y
end

conj(I::MultiPreconditioner{TP, T}) where {TP, T} = MultiPreconditioner{TP, T}(conj.(I.precs))
adjoint(I::MultiPreconditioner{TP, T}) where {TP, T} = MultiPreconditioner{TP, T}(adjoint.(reverse(I.precs)))
transpose(I::MultiPreconditioner{TP, T}) where {TP, T} = MultiPreconditioner{TP, T}(transpose.(reverse(I.precs)))

getindex(I::MultiPreconditioner{TP, T}, i) where {TP, T} = MultiPreconditioner{TP, T}([getindex(P, i) for P in I.precs])

for T in [DataPreconditioner, ModelPreconditioner]
    @eval *(P1::$(T){DT}, P2::$(T){DT}) where DT = MultiPreconditioner{$(T), DT}([P1, P2])
    @eval *(P::$(T){DT}, P2::MultiPreconditioner{$(T), DT}) where DT = MultiPreconditioner{$(T), DT}([P, P2.precs...])
    @eval *(P2::MultiPreconditioner{$(T), DT}, P::$(T){DT}) where DT = MultiPreconditioner{$(T), DT}([P2.precs..., P])
end

time_resample(x::MultiPreconditioner{TP, T}, newt) where {TP, T} = MultiPreconditioner{TP, T}([time_resample(P, newt) for P in x.precs])
