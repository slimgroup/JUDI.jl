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
*(J::Preconditioner, v::Vector{T}) where T = matvec(J, v)
mul!(out::judiMultiSourceVector, J::Preconditioner, ms::judiMultiSourceVector) = copyto!(out, matvec(J, ms))
mul!(out::PhysicalParameter, J::Preconditioner, ms::PhysicalParameter) = copyto!(out, matvec(J, ms))

# Unsupported OOC
function *(J::DataPreconditioner, v::judiVector{T, SegyIO.SeisCon}) where T
    @warn  "Data preconditionners only support in-core judiVector. Converting (might run out of memory)"
    return J * get_data(v)
end