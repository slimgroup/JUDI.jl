############################################################
# judiJacobian ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiAbstractJacobian, judiJacobian, judiJacobianExQ, subsample

############################################################
abstract type judiAbstractJacobian{D, R} <: judiAbstractLinearOperator{D, R} end

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiJacobian{T, D, R} <: judiAbstractJacobian{D, R}
    m::Integer
    n::Integer
    info::Info
    model::Model
    source::T
    recGeometry::Geometry
    options::Options
    offsets::Union{Integer, Vector{Integer}}
end

const judiJacobianExQ{D, R} = judiJacobian{judiExtendedSource{D}, D, R} where {D, R}
############################################################
## Constructor
"""
    judiJacobian(F,q)

Create a linearized modeling operator from the non-linear modeling operator `F` and \\
the source `q`. `F` is a full modeling operator including source/receiver projections.

Examples
========

1) `F` is a modeling operator without source/receiver projections:

    J = judiJacobian(Pr*F*Ps',q)

2) `F` is the combined operator `Pr*F*Ps'`:

    J = judiJacobian(F,q)
"""
function judiJacobian(F::judiPDEfull, source::sT; DDT::DataType=Float32, RDT::DataType=DDT, options=nothing, offsets=0) where {sT<:judiVector}
# JOLI wrapper for nonlinear forward modeling
    compareGeometry(F.srcGeometry, source.geometry) == true || judlLinearException("Source geometry mismatch")
    (DDT == Float32 && RDT == Float32) || throw(judlLinearException("Domain and range types not supported"))
    m = n_samples(F.recGeometry, F.info)
    n = F.info.n

    isnothing(options) && (options = F.options)
    return judiJacobian{typeof(source), Float32, Float32}(m, n, F.info, F.model, source, F.recGeometry, options, offsets)
end

function judiJacobian(F::judiPDEextended, weights::Union{judiWeights, Array}; DDT::DataType=Float32, RDT::DataType=DDT, options=nothing, offsets=0)
    (DDT == Float32 && RDT == Float32) || throw(judlLinearException("Domain and range types not supported"))
    m = n_samples(F.recGeometry, F.info)
    n = F.info.n

    weights = process_input_data(weights, F.model, F.info)  # extract cell array
    source = judiExtendedSource(F.info, F.wavelet, weights; vDT=DDT)
    isnothing(options) && (options = F.options)
    return judiJacobian{typeof(source), Float32, Float32}(m, n, F.info, F.model, source, F.recGeometry, options, offsets)
end

"""
    judiJacobian(F, q, offsets)

Create a extended linearized modeling operator from the non-linear modeling operator `F` and \\
the source `q`. `F` is a full modeling operator including source/receiver projections.
The offsets are the offset indices on the current model grid. I.e `-1` correspond to the offset
for `u[x - h], v[x+h]`.

Currently only supports  adjoint mode that computes the subsurface offset image gathher for all offsets inputed.
"""
judiJacobian(F, source, offsets::Vector{Integer}; kw...) = judiJacobian(F, source; kw..., offsets=offsets)

judiJacobian(J::judiJacobian{T, DDT,RDT}; m=J.m, n=J.n, info=J.info, model=J.model, source=J.source, geom=J.recGeometry, opt=J.options, offsets=J.offsets) where {T, DDT, RDT} =
    judiJacobian{T, DDT,RDT}(m, n, info, model, source, geom, opt, offsets)

############################################################
## overloaded Base *(...judiJacobian...)
judi_forward(J::judiJacobian{qt,ADDT,ARDT}, v::AbstractVector{Float32}) where {ADDT,ARDT,qt<:judiVector} =
    time_modeling(J.model, J.source.geometry, J.source.data, J.recGeometry, nothing, v, 1:J.info.nsrc, 'J', 1, J.options)

judi_adjoint(J::judiJacobian{qt,ADDT,ARDT}, v::judiVector{vDT, AT}) where {ADDT,ARDT,vDT, AT, qt<:judiVector} =
    time_modeling(J.model, J.source.geometry, J.source.data, J.recGeometry, process_input_data(v, J.recGeometry, J.info), nothing, 1:J.info.nsrc, 'J', -1, J.options)

judi_forward(J::judiJacobianExQ{ADDT,ARDT}, v::AbstractVector{Float32}) where {ADDT,ARDT} =
    extended_source_modeling(J.model, J.source.wavelet, J.recGeometry, nothing, J.source.weights, v, 1:J.info.nsrc, 'J', 1, J.options)

judi_adjoint(J::judiJacobianExQ{ADDT,ARDT}, v::judiVector{vDT, AT}) where {ADDT,ARDT,vDT, AT} =
    extended_source_modeling(J.model, J.source.wavelet, J.recGeometry, process_input_data(v, J.info), J.source.weights, nothing, 1:J.info.nsrc, 'J', -1, J.options)


judi_forward(A::judiAbstractJacobian{ADDT,ARDT}, v::AbstractMatrix{vDT}) where {qT, ADDT,ARDT,vDT} = judi_forward(A, vec(v))
judi_forward(A::judiAbstractJacobian{ADDT,ARDT}, v::AbstractVector{Float64}) where {qT, ADDT,ARDT} = judi_forward(A, jo_convert(Float32, v, false))

# *(num,judiJacobian)
*(a::Number, A::judiAbstractJacobian{ADDT,ARDT}) where {ADDT,ARDT} = judiJacobian(A; source=a*A.source)
# -(judiJacobian)
-(A::judiAbstractJacobian{DDT,RDT}) where {DDT,RDT} = judiJacobian(A; source=-A.source)

############################################################
## Additional overloaded functions

# Subsample Jacobian
function subsample(J::judiJacobian{ADDT,ARDT}, srcnum) where {ADDT, ARDT}
    recGeometry = subsample(J.recGeometry, srcnum)
    nsrc = typeof(srcnum) <: Int ? 1 : length(srcnum)
    info = Info(J.info.n, nsrc, J.info.nt[srcnum])
    m = n_samples(recGeometry, info)
    return judiJacobian(J; m=m, info=info, source=J.source[srcnum], geom=recGeometry, opt=subsample(J.options, srcnum))
end

getindex(J::judiAbstractJacobian, a) = subsample(J, a)

############################################################
## Backward compat

bornop(J::judiAbstractJacobian, v) = judi_forward(J, v)
adjbornop(J::judiAbstractJacobian, w) = judi_adjoint(J, w)