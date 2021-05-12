############################################################
# judiInitial #############################################
############################################################

# Authors: Rafael Orozco
# Date: April 2021

export judiInitial, judiInitialexception

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Piv',
# i.e. it includes source and receiver projections
struct judiInitial{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
end


mutable struct judiInitialexception <: Exception
    msg :: String
end


############################################################
## Constructor
"""
    judiInitial(info, geometry)

Examples
========

"""

function judiInitial(info::Info; DDT::DataType=Float32, RDT::DataType=DDT)
    (DDT == Float32 && RDT == Float32) || throw(judiInitialException("Domain and range types not supported"))
    m = info.n * 1#info.nsrc should only be one source
    n = info.n * sum(info.nt)
    return judiInitial{Float32,Float32}("restriction operator",m,n,info)
end



############################################################
## overloaded Base functions

# conj(judiProjection)
conj(A::judiInitial{DDT,RDT}) where {DDT,RDT} =
    judiInitial{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info)

# transpose(judiProjection)
transpose(A::judiInitial{DDT,RDT}) where {DDT,RDT} =
    judiInitial{DDT,RDT}("injection operator",A.n,A.m,A.info)

adjoint(A::judiInitial{DDT,RDT}) where {DDT,RDT} =
    judiInitial{DDT,RDT}("injection operator",A.n,A.m,A.info)

############################################################
## overloaded Base *(...judiProjection...)

# *(judiInitial, judiWeights)
function *(A::judiInitial{ADDT,ARDT}, v::judiWeights{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiLRWFexception("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiInitial,judiVector):",A.name,typeof(A),vDT]," / "))
    V = judiSpatialSource(A.info,v.weights)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiInitial,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(judiSpatial, vec)
# function *(A::judiSpatial{ADDT,ARDT}, v::AbstractVector{vDT}) where {ADDT,ARDT,vDT}
#     A.n == size(v, 1) || throw(judiLRWFexception("shape mismatch"))
#     jo_check_type_match(ADDT,vDT,join(["DDT for *(judiSpatial,judiVector):",A.name,typeof(A),vDT]," / "))
#     V = judiExtendedSource(A.info,A.wavelet, v)
#     jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiSpatial,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
#     return V
# end

############################################################