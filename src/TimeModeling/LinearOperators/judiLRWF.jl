############################################################
# judiLRWF #############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017
# Updated: March 2021, Mathias Louboutin (mlouboutin3@gatech.edu)

export judiLRWF, judiLRWFexception

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiLRWF{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    wavelet
end


mutable struct judiLRWFexception <: Exception
    msg :: String
end


############################################################
## Constructor
"""
    judiLRWF(info, geometry)
Low-rank wavefield operator which injects a wavelet q at every point of the subsurface. \\
`info` is an `Info` structure and `wavelet` is a cell array containing the wavelet(s).
Examples
========
`F` is a modeling operator of type `judiModeling` and `w` is a weighting matrix of type `judiWeights`:
    Pr = judiProjection(info, rec_geometry)
    Pw = judiLRWF(info, q.data)
    dobs = Pr*F*Pw'*w
    dw = Pw*F'*Pr'*dobs
"""
function judiLRWF(info::Info, data::Array{T, N}; DDT::DataType=Float32, RDT::DataType=DDT) where {T, N}
    (DDT == Float32 && RDT == Float32 && T == Float32) || throw(judiProjectionException("Domain and range types not supported"))
    m = info.n * info.nsrc
    n = info.n * sum(info.nt)
    wavelet = Array{Array{DDT, N}, 1}(undef, info.nsrc)
    for j=1:info.nsrc
        wavelet[j] = data
    end
    return judiLRWF{Float32,Float32}("restriction operator",m,n,info,wavelet)
end


function judiLRWF(info::Info, wavelet::Array{Array{T, N}, 1}; DDT::DataType=Float32, RDT::DataType=DDT) where {T, N}
    (DDT == Float32 && RDT == Float32 && T == Float32) || throw(judiProjectionException("Domain and range types not supported"))
    m = info.n * info.nsrc
    n = info.n * sum(info.nt)
    return judiLRWF{Float32,Float32}("restriction operator",m,n,info,wavelet)
end



############################################################
## overloaded Base functions

# conj(judiProjection)
conj(A::judiLRWF{DDT,RDT}) where {DDT,RDT} =
    judiLRWF{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.wavelet)

# transpose(judiProjection)
transpose(A::judiLRWF{DDT,RDT}) where {DDT,RDT} =
    judiLRWF{DDT,RDT}("injection operator",A.n,A.m,A.info,A.wavelet)

adjoint(A::judiLRWF{DDT,RDT}) where {DDT,RDT} =
    judiLRWF{DDT,RDT}("injection operator",A.n,A.m,A.info,A.wavelet)

############################################################
## overloaded Base *(...judiProjection...)

# *(judiLRWF, judiWeights)
function *(A::judiLRWF{ADDT,ARDT}, v::judiWeights{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiLRWFexception("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiLRWF,judiVector):",A.name,typeof(A),vDT]," / "))
    V = judiExtendedSource(A.info,A.wavelet,v.weights)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiLRWF,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(judiLRWF, vec)
function *(A::judiLRWF{ADDT,ARDT}, v::AbstractVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v, 1) || throw(judiLRWFexception("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiLRWF,judiVector):",A.name,typeof(A),vDT]," / "))
    V = judiExtendedSource(A.info,A.wavelet, v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiLRWF,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(P::judiLRWF{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    info = Info(P.info.n, length(srcnum), P.info.nt[srcnum])
    return judiLRWF(info, P.wavelet[srcnum];DDT=ADDT,RDT=ARDT)
end

getindex(P::judiLRWF,a) = subsample(P,a)
