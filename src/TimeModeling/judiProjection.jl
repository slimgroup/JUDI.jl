############################################################
# judiProjection #############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiProjection, judiProjectionException

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiProjection{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    geometry::Geometry
end


mutable struct judiProjectionException <: Exception
    msg :: String
end


############################################################
## Constructor
"""
    judiProjection(info, geometry)

Projection operator for sources/receivers to restrict or inject data at specified locations.\\
`info` is an `Info` structure and `geometry` is a `Geometry` structure with either source or\\
receiver locations.

Examples
========

`F` is a modeling operator of type `judiModeling` and `q` is a seismic source of type `judiVector`:

    Pr = judiProjection(info, rec_geometry)
    Ps = judiProjection(info, q.geometry)

    dobs = Pr*F*Ps'*q
    qad = Ps*F'*Pr'*dobs

"""
function judiProjection(info::Info, geometry::GeometryIC; DDT::DataType=Float32, RDT::DataType=DDT)
    (DDT == Float32 && RDT == Float32) || throw(judiProjectionException("Domain and range types not supported"))
    m = 0
    for j=1:length(geometry.xloc)
        m += length(geometry.xloc[j])*geometry.nt[j]
    end
    n = info.n * sum(info.nt)

    return judiProjection{Float32,Float32}("restriction operator",m,n,info,geometry)
end

function judiProjection(info::Info, geometry::GeometryOOC; DDT::DataType=Float32, RDT::DataType=DDT)
    (DDT == Float32 && RDT == Float32) || throw(judiProjectionException("Domain and range types not supported"))
    m = sum(geometry.nsamples)
    n = info.n * sum(info.nt)

    return judiProjection{Float32,Float32}("restriction operator",m,n,info,geometry)
end


############################################################
## overloaded Base functions

# conj(judiProjection)
conj(A::judiProjection{DDT,RDT}) where {DDT,RDT} =
    judiProjection{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.geometry)

# transpose(judiProjection)
transpose(A::judiProjection{DDT,RDT}) where {DDT,RDT} =
    judiProjection{DDT,RDT}("injection operator",A.n,A.m,A.info,A.geometry)

adjoint(A::judiProjection{DDT,RDT}) where {DDT,RDT} =
    judiProjection{DDT,RDT}("injection operator",A.n,A.m,A.info,A.geometry)

############################################################
## overloaded Base *(...judiProjection...)

# *(judiProjection,judiVector)
function *(A::judiProjection{ADDT,ARDT},v::judiVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiProjectionException("shape mismatch"))
    compareGeometry(A.geometry,v.geometry) == true || throw(judiProjectionException("geometry mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiProjection,judiVector):",A.name,typeof(A),vDT]," / "))
    V = judiRHS(A.info,v.geometry,v.data)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiProjection,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(judiProjection,judiModeling)
function *(A::judiProjection{CDT,ARDT},B::judiModeling{BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judiProjectionException("shape mismatch"))
    compareInfo(A.info, B.info) == true || throw(judiProjectionException("info mismatch"))
    if typeof(A.geometry) == GeometryOOC
        m = sum(A.geometry.nsamples)
    else
        m = 0; for j=1:B.info.nsrc m+= length(A.geometry.xloc[j])*A.geometry.nt[j] end
    end
    n = B.info.n * sum(B.info.nt)
    return judiPDE("judiProjection*judiModeling",B.info,B.model,A.geometry;options=B.options,DDT=CDT,RDT=ARDT)
end

function *(A::judiProjection{CDT,ARDT},B::judiModelingAdjoint{BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judiProjectionException("shape mismatch"))
    compareInfo(A.info, B.info) == true || throw(judiProjectionException("info mismatch"))
    if typeof(A.geometry) == GeometryOOC
        m = sum(A.geometry.nsamples)
    else
        m = 0; for j=1:B.info.nsrc m+= length(A.geometry.xloc[j])*A.geometry.nt[j] end
    end
    n = B.info.n * sum(B.info.nt)
    return judiPDEadjoint("judiProjection*judiModelingAdjoint",B.info,B.model,A.geometry;options=B.options,DDT=CDT,RDT=ARDT)
end

############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(P::judiProjection{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    geometry = subsample(P.geometry,srcnum)     # Geometry of subsampled data container
    info = Info(P.info.n, length(srcnum), P.info.nt[srcnum])
    return judiProjection(info, geometry;DDT=ADDT,RDT=ARDT)
end

getindex(P::judiProjection,a) = subsample(P,a)
