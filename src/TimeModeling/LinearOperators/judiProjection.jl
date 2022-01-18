############################################################
# judiProjection #############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiProjection

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiProjection{D<:Number,R<:Number} <: judiAbstractLinearOperator{D,R}
    m::Integer
    n::Integer
    info::Info
    geometry::Geometry
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
function judiProjection(info::Info, geometry::Geometry; DDT::DataType=Float32, RDT::DataType=DDT)
    (DDT == Float32 && RDT == Float32) || throw(judiLinearException("Domain and range types not supported"))
    m = n_samples(geometry, info)
    n = info.n * sum(info.nt)

    return judiProjection{Float32,Float32}(m,n,info,geometry)
end

############################################################
## overloaded Base *(...judiProjection...)
judi_adjoint(A::judiProjection{ADDT,ARDT}, v::judiVector{vDT, AT}) where {ADDT,ARDT,vDT, AT} = judiRHS(A.info, v.geometry, v.data)
judi_adjoint(A::judiProjection{ADDT,ARDT}, v::AbstractVector{vDT}) where {ADDT,ARDT,vDT} = judiRHS(A.info, A.geometry, process_input_data(v, A.geometry, A.info))
*(A::judiProjection{D,R}, B::judiModeling{R, R}) where {D, R} = judiPDE(B.info,B.model,A.geometry;options=B.options,DDT=D,RDT=R)
*(A::judiProjection{D,R}, B::jAdjoint{<:judiModeling, D, R}) where {D, R} = adjoint(judiPDE(B.J.info,B.J.model,A.geometry;options=B.J.options,DDT=D,RDT=R))

############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(P::judiProjection{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    geometry = subsample(P.geometry,srcnum)     # Geometry of subsampled data container
    info = Info(P.info.n, length(srcnum), P.info.nt[srcnum])
    return judiProjection(info, geometry;DDT=ADDT,RDT=ARDT)
end

getindex(P::judiProjection,a) = subsample(P,a)
