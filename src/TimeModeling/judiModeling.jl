############################################################
# judiModeling ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiModeling, judiModelingException, judiSetupModeling, judiModelingAdjoint, judiModelingAdjointException, judiSetupModelingAdjoint, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps, 
# i.e. it includes source and receiver projections
struct judiModeling{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    options::Options
end

mutable struct judiModelingException <: Exception
    msg :: String
end


struct judiModelingAdjoint{DDT,RDT} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    options::Options
end

mutable struct judiModelingAdjointException <: Exception
    msg :: String
end


############################################################
## outer constructors
"""
    judiModeling(info, model; options=Options())
    judiModeling(info, model, src_geometry, rec_geometry; options=Options())

Create seismic modeling operator for a velocity model given as a `Model` structure. `info` is an `Info` structure\\
containing necessary dimensions to set up the operator. The function also takes the source and receiver geometries\\
as additional input arguments, which creates a combined operator `judiProjection*judiModeling*judiProjection'`.

Example
=======

`Pr` and `Ps` are projection operatos of type `judiProjection` and\\
`q` is a data vector of type `judiVector`:

    F = judiModeling(info, model)
    dobs = Pr*F*Ps'*q

    F = judiModeling(info, model, q.geometry, rec\_geometry)
    dobs = F*q

"""
function judiModeling(info::Info, model::Model; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiModelingException("Domain and range types not supported"))
    m = info.n * sum(info.nt)
    n = m
    if info.nsrc > 1
        srcnum = 1:info.nsrc
    else
        srcnum = 1
    end
    return F = judiModeling{Float32,Float32}("forward wave equation", m, n, info, model, options)
end

function judiModelingAdjoint(info::Info, model::Model; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiModelingAdjointException("Domain and range types not supported"))
    m = info.n * sum(info.nt)
    n = m
    if info.nsrc > 1
        srcnum = 1:info.nsrc
    else
        srcnum = 1
    end
    return F = judiModelingAdjoint{Float32,Float32}("adjoint wave equation", m, n, info, model, options)
end


############################################################
## overloaded Base functions

# conj(jo)
conj{DDT,RDT}(A::judiModeling{DDT,RDT}) =
    judiModeling{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.options)

# transpose(jo)
transpose{DDT,RDT}(A::judiModeling{DDT,RDT}) =
    judiModelingAdjoint{DDT,RDT}("adjoint wave equation",A.n,A.m,A.info,A.model,A.options)

# ctranspose(jo)
ctranspose{DDT,RDT}(A::judiModeling{DDT,RDT}) =
    judiModelingAdjoint{DDT,RDT}("adjoint wave equation",A.n,A.m,A.info,A.model,A.options)

# conj(jo)
conj{DDT,RDT}(A::judiModelingAdjoint{DDT,RDT}) =
    judiModelingAdjoint{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.options)

# transpose(jo)
transpose{DDT,RDT}(A::judiModelingAdjoint{DDT,RDT}) =
    judiModeling{DDT,RDT}("forward wave equation",A.n,A.m,A.info,A.model,A.options)

# ctranspose(jo)
ctranspose{DDT,RDT}(A::judiModelingAdjoint{DDT,RDT}) =
    judiModeling{DDT,RDT}("forward wave equation",A.n,A.m,A.info,A.model,A.options)


############################################################
## Additional overloaded functions

# Subsample Modeling functino
function subsample(F::judiModeling, srcnum)
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiModeling(info, F.model, options=F.options)
end

function subsample(F::judiModelingAdjoint, srcnum)
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiModelingAdjoint(info, F.model, options=F.options)
end

getindex(F::judiModeling,a) = subsample(F,a)
getindex(F::judiModelingAdjoint,a) = subsample(F,a)







