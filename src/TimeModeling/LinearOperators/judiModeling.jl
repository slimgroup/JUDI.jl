############################################################
# judiModeling ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiModeling, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiModeling{D<:Number,R<:Number} <: judiAbstractLinearOperator{D, R}
    m::Integer
    n::Integer
    info::Info
    model::Model
    options::Options
end

############################################################
## outer constructors
"""
    judiModeling(info, model; options=Options())
	judiModeling(model, src_geometry, rec_geometry; options=Options())
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

    F = judiModeling(info, model, q.geometry, rec_geometry)
    dobs = F*q

"""
function judiModeling(info::Info, model::Model; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
	(DDT == Float32 && RDT == Float32) || throw(judiModelingException("Domain and range types not supported"))
	m = info.n * sum(info.nt)
	n = m
	return F = judiModeling{Float32,Float32}(m, n, info, model, options)
end

############################################################
## Additional overloaded functions
*(J::judiModeling, v::judiRHS{vT}) where vT = mul(J, v)
*(J::jAdjoint{<:judiModeling, D, R}, v::judiRHS{vT}) where {D, R, vT} = mul(J, v)

judi_forward(A::judiModeling{D,R}, v::judiWavefield{vDT}) where {D, R, vDT} =
	time_modeling(A.model, nothing, v.data, nothing, nothing, nothing, 1:A.info.nsrc, 'F', 1, A.options)

judi_adjoint(A::judiModeling{D,R}, v::judiWavefield{vDT}) where {D, R, vDT} =
	time_modeling(A.model, nothing, nothing, nothing, v.data, nothing, 1:A.info.nsrc, 'F', -1, A.options)

judi_forward(A::judiModeling{D,R}, v::judiRHS{vDT}) where {D, R, vDT} =
	time_modeling(A.model, v.geometry, v.data, nothing, nothing, nothing, 1:A.info.nsrc, 'F', 1, A.options)

judi_adjoint(A::judiModeling{D,R}, v::judiRHS{vDT}) where {D, R, vDT} =
	time_modeling(A.model, nothing, nothing, v.geometry, v.data, nothing, 1:A.info.nsrc, 'F', -1, A.options)

############################################################
## Additional overloaded functions

# Subsample Modeling function
function subsample(F::judiModeling, srcnum)
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiModeling(info, F.model, options=F.options)
end

getindex(F::judiModeling,a) = subsample(F, a)
