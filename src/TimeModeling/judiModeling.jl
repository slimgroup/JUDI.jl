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
	fop::Function              # forward
	fop_T::Function  # transpose
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
	fop::Function              # forward
	fop_T::Function  # transpose
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

    F = judiModeling(info, model, q.geometry, rec_geometry)
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
	return F = judiModeling{Float32,Float32}("forward wave equation", m, n, info, model, options,
							  args -> time_modeling(model, args[1], args[2], args[3], args[4], args[5], srcnum, 'F', 1, options),
							  args_T -> time_modeling(model, args_T[1], args_T[2], args_T[3], args_T[4], args_T[5], srcnum, 'F', -1, options)
							  )
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
	return F = judiModelingAdjoint{Float32,Float32}("adjoint wave equation", m, n, info, model, options,
							  args_T -> time_modeling(model, args_T[1], args_T[2], args_T[3], args_T[4], args_T[5], srcnum, 'F', -1, options),
							  args -> time_modeling(model, args[1], args[2], args[3], args[4], args[5], srcnum, 'F', 1, options)
							  )
end




############################################################
## overloaded Base functions

# conj(jo)
conj(A::judiModeling{DDT,RDT}) where {DDT,RDT} =
    judiModeling{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.options,A.fop, A.fop_T)

# transpose(jo)
transpose(A::judiModeling{DDT,RDT}) where {DDT,RDT} =
    judiModelingAdjoint{DDT,RDT}("adjoint wave equation",A.n,A.m,A.info,A.model,A.options,A.fop_T, A.fop)

adjoint(A::judiModeling{DDT,RDT}) where {DDT,RDT} =
	judiModelingAdjoint{DDT,RDT}("adjoint wave equation",A.n,A.m,A.info,A.model,A.options,A.fop_T, A.fop)

# conj(jo)
conj(A::judiModelingAdjoint{DDT,RDT}) where {DDT,RDT} =
    judiModelingAdjoint{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.options,A.fop, A.fop_T)

# transpose(jo)
transpose(A::judiModelingAdjoint{DDT,RDT}) where {DDT,RDT} =
    judiModeling{DDT,RDT}("forward wave equation",A.n,A.m,A.info,A.model,A.options,A.fop_T, A.fop)

adjoint(A::judiModelingAdjoint{DDT,RDT}) where {DDT,RDT} =
	judiModeling{DDT,RDT}("forward wave equation",A.n,A.m,A.info,A.model,A.options,A.fop_T, A.fop)

############################################################
## Additional overloaded functions

# *(judiModelig,judiWavefield)
function *(A::judiModeling{ADDT,ARDT},v::judiWavefield{vDT}) where {ADDT,ARDT,vDT}
	A.n == size(v,1) || throw(judiModelingException("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(judiModeling,judiWavefield):",A.name,typeof(A),vDT]," / "))
	args = (nothing,v.data,nothing,nothing,nothing)
	V = A.fop(args)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiModeling,judiWavefield):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(judiModeligAdjoint,judiWavefield)
function *(A::judiModelingAdjoint{ADDT,ARDT},v::judiWavefield{vDT}) where {ADDT,ARDT,vDT}
	A.n == size(v,1) || throw(judiModelingAdjointException("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(judiModeling,judiWavefield):",A.name,typeof(A),vDT]," / "))
	args = (nothing,nothing,nothing,v.data,nothing)
	V = A.fop(args)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiModeling,judiWavefield):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(judiModelig,judiRHS)
function *(A::judiModeling{ADDT,ARDT},v::judiRHS{vDT}) where {ADDT,ARDT,vDT}
	A.n == size(v,1) || throw(judiModelingException("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(judiModeling,judiRHS):",A.name,typeof(A),vDT]," / "))
	args = (v.geometry,v.data,nothing,nothing,nothing)
	V = A.fop(args)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiModeling,judiRHS):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(judiModeligAdjoint,judiRHS)
function *(A::judiModelingAdjoint{ADDT,ARDT},v::judiRHS{vDT}) where {ADDT,ARDT,vDT}
	A.n == size(v,1) || throw(judiModelingAdjointException("shape mismatch"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(judiModeling,judiRHS):",A.name,typeof(A),vDT]," / "))
	args = (nothing,nothing,v.geometry,v.data,nothing)
	V = A.fop(args)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiModeling,judiRHS):",A.name,typeof(A),eltype(V)]," / "))
	return V
end


############################################################
## Additional overloaded functions

# Subsample Modeling function
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
