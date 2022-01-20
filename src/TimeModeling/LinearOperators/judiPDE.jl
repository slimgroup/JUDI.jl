############################################################
# judiPDE ####################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiPDE, judiPDEexception, judiPDEadjoint, judiPDEadjointException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections

# Forward
struct judiPDE{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    geometry::Geometry
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end

mutable struct judiPDEexception <: Exception
    msg :: String
end

# Adjoint
struct judiPDEadjoint{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    geometry::Geometry
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end

mutable struct judiPDEadjointException <: Exception
    msg :: String
end



############################################################
## outer constructors

function judiPDE(name::String,info::Info,model::Model, geometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiPDEexception("Domain and range types not supported"))
    m = n_samples(geometry, info)
    n = info.n * sum(info.nt)
    srcnum = 1:info.nsrc

    return F = judiPDE{Float32,Float32}(name, m, n, info, model, geometry, options,
                              args -> time_modeling(model, args[1], args[2], args[3], args[4], args[5], srcnum, 'F', 1, options),
                              args_T -> time_modeling(model, args_T[1], args_T[2], args_T[3], args_T[4], args_T[5], srcnum, 'F', -1, options)
                              )
end

function judiPDEadjoint(name::String,info::Info,model::Model, geometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiPDEadjointException("Domain and range types not supported"))
    m = n_samples(geometry, info)
    n = info.n * sum(info.nt)

    srcnum = 1:info.nsrc

    return F = judiPDEadjoint{Float32,Float32}(name, m, n, info, model, geometry, options,
                                args_T -> time_modeling(model, args_T[1], args_T[2], args_T[3], args_T[4], args_T[5], srcnum, 'F', -1, options),
                                args -> time_modeling(model, args[1], args[2], args[3], args[4], args[5], srcnum, 'F', 1, options),
                                )
end


############################################################
## overloaded Base functions

# conj(judiPDE)
conj(A::judiPDE{DDT,RDT}) where {DDT,RDT} =
    judiPDE{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
        A.fop,
        A.fop_T
        )

# transpose(jo)
transpose(A::judiPDE{DDT,RDT}) where {DDT,RDT} =
    judiPDEadjoint{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(jo)
adjoint(A::judiPDE{DDT,RDT}) where {DDT,RDT} =
	judiPDEadjoint{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
		A.fop_T,
		A.fop
		)

# conj(jo)
conj(A::judiPDEadjoint{DDT,RDT}) where {DDT,RDT}=
    judiPDEadjoint{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
        A.fop,
        A.fop_T
        )

# transpose(jo)
transpose(A::judiPDEadjoint{DDT,RDT}) where {DDT,RDT} =
    judiPDE{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
        A.fop_T,
        A.fop
        )
# adjoint(jo)
adjoint(A::judiPDEadjoint{DDT,RDT}) where {DDT,RDT} =
    judiPDE{DDT,RDT}(A.name,A.n,A.m,A.info,A.model,A.geometry,A.options,
        A.fop_T,
        A.fop
        )


############################################################
## overloaded Base *(...judi...)

# *(judiPDE,judiRHS)
function *(A::judiPDE{ADDT,ARDT},v::judiRHS{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiPDEexception("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDE,judiRHS):",A.name,typeof(A),vDT]," / "))
	args = (v.geometry, v.data, A.geometry, nothing, nothing)
    V = A.fop(args)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDE,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

function *(A::judiPDEadjoint{ADDT,ARDT},v::judiRHS{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiPDEadjointException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDE,judiRHS):",A.name,typeof(A),vDT]," / "))
	args = (A.geometry,nothing,v.geometry,v.data,nothing)
    V = A.fop(args)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDE,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(judiPDE,judiWavefield)
function *(A::judiPDE{ADDT,ARDT},v::judiWavefield{vDT}) where {ADDT,ARDT,vDT}
	A.n == size(v,1) || throw(judiPDEexception("Shape mismatch: A:$(size(A)), v: $(size(v))"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDE,judiWavefield):",A.name,typeof(A),vDT]," / "))
	args = (nothing,v.data,A.geometry,nothing,nothing)
	V = A.fop(args)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDE,judiVector):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

function *(A::judiPDEadjoint{ADDT,ARDT},v::judiWavefield{vDT}) where {ADDT,ARDT,vDT}
	A.n == size(v,1) || throw(judiPDEadjointException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
	jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDE,judiWavefield):",A.name,typeof(A),vDT]," / "))
	args = (A.geometry,nothing,nothing,v.data,nothing)
	V = A.fop(args)
	jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDE,judiVector):",A.name,typeof(A),eltype(V)]," / "))
	return V
end

# *(judiPDE,judiProjection)
function *(A::judiPDE{CDT,ARDT},B::judiProjection{BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judiPDEexception("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return judiModeling(A.info,A.model,B.geometry,A.geometry;options=A.options,DDT=CDT,RDT=ARDT)
end

function *(A::judiPDEadjoint{CDT,ARDT},B::judiProjection{BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judiPDEadjointException("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return adjoint(judiModeling(A.info,A.model,A.geometry,B.geometry,options=A.options,DDT=CDT,RDT=ARDT))
end

# *(judiPDE,judiLRWF)
function *(A::judiPDE{CDT,ARDT},B::judiLRWF{BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judiPDEexception("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return judiPDEextended(A.info,A.model,B.wavelet,A.geometry;options=A.options,DDT=CDT,RDT=ARDT)
end

function *(A::judiPDEadjoint{CDT,ARDT},B::judiLRWF{BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judiPDEadjointException("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return adjoint(judiPDEextended(A.info,A.model,B.wavelet,A.geometry,options=A.options,DDT=CDT,RDT=ARDT))
end

#multiplications with judiExtendedSource
# *(judiPDE,judiExtendedSource)
function *(A::judiPDE{ADDT,ARDT}, B::judiExtendedSource{vDT}) where {ADDT, ARDT,vDT}
    A.n == size(B,1) || throw(judiPDEexception("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    jo_check_type_match(ADDT, vDT, join(["DDT for *(judiPDE,judiExtendedSource):",A.name,typeof(A),vDT]," / "))
    F = judiPDEextended(A.info, A.model, B.wavelet, A.geometry; options=A.options,DDT=vDT, RDT=ARDT)
    V = F*B.weights
    jo_check_type_match(ARDT, eltype(V),join(["RDT from *(judiPDE, judiExtendedSource):",A.name,typeof(A), eltype(V)]," / "))
    return V
end

# *(judiPDEadjoint, judiExtendedSource)
function *(A::judiPDEadjoint{ADDT,ARDT}, B::judiExtendedSource{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(B, 1) || throw(judiPDEadjointException("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    jo_check_type_match(ADDT, vDT, join(["DDT for *(judiPDEadjoint,judiExtendedSource):",A.name,typeof(A),vDT]," / "))
    F = adjoint(judiPDEextended(A.info, A.model, B.wavelet, A.geometry; options=A.options,DDT=vDT,RDT=ARDT))
    V = F*B.weights
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDE,judiExtendedSource):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(num,judiPDE)
function *(a::Number,A::judiPDE{ADDT,ARDT}) where {ADDT,ARDT}
    return judiPDE{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
        v1 -> jo_convert(ARDT,a, false)*A.fop(v1),
        v2 -> jo_convert(ARDT,a, false)*A.fop_T(v2)
        )
end

function *(a::Number,A::judiPDEadjoint{ADDT,ARDT}) where {ADDT,ARDT}
    return judiPDEadjoint{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
        v1 -> jo_convert(ARDT,a, false)*A.fop(v1),
        v2 -> jo_convert(ADDT,a, false)*A.fop_T(v2)
        )
end


############################################################
## overloaded Basees +(...judiPDE...), -(...judiPDE...)

# -(judiPDE)
-(A::judiPDE{DDT,RDT}) where {DDT,RDT} =
    judiPDE{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
        v1->-A.fop(v1),
        v2->-A.fop_T(v2)
        )

-(A::judiPDEadjoint{DDT,RDT}) where {DDT,RDT} =
    judiPDEadjoint{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.geometry,A.options,
        v1->-A.fop(v1),
        v2->-A.fop_T(v2)
        )


############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(F::judiPDE{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    geometry = subsample(F.geometry,srcnum)     # Geometry of subsampled data container
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiPDE(F.name, info, F.model, geometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

function subsample(F::judiPDEadjoint{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    geometry = subsample(F.geometry,srcnum)     # Geometry of subsampled data container
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiPDEadjoint(F.name, info, F.model, geometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::judiPDE,a) = subsample(F,a)
getindex(F::judiPDEadjoint,a) = subsample(F,a)
