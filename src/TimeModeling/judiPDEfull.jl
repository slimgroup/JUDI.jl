############################################################
# judiPDEfull ################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiPDEfull, judiPDEfullException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections

struct judiPDEfull{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    srcGeometry::Geometry
    recGeometry::Geometry
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end

mutable struct judiPDEfullException <: Exception
    msg :: String
end


############################################################
## Constructor

function judiModeling(info::Info,model::Model, srcGeometry::Geometry, recGeometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiPDEfullException("Domain and range types not supported"))

    # Determine dimensions
    if typeof(recGeometry) == GeometryOOC
        m = sum(recGeometry.nsamples)
    else
        m = 0
        for j=1:info.nsrc m += length(recGeometry.xloc[j])*recGeometry.nt[j] end

    end
    if typeof(srcGeometry) == GeometryOOC
        n = sum(srcGeometry.nsamples)
    else
        n = 0
        for j=1:info.nsrc n += length(srcGeometry.xloc[j])*srcGeometry.nt[j] end
    end

    if info.nsrc > 1
        srcnum = 1:info.nsrc
    else
        srcnum = 1
    end

    return F = judiPDEfull{Float32,Float32}("Proj*F*Proj'", m, n, info, model, srcGeometry, recGeometry, options,
                              src -> time_modeling(model, srcGeometry, src.data, recGeometry, nothing, nothing, srcnum, 'F', 1, options),
                              rec -> time_modeling(model, srcGeometry, nothing, recGeometry, rec.data, nothing, srcnum, 'F', -1, options),
                              )
end


############################################################
## overloaded Base functions

# conj(judiPDEfull)
conj(A::judiPDEfull{DDT,RDT}) where {DDT,RDT} =
    judiPDEfull{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
        A.fop,
        A.fop_T
        )

# transpose(judiPDEfull)
transpose(A::judiPDEfull{DDT,RDT}) where {DDT,RDT} =
    judiPDEfull{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(judiPDEfull)
adjoint(A::judiPDEfull{DDT,RDT}) where {DDT,RDT} =
    judiPDEfull{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

############################################################
## overloaded Base *(...judiPDEfull...)

# *(judiPDEfull,judiVector)
function *(A::judiPDEfull{ADDT,ARDT},v::judiVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiPDEfullException("shape mismatch"))
    if compareGeometry(A.srcGeometry,v.geometry) == false && compareGeometry(A.recGeometry,v.geometry) == false
        throw(judiPDEfullException("geometry mismatch"))
    end
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDEfull,judiVector):",A.name,typeof(A),vDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDEfull,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(num,judiPDEfull)
function *(a::Number,A::judiPDEfull{ADDT,ARDT}) where {ADDT,ARDT}
    return judiPDEfull{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
        v1 -> jo_convert(ARDT,a*A.fop(v1),false),
        v2 -> jo_convert(ADDT,a*A.fop_T(v2),false)
        )
end

function A_mul_B!(x::judiVector,A::judiPDEfull,y::judiVector)
    z = A*y
    x.data = z.data
end

function Ac_mul_B!(x::judiVector,A::judiPDEfull,y::judiVector)
    z = adjoint(A)*y
    x.data = z.data
end

############################################################
## overloaded Bases +(...judiPDEfull...), -(...judiPDEfull...)

# +(judiPDEfull,num)
function +(A::judiPDEfull{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiPDE{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
        v1 -> A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
        v2 -> A.fop_T(v2)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
        )
end

# -(judiPDEfull,num)
function -(A::judiPDEfull{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiPDE{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
        v1 -> A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
        v2 -> A.fop_T(v2)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v2
        )
end

# -(judiPDEfull)
-(A::judiPDEfull{DDT,RDT}) where {DDT,RDT} =
    judiPDEfull{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.options,
        v1 -> -A.fop(v1),
        v2 -> -A.fop_T(v2)
        )


############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(F::judiPDEfull{ADDT,ARDT}, srcnum) where {ADDT,ARDT}

    srcGeometry = subsample(F.srcGeometry,srcnum)       # Geometry of subsampled data container
    recGeometry = subsample(F.recGeometry,srcnum)

    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiModeling(info, F.model, srcGeometry, recGeometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::judiPDEfull,a) = subsample(F,a)
