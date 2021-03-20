############################################################
# judiJacobian ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiJacobian, judiJacobianException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiJacobian{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    source::judiVector
    recGeometry::Geometry
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end


mutable struct judiJacobianException <: Exception
    msg :: String
end

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
function judiJacobian(F::judiPDEfull, source::judiVector; DDT::DataType=Float32, RDT::DataType=DDT, options=nothing)
# JOLI wrapper for nonlinear forward modeling
    compareGeometry(F.srcGeometry, source.geometry) == true || judiJacobianException("Source geometry mismatch")
    (DDT == Float32 && RDT == Float32) || throw(judiJacobianException("Domain and range types not supported"))
    m = typeof(F.recGeometry) == GeometryOOC ? sum(F.recGeometry.nsamples) : sum([length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] for j=1:source.nsrc])
    n = F.info.n

    isnothing(options) && (options = F.options)
    return J = judiJacobian{Float32,Float32}("linearized wave equation", m, n, 
        F.info, F.model, source, F.recGeometry, options, bornop, adjbornop)
end

############################################################
## overloaded Base functions

# conj(judiJacobian)
conj(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.source,A.recGeometry,A.options,
        A.fop,
        A.fop_T
        )

# transpose(judiJacobian)
transpose(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.source,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(judiJacobian)
adjoint(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.source,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

############################################################
## overloaded Base *(...judiJacobian...)

# *(judiJacobian,vec)
function *(A::judiJacobian{ADDT,ARDT}, v::AbstractVector{Float32}) where {ADDT,ARDT}
    A.n == size(v,1) || throw(judiJacobianException("shape mismatch"))
    V = A.fop(A, v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobian,vec):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

*(A::judiJacobian{ADDT,ARDT}, v::AbstractMatrix{vDT}) where {ADDT,ARDT,vDT} = *(A, vec(v))
*(A::judiJacobian{ADDT,ARDT}, v::AbstractVector{Float64}) where {ADDT,ARDT} = *(A, jo_convert(Float32, v, false))

# *(judiJacobian,judiVector)
function *(A::judiJacobian{ADDT,ARDT}, v::judiVector{vDT, AT}) where {ADDT,ARDT,vDT, AT}
    A.n == size(v,1) || throw(judiJacobianException("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiJacobian,judiVector):",A.name,typeof(A),vDT]," / "))
    compareGeometry(A.recGeometry,v.geometry) == true || throw(judiJacobianException("Geometry mismatch"))
    V = A.fop(A, v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobian,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(num,judiJacobian)
*(a::Number,A::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT} =  judiJacobian{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,a*A.source,A.recGeometry,A.options, A.fop, A.fop_T)

############################################################
## overloaded Bases +(...judiJacobian...), -(...judiJacobian...)

# +(judiJacobian,num)
function +(A::judiJacobian{ADDT,ARDT}, b::Number) where {ADDT,ARDT}
    return judiJacobian{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.source,A.recGeometry,A.options,
                            v1 -> A.fop(A, v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(A, v2)+joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobian,num)
function -(A::judiJacobian{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiJacobian{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.source,A.recGeometry,A.options,
                            v1 -> A.fop(A, v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(A, v2)-joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobian)
-(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,-A.source,A.recGeometry,A.options, A.fop, A.fop_T)

############################################################
## Additional overloaded functions

# Subsample Jacobian
function subsample(J::judiJacobian{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    recGeometry = subsample(J.recGeometry, srcnum)
    nsrc = typeof(srcnum) <: Int ? 1 : length(srcnum)
    info = Info(J.info.n, nsrc, J.info.nt)
    m = typeof(recGeometry) == GeometryOOC ? sum(recGeometry.nsamples) : sum([length(recGeometry.xloc[j])*recGeometry.nt[j] for j=1:nsrc])

    return J = judiJacobian{Float32,Float32}("linearized wave equation", m, J.n, 
        info, J.model, J.source[srcnum], recGeometry, subsample(J.options, srcnum), J.fop, J.fop_T)

end

getindex(J::judiJacobian, a) = subsample(J, a)

############################################################
## Forward/adjoint function to avoid unecessary extra declaration

function bornop(J::judiJacobian, v)
    srcnum = 1:J.info.nsrc
    return time_modeling(J.model, J.source.geometry, J.source.data, J.recGeometry, nothing, v, srcnum, 'J', 1, J.options)
end

function adjbornop(J::judiJacobian, w)
    srcnum = 1:J.info.nsrc
    return time_modeling(J.model, J.source.geometry, J.source.data, J.recGeometry,
                         process_input_data(w, J.recGeometry, J.info), nothing, srcnum, 'J', -1, J.options)
end