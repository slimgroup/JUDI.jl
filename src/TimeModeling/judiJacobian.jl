############################################################
# judiJacobian ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiJacobian, judiJacobianException, subsample, zerox

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiJacobian{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Modelall
    srcGeometry::Geometry
    recGeometry::Geometry
    source
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
    if typeof(F.recGeometry) == GeometryOOC
        m = sum(F.recGeometry.nsamples)
    else
        m = 0
        for j=1:F.info.nsrc m += length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] end
    end
    n = F.info.n
    if F.info.nsrc > 1
        srcnum = 1:F.info.nsrc
    else
        srcnum = 1
    end
    isnothing(options) && (options = F.options)
    return J = judiJacobian{Float32,Float32}("linearized wave equation", m, n, 
        F.info, F.model, F.srcGeometry, F.recGeometry, source.data, options,
        v -> time_modeling(F.model, F.srcGeometry, source.data, F.recGeometry, nothing, v, srcnum, 'J', 1, options),
        w -> time_modeling(F.model, F.srcGeometry, source.data, F.recGeometry, process_input_data(w, F.recGeometry, F.info),
        nothing, srcnum, 'J', -1, options)
    )
end


############################################################
## overloaded Base functions

# conj(judiJacobian)
conj(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
        A.fop,
        A.fop_T
        )

# transpose(judiJacobian)
transpose(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(judiJacobian)
adjoint(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
        A.fop_T,
        A.fop
        )

############################################################
## overloaded Base *(...judiJacobian...)

# *(judiJacobian,vec)
function *(A::judiJacobian{ADDT,ARDT},v::AbstractVector{Float32}) where {ADDT,ARDT}
    A.n == size(v,1) || throw(judiJacobianException("shape mismatch"))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobian,vec):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

*(A::judiJacobian{ADDT,ARDT},v::AbstractMatrix{vDT}) where {ADDT,ARDT,vDT} = *(A, vec(v))
*(A::judiJacobian{ADDT,ARDT},v::AbstractVector{Float64}) where {ADDT,ARDT} = *(A, jo_convert(Float32, v, false))

# *(judiJacobian,judiVector)
function *(A::judiJacobian{ADDT,ARDT}, v::judiVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiJacobianException("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiJacobian,judiVector):",A.name,typeof(A),vDT]," / "))
    compareGeometry(A.recGeometry,v.geometry) == true || throw(judiJacobianException("Geometry mismatch"))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobian,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(num,judiJacobian)
function *(a::Number,A::judiJacobian{ADDT,ARDT}) where {ADDT,ARDT}
    return judiJacobian{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
                                v1 -> jo_convert(ARDT,a, false)*A.fop(v1),
                                v2 -> jo_convert(ARDT,a, false)*A.fop_T(v2)
                                )
end

function A_mul_B!(x::judiVector, J::judiJacobian, y::Array)
    # check if already adjoint(i.e lsqr)
    J.m == size(y, 1) ? z = adjoint(J)*y : z = J*y
    for j=1:length(x.data)
        x.data[j] .= z.data[j]
    end
end

function A_mul_B!(x::Array, J::judiJacobian, y::judiVector)
    # check if already adjoint(i.e lsqr)
    J.m == size(y, 1) ? x[:] = adjoint(J)*y : x[:] = J*y
end

mul!(x::Array, J::judiJacobian, y::judiVector) = A_mul_B!(x, J, y)
mul!(x::judiVector, J::judiJacobian, y::Array) = A_mul_B!(x, J, y)

# Needed by lsqr
zerox(J::judiJacobian, y::judiVector) = zeros(Float32, size(J, 2))

############################################################
## overloaded Bases +(...judiJacobian...), -(...judiJacobian...)

# +(judiJacobian,num)
function +(A::judiJacobian{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiJacobian{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
                            v1 -> A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(v2)+joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobian,num)
function -(A::judiJacobian{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiJacobian{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
                            v1 -> A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(v2)-joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobian)
-(A::judiJacobian{DDT,RDT}) where {DDT,RDT} =
    judiJacobian{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.srcGeometry,A.recGeometry,A.source,A.options,
                    v1 -> -A.fop(v1),
                    v2 -> -A.fop_T(v2)
                    )

############################################################
## Additional overloaded functions

# Subsample Jacobian
function subsample(J::judiJacobian{ADDT,ARDT}, srcnum) where {ADDT,ARDT}

    srcGeometry = subsample(J.srcGeometry,srcnum)       # Geometry of subsampled data container
    recGeometry = subsample(J.recGeometry,srcnum)

    info = Info(J.info.n, length(srcnum), J.info.nt[srcnum])
    Fsub = judiModeling(info, J.model, srcGeometry, recGeometry; options=J.options)
    qsub = judiVector(srcGeometry, J.source[srcnum])
    return judiJacobian(Fsub, qsub; DDT=ADDT, RDT=ARDT)
end

getindex(J::judiJacobian,a) = subsample(J,a)
