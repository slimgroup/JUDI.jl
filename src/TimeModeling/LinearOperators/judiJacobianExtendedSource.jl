############################################################
# judiJacobianExQ ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiJacobianExQ, judiJacobianExQException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiJacobianExQ{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    recGeometry::Geometry
    wavelet
    weights
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end


mutable struct judiJacobianExQException <: Exception
    msg :: String
end

############################################################
## Constructor
"""
    judiJacobianExQ(F,q)
Create a linearized modeling operator from the non-linear modeling operator `F` and \\
the source `q`. `F` is a full modeling operator including source/receiver projections.
Examples
========
1) `F` is a modeling operator without source/receiver projections:
    J = judiJacobianExQ(Pr*F*Ps',q)
2) `F` is the combined operator `Pr*F*Ps'`:
    J = judiJacobianExQ(F,q)
"""
function judiJacobian(F::judiPDEextended, weights::Union{judiWeights, Array}; DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling w/ extended source

    (DDT == Float32 && RDT == Float32) || throw(judiJacobianExQException("Domain and range types not supported"))
    if typeof(F.recGeometry) == GeometryOOC
        m = sum(F.recGeometry.nsamples)
    else
        m = 0
        for j=1:F.info.nsrc m += length(F.recGeometry.xloc[j])*F.recGeometry.nt[j] end
    end
    n = F.info.n
    srcnum = 1:F.info.nsrc

    weights = process_input_data(weights, F.model, F.info)  # extract cell array

    return J = judiJacobianExQ{Float32,Float32}("linearized wave equation", m, n, F.info, F.model, F.recGeometry, F.wavelet, weights, F.options,
        v -> extended_source_modeling(F.model, F.wavelet, F.recGeometry, nothing, weights, v, srcnum, 'J', 1, F.options),
        w -> extended_source_modeling(F.model, F.wavelet, F.recGeometry, process_input_data(w, F.recGeometry, F.info),
                                      weights, nothing, srcnum, 'J', -1, F.options)
    )
end


############################################################
## overloaded Base functions

# conj(judiJacobianExQ)
conj(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
        A.fop,
        A.fop_T
        )

# transpose(judiJacobianExQ)
transpose(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(judiJacobianExQ)
adjoint(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("adjoint linearized wave equation",A.n,A.m,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
        A.fop_T,
        A.fop
        )

############################################################
## overloaded Base *(...judiJacobianExQ...)

# *(judiJacobianExQ,vec)
function *(A::judiJacobianExQ{ADDT,ARDT},v::AbstractVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiJacobianExQException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiJacobianExQ,vec):",A.name,typeof(A),vDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobianExQ,vec):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

*(A::judiJacobianExQ{ADDT,ARDT},v::AbstractMatrix{vDT}) where {ADDT,ARDT,vDT} = *(A, vec(v))

# *(judiJacobianExQ,judiVector)
function *(A::judiJacobianExQ{ADDT,ARDT},v::judiVector{vDT, AT}) where {ADDT,ARDT,vDT, AT}
    A.n == size(v,1) || throw(judiJacobianExQException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiJacobianExQ,judiVector):",A.name,typeof(A),vDT]," / "))
    compareGeometry(A.recGeometry,v.geometry) == true || throw(judiJacobianExQException("Geometry mismatch"))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiJacobianExQ,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(num,judiJacobianExQ)
function *(a::Number,A::judiJacobianExQ{ADDT,ARDT}) where {ADDT,ARDT}
    return judiJacobianExQ{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                                v1 -> jo_convert(ARDT,a, false)*A.fop(v1),
                                v2 -> jo_convert(ADDT,a, false)*A.fop_T(v2)
                                )
end


############################################################
## overloaded Bases +(...judiJacobianExQ...), -(...judiJacobianExQ...)

# +(judiJacobianExQ,num)
function +(A::judiJacobianExQ{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiJacobianExQ{ADDT,ARDT}("("*A.name*"+N)",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                            v1 -> A.fop(v1)+joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(v2)+joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobianExQ,num)
function -(A::judiJacobianExQ{ADDT,ARDT},b::Number) where {ADDT,ARDT}
    return judiJacobianExQ{ADDT,ARDT}("("*A.name*"-N)",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                            v1 -> A.fop(v1)-joConstants(A.m,A.n,b;DDT=ADDT,RDT=ARDT)*v1,
                            v2 -> A.fop_T(v2)-joConstants(A.n,A.m,b;DDT=ADDT,RDT=ARDT)*v2
                            )
end

# -(judiJacobianExQ)
-(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} =
    judiJacobianExQ{DDT,RDT}("(-"*A.name*")",A.m,A.n,A.info,A.model,A.recGeometry,A.wavelet,A.weights,A.options,
                    v1 -> -A.fop(v1),
                    v2 -> -A.fop_T(v2)
                    )

############################################################
## Additional overloaded functions

# Subsample Jacobian
function subsample(J::judiJacobianExQ{ADDT,ARDT}, srcnum) where {ADDT,ARDT}

    recGeometry = subsample(J.recGeometry,srcnum)
    info = Info(J.info.n, length(srcnum), J.info.nt[srcnum])
    Fsub = judiModeling(info, J.model; options=J.options)
    Pr = judiProjection(info, recGeometry)
    Pw = judiLRWF(info, J.wavelet[srcnum])
    wsub = judiWeights(J.weights[srcnum])
    return judiJacobian(Pr*Fsub*Pw', wsub; DDT=ADDT, RDT=ARDT)
end

getindex(J::judiJacobianExQ,a) = subsample(J, a)
