############################################################
# judiPDEextended ##########################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiPDEextended, judiPDEextendedException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections

struct judiPDEextended{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    wavelet::Array
    recGeometry::Geometry
    options::Options
    fop::Function              # forward
    fop_T::Union{Function, Nothing}  # transpose
end

mutable struct judiPDEextendedException <: Exception
    msg :: String
end


############################################################
## Constructor

function judiPDEextended(info::Info,model::Model, wavelet::Array, recGeometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiPDEextendedException("Domain and range types not supported"))

    # Determine dimensions
    if typeof(recGeometry) == GeometryOOC
        m = sum(recGeometry.nsamples)
    else
        m = 0
        for j=1:info.nsrc m += length(recGeometry.xloc[j])*recGeometry.nt[j] end

    end
    n = info.n * info.nsrc

    if info.nsrc > 1
        srcnum = 1:info.nsrc
    else
        srcnum = 1
    end

    return F = judiPDEextended{Float32,Float32}("Proj*F*Proj'", m, n, info, model, wavelet, recGeometry, options,
                              w -> time_modeling(model, nothing, wavelet, recGeometry, nothing, w.weights, srcnum, 'F', 1, options),
                              rec -> time_modeling(model, nothing, wavelet, recGeometry, rec.data, nothing, srcnum, 'F', -1, options),
                              )
end


############################################################
## overloaded Base functions

# conj(judiPDEextended)
conj(A::judiPDEextended{DDT,RDT}) where {DDT,RDT} =
    judiPDEextended{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.wavelet,A.recGeometry,A.options,
        A.fop,
        A.fop_T
        )

# transpose(judiPDEextended)
transpose(A::judiPDEextended{DDT,RDT}) where {DDT,RDT} =
    judiPDEextended{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.wavelet,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(judiPDEextended)
adjoint(A::judiPDEextended{DDT,RDT}) where {DDT,RDT} =
    judiPDEextended{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.wavelet,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

############################################################
## overloaded Base *(...judiPDEextended...)

# *(judiPDEextended,judiWeights)
function *(A::judiPDEextended{ADDT,ARDT},v::judiWeights{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiPDEextendedException("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDEextended,judiWeights):",A.name,typeof(A),vDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDEextended,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(judiPDEextended,judiVector)
function *(A::judiPDEextended{ADDT,ARDT},v::judiVector{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiPDEextendedException("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDEextended,judiWeights):",A.name,typeof(A),vDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDEextended,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(num,judiPDEextended)
function *(a::Number,A::judiPDEextended{ADDT,ARDT}) where {ADDT,ARDT}
    return judiPDEextended{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.wavelet,A.recGeometry,A.options,
        v1 -> jo_convert(ARDT,a*A.fop(v1),false),
        v2 -> jo_convert(ADDT,a*A.fop_T(v2),false)
        )
end

############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(F::judiPDEextended{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    recGeometry = subsample(F.recGeometry,srcnum)
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiPDEextended(info, F.model, F.wavelet[srcnum], recGeometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::judiPDEextended,a) = subsample(F,a)
