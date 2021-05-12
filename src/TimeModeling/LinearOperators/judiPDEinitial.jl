############################################################
# judiPDEinitial ##########################################
############################################################

# Authors: rafael orozco (rorozco@gatech.edu)
# Date: April 2021

export judiPDEinitial, judiPDEinitialException, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Pi*iv,
# i.e. it includes receiver projections

struct judiPDEinitial{DDT<:Number,RDT<:Number} <: joAbstractLinearOperator{DDT,RDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    model::Model
    recGeometry::Geometry
    options::Options
    fop::Function                    # forward
    fop_T::Union{Function, Nothing}  # transpose
end

mutable struct judiPDEinitialException <: Exception
    msg :: String
end

############################################################
## Constructor

function judiPDEinitial(info::Info, model::Model, recGeometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiPDEinitialException("Domain and range types not supported"))

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

    return F = judiPDEinitial{Float32,Float32}("Proj*F*iv", m, n, info, model, recGeometry, options,
        w ->   initial_value_modeling(model, recGeometry, nothing,
                                       process_input_data(w,model,info), nothing, srcnum, 'F', 1, options),
        rec -> initial_value_modeling(model, recGeometry, process_input_data(rec,recGeometry,info),
                                       nothing,                          nothing, srcnum, 'F',-1, options),
        )
end
  

############################################################
## overloaded Base functions

# conj(judiPDEinitial)
conj(A::judiPDEinitial{DDT,RDT}) where {DDT,RDT} =
    judiPDEinitial{DDT,RDT}("conj("*A.name*")",A.m,A.n,A.info,A.model,A.recGeometry,A.options,
        A.fop,
        A.fop_T
        )

# transpose(judiPDEinitial)
transpose(A::judiPDEinitial{DDT,RDT}) where {DDT,RDT} =
    judiPDEinitial{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

# adjoint(judiPDEinitial)
adjoint(A::judiPDEinitial{DDT,RDT}) where {DDT,RDT} =
    judiPDEinitial{DDT,RDT}("Proj*F'*Proj'",A.n,A.m,A.info,A.model,A.recGeometry,A.options,
        A.fop_T,
        A.fop
        )

############################################################
## overloaded Base *(...judiPDEinitial...)

#*(judiPDEinitial,judiInitialValue)
function *(A::judiPDEinitial{ADDT,ARDT},v::judiInitialValue{vDT}) where {ADDT,ARDT,vDT}
    A.n == size(v,1) || throw(judiPDEinitialException("shape mismatch"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDEinitial,judiInitialValue):",A.name,typeof(A),vDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDEinitial,judiInitialValue):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(judiPDEinitial,judiVector)
function *(A::judiPDEinitial{ADDT,ARDT},v::judiVector{vDT, AT}) where {ADDT,ARDT,vDT, AT}
    A.n == size(v,1) || throw(judiPDEinitialException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,vDT,join(["DDT for *(judiPDEinitial,judiWeights):",A.name,typeof(A),vDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(judiPDEinitial,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    return V
end


*(A::judiPDEinitial{ADDT,ARDT}, v::AbstractMatrix{vDT}) where {ADDT,ARDT,vDT} = *(A, vec(v))

# *(num,judiPDEinitial)
function *(a::Number,A::judiPDEinitial{ADDT,ARDT}) where {ADDT,ARDT}
    return judiPDEinitial{ADDT,ARDT}("(N*"*A.name*")",A.m,A.n,A.info,A.model,A.wavelet,A.recGeometry,A.options,
        v1 -> jo_convert(ARDT,a, false) * A.fop(v1),
        v2 -> jo_convert(ARDT,a, false) * A.fop_T(v2)
        )
end

############################################################
