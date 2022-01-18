############################################################
# judiPDEextended ##########################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiPDEextended, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections

struct judiPDEextended{D<:Number,R<:Number} <: judiAbstractLinearOperator{D, R}
    m::Integer
    n::Integer
    info::Info
    model::Model
    wavelet::Array
    recGeometry::Geometry
    options::Options
end

############################################################
## Constructor

function judiPDEextended(info::Info,model::Model, wavelet::Array, recGeometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiLinearException("Domain and range types not supported"))

    # Determine dimensions
    m = n_samples(recGeometry, info)
    n = info.n * info.nsrc

    srcnum = 1:info.nsrc

    return judiPDEextended{Float32,Float32}(m, n, info, model, wavelet, recGeometry, options)
end

###########################################################

judi_forward(A::judiPDEextended{ADDT,ARDT},v::vT) where {ADDT,ARDT,vT<:Union{<:judiWeights, judiVector, AbstractVector}} =
    extended_source_modeling(A.model, A.wavelet, A.recGeometry, nothing, process_input_data(v, A.model, A.info), nothing, 1:A.info.nsrc, 'F', 1, A.options)

judi_forward(A::judiPDEextended{ADDT,ARDT}, v::AbstractMatrix{vDT}) where {ADDT,ARDT,vDT} = judi_forward(A, vec(v))

judi_adjoint(A::judiPDEextended{ADDT,ARDT}, v::vT) where {ADDT,ARDT,vT<:Union{<:judiWeights, judiVector, AbstractVector}} =
    extended_source_modeling(A.model, A.wavelet, A.recGeometry, process_input_data(v,A.recGeometry,A.info), nothing, nothing, 1:A.info.nsrc, 'F', -1, A.options)

# *(num,judiPDEextended)
*(a::Number,A::judiPDEextended{ADDT,ARDT}) where {ADDT,ARDT} = judiPDEextended{ADDT,ARDT}(A.m,A.n,A.info,A.model,a*A.wavelet,A.recGeometry,A.options)

############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(F::judiPDEextended{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    recGeometry = subsample(F.recGeometry,srcnum)
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiPDEextended(info, F.model, F.wavelet[srcnum], recGeometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::judiPDEextended,a) = subsample(F,a)
