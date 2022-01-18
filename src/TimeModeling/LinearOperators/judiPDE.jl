############################################################
# judiPDE ####################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiPDE, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections

# Forward
struct judiPDE{D<:Number,R<:Number} <: judiAbstractLinearOperator{D, R}
    m::Integer
    n::Integer
    info::Info
    model::Model
    geometry::Geometry
    options::Options
end

############################################################
## outer constructors

function judiPDE(info::Info,model::Model, geometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judlLinearException("Domain and range types not supported"))
    m = n_samples(geometry, info)
    n = info.n * sum(info.nt)

    return judiPDE{Float32,Float32}(m, n, info, model, geometry, options)
end

############################################################
## overloaded Base *(...judi...)

judi_forward(A::judiPDE{D,R}, v::judiWavefield{vDT}) where {D, R, vDT} =
	time_modeling(A.model, nothing, v.data, A.geometry, nothing, nothing, 1:A.info.nsrc, 'F', 1, A.options)

judi_adjoint(A::judiPDE{D,R}, v::judiWavefield{vDT}) where {D, R, vDT} =
	time_modeling(A.model, A.geometry, nothing, nothing, v.data, nothing, 1:A.info.nsrc, 'F', 1, A.options)

judi_forward(A::judiPDE{D,R}, v::judiRHS{vDT}) where {D, R, vDT} =
	time_modeling(A.model, v.geometry, v.data, A.geometry, nothing, nothing, 1:A.info.nsrc, 'F', 1, A.options)

judi_adjoint(A::judiPDE{D,R}, v::judiRHS{vDT}) where {D, R, vDT} =
	time_modeling(A.model, A.geometry, nothing, v.geometry, v.data, nothing, 1:A.info.nsrc, 'F', 1, A.options)


# *(judiPDE,judiProjection)    
function *(A::judiPDE{CDT,ARDT},B::jAdjoint{<:judiProjection, BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judiLinearException("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return judiModeling(A.info,A.model,B.geometry,A.geometry;options=A.options,DDT=CDT,RDT=ARDT)
end

function *(J::jAdjoint{<:judiPDE, CDT,ARDT}, B::jAdjoint{<:judiProjection, BDDT,CDT}) where {ARDT,BDDT,CDT}
    A = J.J
    A.n == size(B,1) || throw(judiLinearException("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return adjoint(judiModeling(A.info,A.model,A.geometry,B.geometry,options=A.options,DDT=CDT,RDT=ARDT))
end

# *(judiPDE,judiLRWF)
function *(A::judiPDE{CDT,ARDT},B::jAdjoint{<:judiLRWF, BDDT,CDT}) where {ARDT,BDDT,CDT}
    A.n == size(B,1) || throw(judlLinearException("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return judiPDEextended(A.info,A.model,B.wavelet,A.geometry;options=A.options,DDT=CDT,RDT=ARDT)
end

function *(J::jAdjoint{<:judiPDE, CDT,ARDT},B::jAdjoint{<:judiLRWF, BDDT,CDT}) where {ARDT,BDDT,CDT}
    A = J.J
    A.n == size(B,1) || throw(judiLinearException("Shape mismatch: A:$(size(A)), B: $(size(B))"))
    return adjoint(judiPDEextended(A.info,A.model,B.wavelet,A.geometry,options=A.options,DDT=CDT,RDT=ARDT))
end

*(J::judiPDE, R::judiRHS) = mul(J, R)
*(J::judiPDE, v::judiExtendedSource) = mul(J, v)

#multiplications with judiExtendedSource
# *(judiPDE,judiExtendedSource)
judi_forward(A::judiPDE{ADDT,ARDT}, v::judiExtendedSource{vDT}) where {ADDT, ARDT,vDT} = 
    extended_source_modeling(A.model, v.wavelet, A.geometry, nothing, process_input_data(v.weights, A.model, A.info), nothing, 1:A.info.nsrc, 'F', 1, A.options)

# *(judiPDEadjoint, judiExtendedSource)
judi_adjoint(A::judiPDE{ADDT,ARDT}, v::judiExtendedSource{vDT}) where {ADDT,ARDT,vDT} =
    extended_source_modeling(A.model, v.wavelet, A.geometry, process_input_data(v.weights, A.geometry,A.info), nothing, nothing, 1:A.info.nsrc, 'F', -1, A.options)

############################################################
## Additional overloaded functions

# Subsample Modeling operator
function subsample(F::judiPDE{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    geometry = subsample(F.geometry,srcnum)     # Geometry of subsampled data container
    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiPDE(info, F.model, geometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::judiPDE,a) = subsample(F,a)
