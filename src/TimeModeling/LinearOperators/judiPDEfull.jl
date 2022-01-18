############################################################
# judiPDEfull ################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiPDEfull, subsample

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections

struct judiPDEfull{D<:Number,R<:Number} <: judiAbstractLinearOperator{D, R}
    m::Integer
    n::Integer
    info::Info
    model::Model
    srcGeometry::Geometry
    recGeometry::Geometry
    options::Options
end

############################################################
## Constructor

# Set up info structure for linear operators
function judiModeling(model::Model, srcGeometry::Geometry, recGeometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
    ntComp = get_computational_nt(srcGeometry, recGeometry, model)
    info = Info(prod(model.n), get_nsrc(srcGeometry), ntComp)
    return judiModeling(info, model, srcGeometry, recGeometry; options=options, DDT=DDT, RDT=RDT)
end

function judiModeling(info::Info, model::Model, srcGeometry::Geometry, recGeometry::Geometry; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
# JOLI wrapper for nonlinear forward modeling
    (DDT == Float32 && RDT == Float32) || throw(judiLinearException("Domain and range types not supported"))

    # Determine dimensions
    m, n = n_samples(recGeometry, info), n_samples(srcGeometry, info)
    return judiPDEfull{Float32,Float32}(m, n, info, model, srcGeometry, recGeometry, options)
end

############################################################
## overloaded Base *(...judiPDEfull...)
judi_forward(A::judiPDEfull{D, R}, v::vT) where {D, R, vT<:Union{<:judiVector, <:AbstractVector}} =
    time_modeling(A.model, A.srcGeometry, process_input_data(v, A.srcGeometry, A.info), A.recGeometry, nothing, nothing, 1:A.info.nsrc, 'F', 1, A.options)

judi_adjoint(A::judiPDEfull{D, R}, v::vT) where {D, R, vT<:Union{<:judiVector, <:AbstractVector}} =
    time_modeling(A.model, A.srcGeometry, nothing, A.recGeometry, process_input_data(v, A.recGeometry, A.info), nothing, 1:A.info.nsrc, 'F', 1, A.options)

############################################################
# Subsample Modeling operator
function subsample(F::judiPDEfull{ADDT,ARDT}, srcnum) where {ADDT,ARDT}

    srcGeometry = subsample(F.srcGeometry,srcnum)       # Geometry of subsampled data container
    recGeometry = subsample(F.recGeometry,srcnum)

    info = Info(F.info.n, length(srcnum), F.info.nt[srcnum])
    return judiModeling(info, F.model, srcGeometry, recGeometry; options=F.options, DDT=ADDT, RDT=ARDT)
end

getindex(F::judiPDEfull, a) = subsample(F,a)
