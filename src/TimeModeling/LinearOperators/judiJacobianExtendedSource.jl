############################################################
# judiJacobianExQ ###############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiJacobianExQ, judiJacobianExQException, subsample

###########################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiJacobianExQ{DDT<:Number,RDT<:Number} <: judiAbstractJacobian{DDT,RDT}
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

judiJacobian(J::judiJacobianExQ{DDT,RDT}; name=J.name, m=J.m, n=J.n, info=J.info, model=J.model, wavelet=J.wavelet,
     weights=J.weights, geom=J.recGeometry, opt=J.options, fop=J.fop, fop_T=J.fop_T) where {DDT, RDT} =
        judiJacobianExQ{DDT,RDT}(name, m, n, info, model, geom, wavelet, weights, opt, fop, fop_T)

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
                                                bornop, adjbornop)
end


############################################################
## Additional overloaded functions

# Subsample Jacobian
function subsample(J::judiJacobianExQ{ADDT,ARDT}, srcnum) where {ADDT,ARDT}
    recGeometry = subsample(J.recGeometry,srcnum)
    nsrc = typeof(srcnum) <: Int ? 1 : length(srcnum)
    info = Info(J.info.n, nsrc, J.info.nt[srcnum])
    m = typeof(recGeometry) == GeometryOOC ? sum(recGeometry.nsamples) : sum([length(recGeometry.xloc[j])*recGeometry.nt[j] for j=1:nsrc])

    opt = subsample(J.options, srcnum)
    nsrc == 1 && (srcnum = srcnum:srcnum)
    return judiJacobian(J; m=m, info=info, weights=J.weights[srcnum], wavelet=J.wavelet[srcnum],
                        geom=recGeometry, opt=opt)
end

# *(num,judiJacobian)
*(a::Number,A::judiJacobianExQ{ADDT,ARDT}) where {ADDT,ARDT} =  judiJacobian(A; wavelet=a*A.wavelet)
# -(judiJacobian)
-(A::judiJacobianExQ{DDT,RDT}) where {DDT,RDT} = judiJacobian(A; wavelet=-A.wavelet)

############################################################
## Forward/adjoint function to avoid unecessary extra declaration

function bornop(J::judiJacobianExQ, v)
    srcnum = 1:J.info.nsrc
    return extended_source_modeling(J.model, J.wavelet, J.recGeometry, nothing, J.weights, v, srcnum, 'J', 1, J.options)
end

function adjbornop(J::judiJacobianExQ, w)
    srcnum = 1:J.info.nsrc
    return extended_source_modeling(J.model, J.wavelet, J.recGeometry, process_input_data(w, J.info),
                                    J.weights, nothing, srcnum, 'J', -1, J.options)
end