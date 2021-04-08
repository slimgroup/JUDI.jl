############################################################
# judiExtendedSource ## ####################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiExtendedSource, judiExtendedSourceException

############################################################

struct judiExtendedSource{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    wavelet
    weights
end

mutable struct judiExtendedSourceException <: Exception
    msg :: String
end


############################################################

## outer constructors

"""
    judiExtendedSource
        name::String
        m::Integer
        n::Integer
        info::Info
        wavelet
        weights
Abstract sparse vector for an extended source. The `judiRHS` vector has the\\
dimensions of the full time history of the wavefields, but contains only the wavelet, \\
as well as the weights of the extended source.
Constructor
==========
    judiExtendedSource(info, wavelet, weights)
Examples
========
Assuming `Pw` ia a projection operators of type `judiLRWF` and `w` is a seismic weight \\
vector of type `judiWeights`, then a `judiExtendedSource` vector can be created as follows:
    q_ext = adjoint(Pw)*w    # abstract extended source
"""
function judiExtendedSource(info,wavelet,weights;vDT::DataType=Float32)
    vDT == Float32 || throw(judiExtendedSourceException("Domain type not supported"))
    # length of vector
    m = info.n * sum(info.nt)
    n = 1
    return judiExtendedSource{Float32}("judiExtendedSource",m,n,info,wavelet,weights)
end

####################################################################
## overloaded Base functions

# conj(jo)
conj(A::judiExtendedSource{vDT}) where vDT =
    judiExtendedSource{vDT}("conj("*A.name*")",A.m,A.n,A.info,A.wavelet,A.weights)

# transpose(jo)
transpose(A::judiExtendedSource{vDT}) where vDT =
    judiExtendedSource{vDT}(""*A.name*".'",A.n,A.m,A.info,A.wavelet,A.weights)

# adjoint(jo)
adjoint(A::judiExtendedSource{vDT}) where vDT =
    judiExtendedSource{vDT}(""*A.name*"'",A.n,A.m,A.info,A.wavelet,A.weights)

####################################################################

# +(judiExtendedSource,judiExtendedSource)
function +(A::judiExtendedSource{avDT}, B::judiExtendedSource{bvDT}) where {avDT, bvDT}
    # Error checking
    size(A) == size(B) || throw(judiExtendedSourceException("Shape mismatch: A:$(size(A)), v: $(size(B))"))
    compareInfo(A.info, B.info) == true || throw(judiExtendedSourceException("info mismatch"))

    # Size
    m = A.info.n * sum(A.info.nt)
    n = 1

    # wavelet and weights
    A.wavelet == B.wavelet ? nothing : throw(judiExtendedSourceException("Can only add two extended
                                                                          sources with same wavelet"))
    wavelet = A.wavelet
    weights = A.weights .+ B.weights

    nvDT = promote_type(avDT,bvDT)

    return judiExtendedSource{nvDT}("judiRHS",m,n,A.info, wavelet, weights)
end

# -(judiExtendedSource,judiExtendedSource)
function -(A::judiExtendedSource{avDT}, B::judiExtendedSource{bvDT}) where {avDT, bvDT}

        # Error checking
        size(A) == size(B) || throw(judiExtendedSourceException("Shape mismatch: A:$(size(A)), v: $(size(B))"))
        compareInfo(A.info, B.info) == true || throw(judiExtendedSourceException("info mismatch"))

        # Size
        m = A.info.n * sum(A.info.nt)
        n = 1

        # wavelet and weights
        A.wavelet == B.wavelet ? nothing : throw(judiExtendedSourceException("Can only add two extended
                                                                            sources with same wavelet"))
        wavelet = A.wavelet
        weights = A.weights .- B.weights

        nvDT = promote_type(avDT,bvDT)

        return judiExtendedSource{nvDT}("judiRHS",m,n,A.info, wavelet, weights)
    end

function subsample(a::judiExtendedSource{avDT},srcnum) where avDT
    info = Info(a.info.n, length(srcnum), a.info.nt[srcnum])
    return judiExtendedSource(info,a.wavelet[srcnum],a.weights[srcnum];vDT=avDT)
end

getindex(x::judiExtendedSource,a) = subsample(x,a)
