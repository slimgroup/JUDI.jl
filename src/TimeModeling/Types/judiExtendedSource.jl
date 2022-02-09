############################################################
# judiExtendedSource ## ####################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiExtendedSource

############################################################

struct judiExtendedSource{D<:Number} <: judiAbstractLinearOperator{D,D}
    m::Integer
    n::Integer
    info::Info
    wavelet
    weights
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
function judiExtendedSource(info::Info, wavelet, weights; vDT::DataType=Float32) where T
    vDT == Float32 || throw(judiLinearException("Domain type not supported"))
    # length of vector
    m = info.n * sum(info.nt)
    n = 1
    return judiExtendedSource{Float32}(m,n,info,wavelet,weights)
end

####################################################################

# +(judiExtendedSource,judiExtendedSource)
function +(A::judiExtendedSource{avDT}, B::judiExtendedSource{bvDT}) where {avDT, bvDT}
    # Error checking
    size(A) == size(B) || throw(judiLinearException("Shape mismatch: A:$(size(A)), v: $(size(B))"))
    compareInfo(A.info, B.info) == true || throw(judiLinearException("info mismatch"))

    # Size
    m = A.info.n * sum(A.info.nt)
    n = 1

    # wavelet and weights
    A.wavelet == B.wavelet ? nothing : throw(judiLinearException("Can only add two extended
                                                                          sources with same wavelet"))
    wavelet = A.wavelet
    weights = A.weights .+ B.weights

    nvDT = promote_type(avDT,bvDT)

    return judiExtendedSource{nvDT}(m,n,A.info, wavelet, weights)
end

# -(judiExtendedSource,judiExtendedSource)
function -(A::judiExtendedSource{avDT}, B::judiExtendedSource{bvDT}) where {avDT, bvDT}

        # Error checking
        size(A) == size(B) || throw(judiLinearException("Shape mismatch: A:$(size(A)), v: $(size(B))"))
        compareInfo(A.info, B.info) == true || throw(judiLinearException("info mismatch"))

        # Size
        m = A.info.n * sum(A.info.nt)
        n = 1

        # wavelet and weights
        A.wavelet == B.wavelet ? nothing : throw(judiLinearException("Can only add two extended
                                                                            sources with same wavelet"))
        wavelet = A.wavelet
        weights = A.weights .- B.weights

        nvDT = promote_type(avDT,bvDT)

        return judiExtendedSource{nvDT}(m,n,A.info, wavelet, weights)
    end

function subsample(a::judiExtendedSource{avDT},srcnum) where avDT
    info = Info(a.info.n, length(srcnum), a.info.nt[srcnum])
    return judiExtendedSource(info,a.wavelet[srcnum],a.weights[srcnum];vDT=avDT)
end

*(a::T, E::judiExtendedSource{T}) where T = judiExtendedSource{T}(E.m,E.n,E.info, a*E.wavelet, E.weights)
*(a::Number, E::judiExtendedSource{T}) where T = convert(T, a) * E

getindex(x::judiExtendedSource,a) = subsample(x,a)
