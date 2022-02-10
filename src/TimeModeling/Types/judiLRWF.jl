############################################################
# judiLRWF #############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017
# Updated: March 2021, Mathias Louboutin (mlouboutin3@gatech.edu)

export judiLRWF

############################################################

# Type for linear operator representing  Pr*A(m)^-1*Ps,
# i.e. it includes source and receiver projections
struct judiLRWF{T} <: judiMultiSourceVector{T}
    nsrc::Integer
    data
end

# Bypass mismatch in naming and fields for backward compat
Base.getproperty(obj::judiWeights, sym::Symbol) = sym == :wavelet ? getfield(obj, :data) : getfield(obj, sym)

############################################################
## Constructor
"""
    judiLRWF(nsrc, data)
Low-rank wavefield operator which injects a wavelet q at every point of the subsurface. \\
`info` is an `Info` structure and `wavelet` is a cell array containing the wavelet(s).
Examples
========
`F` is a modeling operator of type `judiModeling` and `w` is a weighting matrix of type `judiWeights`:
    Pr = judiProjection(info, rec_geometry)
    Pw = judiLRWF(nsrc, q.data)
    dobs = Pr*F*Pw'*w
    dw = Pw*F'*Pr'*dobs
"""
function judiLRWF(nsrc::Integer, data::Array{T, N}) where {T, N}
    T == Float32 || throw(judiLinearException("Domain and range types not supported"))
    wavelet = Vector{Array{T, N}}(undef, nsrc)
    for j=1:nsrc
        wavelet[j] = data
    end
    return judiLRWF{Float32}(nsrc, wavelet)
end


function judiLRWF(nsrc::Integer, wavelet::Vector{Array{T, N}}) where {T, N}
    T == Float32 || throw(judiLinearException("Domain and range types not supported"))
    return judiLRWF{Float32}(nsrc, wavelet)
end

judiLRWF(wavelet::Vector{Array{T, N}}) where {T, N}) = judiLRWF(length(wavelet), wavelet)
############################################################
# JOLI conversion
jo_convert(::Type{T}, jv::judiLRWF{T}, ::Bool) where {T<:Real} = jv
jo_convert(::Type{T}, jv::judiLRWF{vT}, B::Bool) where {T<:Real, vT} = judiLRWF{T}(jv.nsrc, jo_convert.(T, jv.data, B))
zero(::Type{T}, v::judiLRWF{vT}) where {T, vT} = judiLRWF{T}(v.nsrc, Vector{Array{T, ndims(v.data[1])}}(undef, v.nsrc))

############################################################
## Additional overloaded functions

# Subsample Modeling operator
subsample(P::judiLRWF{T}, srcnum) where {T} = judiLRWF(P.data[srcnum])
