############################################################
# judiWeights ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2019

# Updated by Ziyi Yin (ziyi.yin@gatech.edu), Nov 2020

export judiWeights

############################################################

# structure for seismic data as an abstract vector
mutable struct judiWeights{T<:Number} <: judiMultiSourceVector{T}
    nsrc::Integer
    data::Vector{Array{T, N}} where N
end

# Bypass mismatch in naming and fields for backward compat
Base.getproperty(obj::judiWeights, sym::Symbol) = sym == :weights ? getfield(obj, :data) : getfield(obj, sym)

############################################################

## outer constructors

"""
    judiWeights
        nsrc
        weights
Abstract vector for weighting an extended source, which is injected at every grid point, as weighted by this vector.
Constructors
============
Construct vector cell array of weights. The `weights` keyword\\
can also be a single (non-cell) array, in which case the weights are the same for all source positions:
    judiWeights(weights; nsrc=1)
"""
function judiWeights(weights::Array{T, N}; nsrc=1, vDT::DataType=Float32) where {T<:Real, N}
    (T != vDT) && (weights = convert(Array{vDT},weights))
    # length of vector
    weightsCell = [deepcopy(weights) for j=1:nsrc]
    return judiWeights{vDT}(nsrc, weightsCell)
end

# constructor if weights are passed as a cell array
judiWeights(weights::Vector{Array}; vDT::DataType=Float32) =
    judiWeights(convert.(Array{vDT, length(size(weights[1]))}, weights); vDT=vDT)

function judiWeights(weights::Vector{Array{T, N}}; vDT::DataType=Float32) where {T<:Number, N}
    nsrc = length(weights)
    weights = convert.(Array{vDT, N}, weights)
    return judiWeights{vDT}(nsrc, weights)
end

############################################################
# JOLI conversion
jo_convert(::Type{T}, jw::judiWeights{T}, ::Bool) where {T<:Real} = jw
jo_convert(::Type{T}, jw::judiWeights{vT}, B::Bool) where {T<:Real, vT} = judiWavefield{T}(jv.nsrc, jo_convert.(T, jw.weights, B))
zero(::Type{T}, v::judiWeights{vT}) where {T, vT} = judiWeights{T}(v.nsrc, T(0) .* v.data)
(w::judiWeights)(x::Vector{<:Array}) = judiWeights(x)

function copy!(jv::judiWeights, jv2::judiWeights)
    jv.data .= jv2.data
    jv
end
copyto!(jv::judiWeights, jv2::judiWeights) = copy!(jv, jv2)

function push!(a::judiWeights{T}, b::judiWeights{T}) where T
	append!(a.weights, b.weights)
	a.nsrc += b.nsrc
end

make_input(w::judiWeights) = w.data[1]

# getindex weights container
"""
    getindex(x,source_numbers)
getindex seismic weights vectors or matrix-free linear operators and extract the entries that correspond\\
to the shot positions defined by `source_numbers`. Works for inputs of type `judiWeights`, `judiModeling`, \\
`judiProjection`, `judiJacobian`, `Geometry`, `judiRHS`, `judiPDE`, `judiPDEfull`.
Examples
========
(1) Extract 2 shots from `judiWeights` vector:
    dsub = getindex(dobs,[1,2])
(2) Extract geometry for shot location 100:
    geometry_sub = getindex(dobs.geometry,100)
(3) Extract Jacobian for shots 10 and 20:
    Jsub = getindex(J,[10,20])
"""
getindex(a::judiWeights{avDT}, srcnum::RangeOrVec) where avDT = judiWeights{avDT}(length(srcnum), a.data[srcnum])
