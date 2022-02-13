############################################################
# judiWeights ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2019

# Updated by Ziyi Yin (ziyi.yin@gatech.edu), Nov 2020

export judiWeights, subsample

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

make_input(w::judiWeights, dtComp) = Dict(:w=>w.data[1])
# Subsample weights container
"""
    subsample(x,source_numbers)
Subsample seismic weights vectors or matrix-free linear operators and extract the entries that correspond\\
to the shot positions defined by `source_numbers`. Works for inputs of type `judiWeights`, `judiModeling`, \\
`judiProjection`, `judiJacobian`, `Geometry`, `judiRHS`, `judiPDE`, `judiPDEfull`.
Examples
========
(1) Extract 2 shots from `judiWeights` vector:
    dsub = subsample(dobs,[1,2])
(2) Extract geometry for shot location 100:
    geometry_sub = subsample(dobs.geometry,100)
(3) Extract Jacobian for shots 10 and 20:
    Jsub = subsample(J,[10,20])
"""
subsample(a::judiWeights{avDT}, srcnum) where avDT = judiWeights(a.weights[srcnum];vDT=avDT)

############################################################

function A_mul_B!(x::judiWeights, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiWeights)
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.weights)
        x.weights[j] .= z.weights[j]
    end
end

function A_mul_B!(x::judiWeights, F::Union{joAbstractLinearOperator, joLinearFunction}, y::Array)
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.weights)
        x.weights[j] .= z.weights[j]
    end
end

function A_mul_B!(x::Array, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiWeights)
    F.m == size(y, 1) ? x[:] .= adjoint(F)*y : x[:] .= F*y
end

mul!(x::judiWeights, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiWeights) = A_mul_B!(x, F, y)
mul!(x::judiWeights, F::Union{joAbstractLinearOperator, joLinearFunction}, y::Array) = A_mul_B!(x, F, y)
mul!(x::Array, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiWeights) = A_mul_B!(x, F, y)
