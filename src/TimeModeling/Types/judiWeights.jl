############################################################
# judiWeights ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2019

# Updated by Ziyi Yin (ziyi.yin@gatech.edu), Nov 2020

export judiWeights, judiWeightsException, subsample

############################################################

# structure for seismic data as an abstract vector
mutable struct judiWeights{T<:Number} <: judiMultiSourceVector{T}
    nsrc::Integer
    data::Vector{Array{T, N}} where N
end

mutable struct judiWeightsException <: Exception
    msg :: String
end


############################################################

## outer constructors

"""
    judiWeights
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
## overloaded Base functions

conj(a::judiWeights{vDT}) where vDT = judiWeights{vDT}(a.nsrc, conj(a.weights))
transpose(a::judiWeights{vDT}) where vDT = a
adjoint(a::judiWeights{vDT}) where vDT = transpose(conj(a))

jo_convert(::Type{T}, jw::judiWeights{T}, ::Bool) where {T} = jw
jo_convert(::Type{T}, jw::judiWeights{vT}, B::Bool) where {T, vT} = 
	judiWavefield{T}(jv.nsrc, jo_convert(T, jw.weights, B))

##########################################################
function as_cell(V::Vector{T}, n, nblock::Integer) where T
    cells = Vector{Array{T, length(n)}}(undef, nblock)
    for i=1:nblock
        cells[i] = reshape(V[:, i], n)
    end
    cells
end

# *(joLinearFunction, judiWeights)
function *(A::joLinearFunction{ADDT,ARDT},v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    n = size(v.weights[1])
    V = as_cell(A * vcat(vec.(v.weights)...), n, v.nsrc)
    return judiWeights{avDT}(v.nsrc, V)
end

# *(joLinearOperator, judiWeights)
function *(A::joAbstractLinearOperator{ADDT,ARDT}, v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.name == "joDirac" && return avDT(1) .* v
    A.name == "(N*joDirac)" && return avDT(A.fop.a) .* v
    A.name == "adjoint(joDirac)" && return avDT(1) .* v
    A.name == "adjoint((N*joDirac))" && return avDT(A.fop.a) .* v
    return mulJ(A, v)
end

function mulJ(A::joAbstractLinearOperator{ADDT,ARDT}, v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    n = size(v.weights[1])
    try
        # Mul may be defined for judiWeights
        V = A.fop(v)
        jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
        return V
    catch e
        V = A.fop(vcat([vec(v.weights[i]) for i=1:v.nsrc]...))
        jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
        return V
    end
end

# *(joCoreBlock, judiWeights)
function *(A::joCoreBlock{ADDT,ARDT}, v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    V = collect(A.fop[i]*v for i=1:length(A.fop))
    [jo_check_type_match(ARDT,eltype(V[i]),join(["RDT from *(joLinearFunction,judiWeights):",A.fop[i].name,typeof(A.fop[i]),eltype(V)]," / ")) for i=1:length(V)]
    return vcat(V...)
end

function push!(a::judiWeights{T}, b::judiWeights{T}) where T
	append!(a.weights, b.weights)
	a.nsrc += b.nsrc
end

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

getindex(x::judiWeights, a::Integer) = subsample(x,a)
setindex!(x::judiWeights, y, i) = x.weights[i][:] = y

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
