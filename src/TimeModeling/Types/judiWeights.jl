############################################################
# judiWeights ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2019

# Updated by Ziyi Yin (ziyi.yin@gatech.edu), Nov 2020

export judiWeights, judiWeightsException, subsample

############################################################

# structure for seismic data as an abstract vector
mutable struct judiWeights{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
    name::String
    m::Integer
    n::Integer
    nsrc::Integer
    weights
end

mutable struct judiWeightsException <: Exception
    msg :: String
end

############################################################

## outer constructors

"""
    judiWeights
        name::String
        m::Integer
        n::Integer
        nsrc::Integer
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
    n = 1
    m = prod(size(weights))*nsrc
    weightsCell = [deepcopy(weights) for j=1:nsrc]
    return judiWeights{vDT}("Extended source weights",m,n,nsrc,weightsCell)
end

# constructor if weights are passed as a cell array
judiWeights(weights::Vector{Array}; vDT::DataType=Float32) =
    judiWeights(convert.(Array{vDT, length(size(weights[1]))}, weights); vDT=vDT)

function judiWeights(weights::Vector{Array{T, N}}; vDT::DataType=Float32) where {T<:Number, N}
    nsrc = length(weights)
    weights = convert.(Array{vDT, N}, weights)
    # length of vector
    n = 1
    m = prod(size(weights[1]))*nsrc
    return judiWeights{vDT}("Extended source weights",m,n,nsrc,weights)
end

############################################################
## overloaded Base functions

# conj(jo)
conj(a::judiWeights{vDT}) where vDT =
    judiWeights{vDT}("conj("*a.name*")",a.m,a.n,a.nsrc,a.weights)

# transpose(jo)
transpose(a::judiWeights{vDT}) where vDT =
    judiWeights{vDT}(""*a.name*".'",a.n,a.m,a.nsrc,a.weights)

# adjoint(jo)
adjoint(a::judiWeights{vDT}) where vDT =
        judiWeights{vDT}(""*a.name*".'",a.n,a.m,a.nsrc,a.weights)

##########################################################
# *(joLinearFunction, judiWeights)
function *(A::joLinearFunction{ADDT,ARDT},v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    n = size(v.weights[1])
    V = A.fop(vcat([vec(v.weights[i]) for i=1:v.nsrc]...))
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    m = length(V)
    return judiWeights{avDT}("Extended source weights", m, 1, v.nsrc, [reshape(V[:, i], n) for i=1:length(v.weights)])
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

# vcat
function vcat(ai::Vararg{judiWeights{avDT}, N}) where {avDT, N}
    N == 1 && (return ai[1])
    N > 2 && (return vcat(ai[1], vcat(ai[2:end]...)))
    a, b = ai
    m = a.m + b.m
    n = 1
    nsrc = a.nsrc + b.nsrc
    weights = vcat(a.weights, b.weights)
    return judiWeights{avDT}(a.name,m,n,nsrc,weights)
end


function push!(a::judiWeights{T}, b::judiWeights{T}) where T
	append!(a.weights, b.weights)
	a.m += b.m
	a.nsrc += b.nsrc
end

# dot product
function dot(a::judiWeights{avDT}, b::judiWeights{bvDT}) where {avDT, bvDT}
# Dot product for data containers
    size(a) == size(b) || throw(judiWeightsException("dimension mismatch"))
    dotprod = 0f0
    for j=1:a.nsrc
        dotprod += dot(vec(a.weights[j]), vec(b.weights[j]))
    end
    return dotprod
end

# norm
function norm(a::judiWeights{avDT}, p::Real=2) where avDT
    if p == Inf
        return max([maximum(abs.(a.weights[i])) for i=1:a.nsrc]...)
    end
    x = 0.f0
    for j=1:a.nsrc
        x += sum(abs.(vec(a.weights[j])).^p)
    end
    return x^(1.f0/p)
end

#maximum
maximum(a::judiWeights{avDT}) where avDT =   max([maximum(a.weights[i]) for i=1:a.nsrc]...)

#minimum
minimum(a::judiWeights{avDT}) where avDT =   min([minimum(a.weights[i]) for i=1:a.nsrc]...)

# abs
function abs(a::judiWeights{avDT}) where avDT
    b = deepcopy(a)
    for j=1:a.nsrc
        b.weights[j] = abs.(a.weights[j])
    end
    return b
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
function subsample(a::judiWeights{avDT},srcnum) where avDT
    srcnum > a.nsrc ? sub = 0 : sub = judiWeights(a.weights[srcnum];vDT=avDT)
    return sub
end

length(x::judiWeights) = x.m

getindex(x::judiWeights, a::Integer) = subsample(x,a)

setindex!(x::judiWeights, y, i) = x.weights[i][:] = y

firstindex(x::judiWeights) = 1

lastindex(x::judiWeights) = x.nsrc

axes(x::judiWeights) = Base.OneTo(x.nsrc)

ndims(x::judiWeights) = length(size(x))

copy(x::judiWeights) = judiWeights{Float32}("Extended source weights", x.m, x.n, x.nsrc, x.weights)

similar(x::judiWeights) = judiWeights{Float32}("Extended source weights", x.m, x.n, x.nsrc, 0f0 * x.weights)

similar(x::judiWeights, element_type::DataType, dims::Union{AbstractUnitRange, Integer}...) = similar(x)

isfinite(x::judiWeights) = all(all(isfinite.(x.weights[i])) for i=1:length(x.weights))

iterate(S::judiWeights, state::Integer=1) = state > length(S.weights) ? nothing : (S.weights[state], state+1)

####################################################################################################

BroadcastStyle(::Type{judiWeights}) = Base.Broadcast.DefaultArrayStyle{1}()

ndims(::Type{judiWeights{Float32}}) = 1

### +/- ####
broadcasted(::typeof(+), x::judiWeights, y::judiWeights) = x + y
broadcasted(::typeof(-), x::judiWeights, y::judiWeights) = x - y

broadcasted(::typeof(+), x::judiWeights, y::Number) = x + y
broadcasted(::typeof(-), x::judiWeights, y::Number) = x - y

broadcasted(::typeof(+), x::Number, y::judiWeights) = x + y
broadcasted(::typeof(-), x::Number, y::judiWeights) = x - y

### * ####
function broadcasted(::typeof(*), x::judiWeights, y::judiWeights)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.weights)
        z.weights[j] = x.weights[j] .* y.weights[j]
    end
    return z
end

function broadcasted(::typeof(*), x::judiWeights, y::Number)
    z = deepcopy(x)
    for j=1:length(x.weights)
        z.weights[j] .*= y
    end
    return z
end

broadcasted(::typeof(*), x::Number, y::judiWeights) = broadcasted(*, y, x)

### / ####
function broadcasted(::typeof(/), x::judiWeights, y::judiWeights)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    z = deepcopy(x)
    for j=1:length(x.weights)
        z.weights[j] = x.weights[j] ./ y.weights[j]
    end
    return z
end

broadcasted(::typeof(/), x::judiWeights, y::Number) = broadcasted(*, x, 1/y)

# Materialize for broadcasting
function materialize!(x::judiWeights, y::judiWeights)
    for j=1:length(x.weights)
        x.weights[j] .= y.weights[j]
    end
    return x
end

function broadcast!(identity, x::judiWeights, y::judiWeights)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    copy!(x,y)
end

function broadcasted(identity, x::judiWeights)
    return x
end

function copy!(x::judiWeights, y::judiWeights)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    for j=1:x.nsrc
        x.weights[j] = y.weights[j]
    end
end

function isapprox(x::judiWeights, y::judiWeights; rtol::Real=sqrt(eps()), atol::Real=0)
    isapprox(x.weights, y.weights; rtol=rtol, atol=atol)
end

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
