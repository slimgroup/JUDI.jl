############################################################
# judiWeights ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2019

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
function judiWeights(weights::Array; nsrc=1, vDT::DataType=Float32)
    vDT == Float32 || throw(judiWeightsException("Domain type not supported"))

    # length of vector
    n = 1
    m = prod(size(weights))*nsrc
    weightsCell = Array{Array}(undef, nsrc)
    for j=1:nsrc
        weightsCell[j] = deepcopy(weights)
    end
    return judiWeights{Float32}("Extended source weights",m,n,nsrc,weightsCell)
end

# constructor if weights are passed as a cell array
function judiWeights(weights::Union{Array{Any,1},Array{Array,1}}; vDT::DataType=Float32)
    vDT == Float32 || throw(judiWeightsException("Domain and range types not supported"))
    nsrc = length(weights)
    # length of vector
    n = 1
    m = prod(size(weights[1]))*nsrc
    return judiWeights{Float32}("Extended source weights",m,n,nsrc,weights)
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

# minus
function -(a::judiWeights{avDT}) where {avDT}
    c = deepcopy(a)
    for j=1:a.nsrc
        c.weights[j] = -c.weights[j]
    end
    return c
end

# +(judiWeights, judiWeights)
function +(a::judiWeights{avDT}, b::judiWeights{bvDT}) where {avDT, bvDT}
    size(a) == size(b) || throw(judiWeightsException("dimension mismatch"))
    c = deepcopy(a)
    for j=1:a.nsrc
        c.weights[j] = a.weights[j] + b.weights[j]
    end
    return c
end

# -(judiWeights, judiWeights)
function -(a::judiWeights{avDT}, b::judiWeights{bvDT}) where {avDT, bvDT}
    size(a) == size(b) || throw(judiWeightsException("dimension mismatch"))
    c = deepcopy(a)
    for j=1:a.nsrc
        c.weights[j] = a.weights[j] - b.weights[j]
    end
    return c
end

# +(judiWeights, number)
function +(a::judiWeights{avDT},b::Number) where avDT
    c = deepcopy(a)
    for j=1:a.nsrc
        c.weights[j] = c.weights[j] .+ b
    end
    return c
end


# -(judiWeights, number)
function -(a::judiWeights{avDT},b::Number) where avDT
    c = deepcopy(a)
    for j=1:a.nsrc
        c.weights[j] = c.weights[j] .- b
    end
    return c
end

# *(judiWeights, number)
function *(a::judiWeights{avDT},b::Number) where avDT
    c = deepcopy(a)
    for j=1:a.nsrc
        c.weights[j] = c.weights[j] .* b
    end
    return c
end


# *(Array, judiWeights)
function *(A::Union{Array, Adjoint, Transpose}, x::judiWeights)
    xvec = vec(x.weights[1])
    if x.nsrc > 1
        for j=2:x.nsrc
            xvec = [xvec; vec(x.weights[j])]
        end
    end
    return A*xvec
end

# /(judiWeights, number)
function /(a::judiWeights{avDT},b::Number) where avDT
    c = deepcopy(a)
    if iszero(b)
        error("Division by zero")
    else
        c.weights = c.weights/b
    end
    return c
end

# +(number, judiWeights)
+(a::Number, b::judiWeights{avDT}) where{avDT} = b + a
-(a::Number, b::judiWeights{avDT}) where{avDT} = -1f0*(b - a)
*(a::Number, b::judiWeights{avDT}) where{avDT} = b * a

# *(joLinearFunction, judiWeights)
function *(A::joLinearFunction{ADDT,ARDT},v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    n = size(v.weights[1])
    V = A.fop(hcat([vec(v.weights[i]) for i=1:v.nsrc]...))
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    m = length(V)
    return judiWeights{avDT}("Extended source weights", m, 1, v.nsrc, [reshape(V[:, i], n) for i=1:length(v.weights)])
end

# *(joLinearOperator, judiWeights)
function *(A::joLinearOperator{ADDT,ARDT}, v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    n = size(v.weights[1])
    V = A.fop(hcat([vec(v.weights[i]) for i=1:v.nsrc]...))
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    m = length(V)
    return judiWeights{avDT}("Extended source weights", m, 1, v.nsrc, [reshape(V[:, i], n) for i=1:v.nsrc])
end

# *(joCoreBlock, judiWeights)
function *(A::joCoreBlock{ADDT,ARDT}, v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    V = vcat(collect(A.fop[i]*v for i=1:length(A.fop))...)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(joKron, judiWeights)
function *(A::joKron{ADDT,ARDT}, v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    n = size(v.weights[1])
    V = A.fop(hcat([vec(v.weights[i]) for i=1:v.nsrc]...))
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    m = length(V)
    return judiWeights{avDT}("Extended source weights", m, 1, v.nsrc, [reshape(V[:, i], n) for i=1:v.nsrc])
end

# *(joMatrix, judiWeights)
function *(A::joMatrix{ADDT,ARDT}, v::judiWeights{avDT}) where {ADDT, ARDT, avDT}
    A.n == size(v,1) || throw(judiWeightsException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiWeights):",A.name,typeof(A),avDT]," / "))
    # Evaluate as mat-mat over the weights
    n = size(v.weights[1])
    V = A.fop(hcat([vec(v.weights[i]) for i=1:v.nsrc]...))
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiWeights):",A.name,typeof(A),eltype(V)]," / "))
    m = length(V)
    return judiWeights{avDT}("Extended source weights", m, 1, v.nsrc, [reshape(V[:, i], n) for i=1:v.nsrc])
end

# vcat
function vcat(a::judiWeights{avDT},b::judiWeights{bvDT}) where {avDT, bvDT}
    m = a.m + b.m
    n = 1
    nsrc = a.nsrc + b.nsrc
    weights = Array{Array}(undef, nsrc)

    # Merge data sets and geometries
    for j=1:a.nsrc
        weights[j] = a.weights[j]
    end
    for j=a.nsrc+1:nsrc
        weights[j] = b.weights[j-a.nsrc]
    end
    nvDT = promote_type(avDT,bvDT)
    return judiWeights{nvDT}(a.name,m,n,nsrc,weights)
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
    x = 0.f0
    for j=1:a.nsrc
        x += sum(abs.(vec(a.weights[j])).^p)
    end
    return x^(1.f0/p)
end

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

getindex(x::judiWeights, a::Integer) = subsample(x,a)

function reshape(x::judiWeights, dims...)
    prod((dims)) == prod((x.m, x.n)) || throw(judiWeightsException(join(["DimensionMismatch(\"new dimensions ", string(dims), " must be consistent with array size ", string(prod((x.m, x.n)))])))
    length(dims) <= 2 || throw(judiWeightsException("Cannot reshape into array with dimensions > 2"))
    x_out = deepcopy(x)
    x_out.m = dims[1]
    x_out.n = dims[2]
    return x_out
end

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
        z.data[j] = x.weights[j] .* y.weights[j]
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
        z.data[j] = x.weights[j] ./ y.weights[j]
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

function broadcast!(identity, x::judiWeights, a::Number, y::judiWeights, z::judiWeights)
    size(x) == size(y) || throw(judiWeightsException("dimension mismatch"))
    size(x) == size(z) || throw(judiWeightsException("dimension mismatch"))
    scale!(y,a)
    copy!(x, y + z)
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

