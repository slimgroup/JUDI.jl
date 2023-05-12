# Model structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
# Author Mathias Louboutin
# Date: 2017-2022
#
export Model, PhysicalParameter, get_dt

abstract type AbstractModel{T, N} end

struct DiscreteGrid{T, N}
    n::NTuple{N, Int64}
    d::NTuple{N, T}
    o::NTuple{N, T}
    nb::Integer # number of absorbing boundaries points on each side
end

###################################################################################################
# PhysicalParameter abstract vector

"""
    PhysicalParameter
        n::NTuple{N, T}
        d::NTuple{N, Tf}
        o::NTuple{N, Tf}
        data::Union{Array, Number}

PhysicalParameter structure for physical space parameter.

`n`: number of gridpoints in (x,y,z) for 3D or (x,z) for 2D

`d`: grid spacing in (x,y,z) or (x,z) (in meters)

`o`: origin of coordinate system in (x,y,z) or (x,z) (in meters)

`data`: the array of the parameter values of size n

Constructor
===========

A `PhysicalParameter` can be constructed in various ways but always require the origin `o` and grid spacing `d` that
cannot be infered from the array.

    PhysicalParameter(v::Array{T}, d, o) where `v` is an n-dimensional array and n=size(v)

    PhysicalParameter(n, d, o; T=Float32) Creates a zero PhysicalParameter

    PhysicalParameter(v::Array{T}, A::PhysicalParameter{T, N}) where {T<:Number, N} Creates a PhysicalParameter from the Array `v` with n, d, o from `A`

    PhysicalParameter(v::Array{T, N}, n::NTuple{N, T}, d::NTuple{N, T}, o::Tuple) where `v` is a vector or nd-array that is reshaped into shape `n`

    PhysicalParameter(v::T, n::NTuple{N, T}, d::NTuple{N, T}, o::Tuple) Creates a constant (single number) PhyicalParameter

"""
mutable struct PhysicalParameter{T, N} <: DenseArray{T, N}
    n::NTuple{N, Int64}
    d::NTuple{N, T}
    o::NTuple{N, T}
    data::Union{Array{T}, T}
end

mutable struct PhysicalParameterException <: Exception
    msg :: String
end

PhysicalParameter(v::BitArray{N}, args...) where N = v
PhysicalParameter(v::Array{Bool}, args...) = v

PhysicalParameter(n::NTuple{N, Int64}, d::NTuple{N, Td}, o::NTuple{N, To}, data::Union{Array{T}, T}) where {Td, To, T, N} =
    PhysicalParameter{T, N}(n, T.(d), T.(o), data)

function PhysicalParameter(v::Array{T, N}, d::NTuple{N, T}, o::NTuple{N, T}) where {T<:Number, N}
    n = size(v)
    length(n) != length(o) && throw(PhysicalParameterException("Input array should be $(length(o))-dimensional"))
    return PhysicalParameter{T, N}(n, d, o, reshape(v, n))
end

function PhysicalParameter(n::NTuple{N, Int}, d::NTuple{N, T}, o::NTuple{N, T}; DT::Type=eltype(d)) where {T<:Number, N}
    return PhysicalParameter{DT, N}(n, d, o, zeros(DT, n))
end

function PhysicalParameter(v::Array{T, N}, A::PhysicalParameter{ADT, N}) where {T<:Number, ADT<:Number, N}
    return PhysicalParameter{T, N}(A.n, A.d, A.o, reshape(v, A.n))
end

function PhysicalParameter(v::Array{T, N1}, n::NTuple{N, Int}, d::NTuple{N, T}, o::NTuple{N, T}) where {T<:Number, N, N1}
    length(v) != prod(n) && throw(PhysicalParameterException("Incompatible number of element in input $(length(v)) with n=$(n)"))
    N1 == 1 && (v = reshape(v, n))
    return PhysicalParameter{T, N}(n, d, o, v)
end

function PhysicalParameter(v::vT, n::NTuple{N, Int}, d::NTuple{N, T}, o::NTuple{N, T}) where {vT<:Number, T<:Number, N}
    return PhysicalParameter{T, N}(n, d, o, T(v))
end

PhysicalParameter(p::PhysicalParameter{T, N}, ::NTuple{N, Int}, ::NTuple{N, T}, ::NTuple{N, T}) where {T<:Number, N} = p 
PhysicalParameter(p::PhysicalParameter{T, N}) where {T<:Number, N} = p 
PhysicalParameter(p::PhysicalParameter{T, N}, v::Array{T, Nv}) where {T<:Number, N, Nv} = PhysicalParameter(reshape(v, p.n), p.d, p.o)

# transpose and such.....
conj(x::PhysicalParameter{T, N}) where {T<:Number, N} = x
transpose(x::PhysicalParameter{T, N}) where {T<:Number, N} = PhysicalParameter{T, N}(x.n[N:-1:1], x.d[N:-1:1], x.o[N:-1:1], permutedims(x.data, N:-1:1))
adjoint(x::PhysicalParameter{T, N}) where {T<:Number, N} = transpose(x)

# Basic overloads
size(A::PhysicalParameter{T, N}) where {T<:Number, N} = A.n
length(A::PhysicalParameter{T, N}) where {T<:Number, N} = prod(A.n)

function norm(A::PhysicalParameter{T, N}, order::Real=2) where {T<:Number, N}
    return norm(vec(A.data), order)
end

dot(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Number, N} = dot(vec(A.data), vec(B.data))
dot(A::PhysicalParameter{T, N}, B::Array{T, N}) where {T<:Number, N} = dot(vec(A.data), vec(B))
dot(A::Array{T, N}, B::PhysicalParameter{T, N}) where {T<:Number, N} = dot(vec(A), vec(B.data))

_repr(A::PhysicalParameter{T, N}) where {T<:Number, N} = "$(typeof(A)) of size $(A.n) with origin $(A.o) and spacing $(A.d)"
display(A::PhysicalParameter{T, N}) where {T<:Number, N} = println(_repr(A))
show(io::IO, A::PhysicalParameter{T, N}) where {T<:Number, N} = print(io, _repr(A))
summary(io::IO, A::PhysicalParameter{T, N}) where {T<:Number, N} = print(io, _repr(A))
showarg(io::IO, A::PhysicalParameter, toplevel) = print(io, _repr(A))
show(io::IO, ::MIME{Symbol("text/plain")}, A::PhysicalParameter{T, N}) where {T<:Number, N} = println(io, _repr(A))

# Indexing
firstindex(A::PhysicalParameter{T, N}) where {T<:Number, N} = 1
lastindex(A::PhysicalParameter{T, N}) where {T<:Number, N} = length(A)
lastindex(A::PhysicalParameter{T, N}, dim::Int) where {T<:Number, N} = A.n[dim]

function promote_shape(p::PhysicalParameter{T, N}, A::Array{T, Na}) where {T, N, Na}
    (size(A) != p.n && N>1) && return promote_shape(p.data, A)
    (length(A) == prod(p.n) && N==1) && return size(A)
    return promote_shape(A, A)
end

promote_shape(A::Array{T, Na}, p::PhysicalParameter{T, N}) where {T<:Number, N, Na}  = promote_shape(p, A)
reshape(p::PhysicalParameter{T, N}, n::Tuple{Vararg{Int64,N}}) where {T<:Number, N} = (n == p.n ? p : reshape(p.data, n))

dotview(A::PhysicalParameter{T, N}, I::Vararg{Union{Function, Int, UnitRange{Int}}, Ni}) where {T, N, Ni} = dotview(A.data, I...)
Base.dotview(m::PhysicalParameter, i) = Base.dotview(m.data, i)

getindex(A::PhysicalParameter{T, N}, i::Int) where {T<:Number, N} = A.data[i]
getindex(A::PhysicalParameter{T, N}, ::Colon) where {T<:Number, N} = A.data[:]

get_step(r::StepRange) = r.step
get_step(r) = 1

function getindex(A::PhysicalParameter{T, N}, I::Vararg{Union{Int, BitArray, Function, StepRange{Int}, UnitRange{Int}}, Ni}) where {N, Ni, T<:Number}
    new_v = getindex(A.data, I...)
    length(size(new_v)) != length(A.n) && (return new_v)
    s = [i == (:) ? 0 : i[1]-1 for i=I]
    st = [get_step(i) for i=I]
    new_o = [ao+i*d for (ao, i, d)=zip(A.o, s, A.d)]
    new_d = [d*s for (d, s)=zip(A.d, st)]
    PhysicalParameter{T, N}(size(new_v), tuple(new_d...), tuple(new_o...), new_v)
end

setindex!(A::PhysicalParameter{T, N}, v, I::Vararg{Union{Int, Function, UnitRange{Int}}, Ni}) where {T<:Number, N, Ni}  = setindex!(A.data, v, I...)
setindex!(A::PhysicalParameter{T, N}, v, i::Int) where {T<:Number, N} = (A.data[i] = v)

# Constructiors by copy
similar(x::PhysicalParameter{T, N}) where {T<:Number, N} = PhysicalParameter{T, N}(x.n, x.d, x.o, fill!(similar(x.data), 0))

copy(x::PhysicalParameter{T, N}) where {T<:Number, N} = PhysicalParameter{T, N}(x.n, x.d, x.o, copy(x.data))
unsafe_convert(::Type{Ptr{T}}, p::PhysicalParameter{T, N}) where {T<:Number, N} = unsafe_convert(Ptr{T}, p.data)

# Equality
==(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Number, N} = (A.data == B.data && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}; kwargs...) where {T<:Number, N} = (isapprox(A.data, B.data) && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter{T, N}, B::AbstractArray{T, N}; kwargs...) where {T<:Number, N} = isapprox(A.data, B)
isapprox(A::AbstractArray{T, N}, B::PhysicalParameter{T, N}; kwargs...) where {T<:Number, N} = isapprox(A, B.data)

# # Arithmetic operations
compare(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Number, N} =  (A.o == B.o &&  A.d == B.d &&  A.n == B.n)

function combine(op, A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Number, N}
    o = min.(A.o, B.o)
    sa = floor.(Int, (A.o .- o) ./ A.d) .+ 1
    ea = sa .+ A.n .- 1
    sb = floor.(Int, (B.o .- o) ./ B.d) .+ 1
    eb = sb .+ B.n .- 1
    mn = max.(ea, eb)
    out = zeros(T, mn)
    ia = [s:e for (s, e) in zip(sa, ea)]
    ib = [s:e for (s, e) in zip(sb, eb)]
    out[ia...] .= A.data
    broadcast!(op, view(out, ib...),  view(out, ib...), B.data)
    return PhysicalParameter{T, N}(mn, A.d, o, out)
end

for op in [:+, :-, :*, :/]
    @eval function $(op)(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Number, N}
        if compare(A, B)
            return PhysicalParameter(broadcast($(op), A.data, B.data), A)
        elseif A.d == B.d
            # same grid but difference origin/shape, merging
            return combine($(op), A, B)
        else
            throw(PhysicalParameterException("Incompatible grid"))
        end
    end
end

function *(A::Union{joMatrix, joLinearFunction, joLinearOperator, joCoreBlock}, p::PhysicalParameter{RDT, N}) where {RDT<:Number, N}
    @warn "JOLI linear operator, returning julia Array"
    return A*vec(p.data)
end

# Brodacsting
BroadcastStyle(::Type{<:PhysicalParameter}) = Broadcast.ArrayStyle{PhysicalParameter}()

function similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ArrayAndChar:
    A = find_pm(bc)
    # Use the char field of A to create the output
    ElType == Bool ? Ad = similar(Array{ElType}, axes(A.data)) : Ad = similar(A.data)
    PhysicalParameter(Ad, A.d, A.o)
end

similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}) = similar(bc, eltype(find_pm(bc)))

"`A = find_pm(As)` returns the first PhysicalParameter among the arguments."
find_pm(bc::Base.Broadcast.Broadcasted) = find_pm(bc.args)
find_pm(args::Tuple) = find_pm(find_pm(args[1]), Base.tail(args))
find_pm(x) = x
find_pm(::Tuple{}) = nothing
find_pm(a::PhysicalParameter, rest) = a
find_pm(::Any, rest) = find_pm(rest)

function materialize!(A::PhysicalParameter{T, N}, ev::PhysicalParameter{T, N}) where {T<:Number, N}
    compare(A, ev)
    A.data .= ev.data
    nothing
end

materialize!(A::PhysicalParameter{T, N}, B::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{PhysicalParameter}}) where {T<:Number, N} = materialize!(A, B.f(B.args...))
materialize!(A::PhysicalParameter{T, N}, B::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{N}}) where {T<:Number, N} = materialize!(A.data, reshape(materialize(B), A.n))
materialize!(A::AbstractArray{T, N}, B::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}) where {T<:Number, N} = materialize!(A, reshape(materialize(B).data, size(A)))

for op in [:+, :-, :*, :/, :\]
    @eval begin
        broadcasted(::typeof($op), A::PhysicalParameter{T, N}, B::DenseVector{T}) where {T<:Number, N} = PhysicalParameter{T, N}(A.n, A.d, A.o, materialize(broadcasted($(op), A.data, reshape(B, A.n))))
        broadcasted(::typeof($op), B::DenseVector{T}, A::PhysicalParameter{T, N}) where {T<:Number, N} = PhysicalParameter{T, N}(A.n, A.d, A.o, materialize(broadcasted($(op), reshape(B, A.n), A.data)))
        broadcasted(::typeof($op), A::PhysicalParameter{T, N}, B::DenseArray{T, N}) where {T<:Number, N} = PhysicalParameter{T, N}(A.n, A.d, A.o, materialize(broadcasted($(op), A.data, B)))
        broadcasted(::typeof($op), B::DenseArray{T, N}, A::PhysicalParameter{T, N}) where {T<:Number, N} = PhysicalParameter{T, N}(A.n, A.d, A.o, materialize(broadcasted($(op), B, A.data)))
        broadcasted(::typeof($op), A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Number, N} = $(op)(A, B)
    end
end

# Linalg Extras
mul!(x::PhysicalParameter{T, N}, F::Union{joAbstractLinearOperator, joLinearFunction, Array}, y::Array) where {T<:Number, N} = mul!(view(x.data, :), F, y)
mul!(x::PhysicalParameter{T, N}, F::Union{joAbstractLinearOperator, joLinearFunction, Array}, y::PhysicalParameter{T, N}) where {T<:Number, N} = mul!(view(x.data, :), F, view(y.data, :))
mul!(x::Array, F::Union{joAbstractLinearOperator, joLinearFunction, Array}, y::PhysicalParameter{T, N}) where {T<:Number, N} = mul!(x, F, y.data[1:end])

# For ploting
NpyArray(p::PhysicalParameter{T, N}, revdims::Bool) where {T<:Number, N} = NpyArray(p.data, revdims)

###################################################################################################
# Isotropic acoustic

"""
    Model
        n::NTuple{N, Int64}
        d::NTuple{N, Float32}
        o::NTuple{N, Float32}
        nb::Integer
        params::Dict
        rho::Array

Model structure for seismic velocity models.

`n`: number of gridpoints in (x,y,z) for 3D or (x,z) for 2D

`d`: grid spacing in (x,y,z) or (x,z) (in meters)

`o`: origin of coordinate system in (x,y,z) or (x,z) (in meters)

`nb`: number of absorbing boundary points in each direction

`params`: Physical parameters such has squared slowness, denisty or THomesne parameters

Constructor
===========

The parameters `n`, `d`, `o` and `m` are mandatory, whith `nb` and other physical parameters being optional input arguments.

    Model(n, d, o, m; nb=40, rho=1, epsilon=0, delta=0, theta=0, phi=0)

where

`m`: velocity model in slowness squared (s^2/km^2)

`epsilon`: Epsilon thomsen parameter ( between -1 and 1)

`delta`: Delta thomsen parameter ( between -1 and 1 and delta < epsilon)

`theta`: Anisotopy dip in radian

`phi`: Anisotropy asymuth in radian

`rho`: density (g / m^3)

"""
const ModelParam{T, N} = Union{T, PhysicalParameter{T, N}}

# Acoustic

struct IsoModel{T, N} <: AbstractModel{T, N}
    G::DiscreteGrid{T, N}
    m::ModelParam{T, N}
    rho::ModelParam{T, N}
end

# VTI/TTI
struct TTIModel{T, N} <: AbstractModel{T, N}
    G::DiscreteGrid{T, N}
    m::ModelParam{T, N}
    rho::ModelParam{T, N}
    epsilon::ModelParam{T, N}
    delta::ModelParam{T, N}
    theta::ModelParam{T, N}
    phi::ModelParam{T, N}
end

# Elastic

struct IsoElModel{T, N} <: AbstractModel{T, N}
    G::DiscreteGrid{T, N}
    lambda::ModelParam{T, N}
    mu::ModelParam{T, N}
    b::ModelParam{T, N}
end

# Visco-acoustic
struct ViscIsoModel{T, N} <: AbstractModel{T, N}
    G::DiscreteGrid{T, N}
    m::ModelParam{T, N}
    rho::ModelParam{T, N}
    qp::ModelParam{T, N}
end

_params(m::TTIModel) = [(:m, m.m), (:rho, m.rho), (:epsilon, m.epsilon), (:delta, m.delta), (:theta, m.theta), (:phi, m.phi)]
_params(m::IsoModel) = [(:m, m.m), (:rho, m.rho)]
_params(m::IsoElModel) = [(:lam, m.lambda), (:mu, m.mu), (:b, m.b)]
_params(m::IsoElModel) = [(:m, m.m), (:rho, m.rho), (:qp, m.qp)]

_mparams(m::AbstractModel) = first.(_params(m))

###################################################################################################
# Constructors
_scalar(::Nothing, ::Type{T}, def=1) where T = T(def)
_scalar(v::Number, ::Type{T}, def=1) where T = T(v)

function Model(n, d, o, m::Array{T, N}; epsilon=nothing, delta=nothing, theta=nothing,
               phi=nothing, rho=nothing, qp=nothing, vs=nothing, nb=40) where {T<:Number, N}

    # Convert dimension to internal types
    n = NTuple{length(n), Int64}(n)
    d = NTuple{length(n), Float32}(d)
    o = NTuple{length(n), Float32}(o)
    G = DiscreteGrid(n, d, o, nb)
    
    size(m) == n || throw(ArgumentError("Grid size $n and squared slowness size $(size(m)) don't match"))

    # Elastic
    if !isnothing(vs)
        rho = isa(rho, Array) ? rho : _scalar(rho, T)
        if any(!isnothing(p) for p in [epsilon, delta, theta, phi])
            @warn "Thomsen parameters no supported for elastic (vs) ignoring them"
        end
        lambda = PhysicalParameter(convert(Array{T, N}, (m.^(-1)- T(2) * vs.^2) .* rho), n, d, o)
        mu = PhysicalParameter(convert(Array{T, N}, vs.^2 .* rho), n, d, o)
        b = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, 1 ./ rho), n, d, o) : _scalar(rho, T)
        return IsoElModel{T, N}(G, lamda, mu, b)
    end

    ## Visco
    if !isnothing(qp)
        if any(!isnothing(p) for p in [epsilon, delta, theta, phi])
            @warn "Thomsen parameters no supported for elastic (vs) ignoring them"
        end
        qp = PhysicalParameter(convert(Array{T, N}, qp), n, d, o)
        m = PhysicalParameter(m, n, d, o)
        rho = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, rho), n, d, o) : _scalar(rho, T)
        return ViscIsoModel{T, N}(G, m, rho, qp)
    end

    ## TTI
    if !isnothing(qp)
        if any(!isnothing(p) for p in [vs, qp])
            @warn "Elastic (vs) and attenuation (qp) not supported for TTI/VTI"
        end
        m = PhysicalParameter(m, n, d, o)
        rho = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, rho), n, d, o) : _scalar(rho, T)
        epsilon = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, epsilon), n, d, o) : _scalar(epsilon, T, 0)
        delta = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, delta), n, d, o) : _scalar(delta, T, 0)
        theta = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, theta), n, d, o) : _scalar(theta, T, 0)
        phi = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, phi), n, d, o) : _scalar(phi, T, 0)
        return TTIModel{T, N}(G, m, rho, epsilon, delta, theta, phi)
    end

    # None of the advanced models, return isotropic acoustic
    m = PhysicalParameter(m, n, d, o)
    rho = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, rho), n, d, o) : _scalar(rho, T)
    return IsoModel{T, N}(G, m, rho)
end

Model(n, d, o, m::Array, rho::Array; nb=40) = Model(n, d, o, m; rho=rho, nb=nb)
Model(n, d, o, m::Array, rho::Array, qp::Array; nb=40) = Model(n, d, o, m; rho=rho, qp=qp, nb=nb)

size(m::AbstractModel) = m.G.n
origin(m::AbstractModel) = m.G.o
spacing(m::AbstractModel) = m.G.d

get_dt(m::AbstractModel; dt=nothing) = calculate_dt(m; dt=dt)

similar(::PhysicalParameter{T, N}, m::AbstractModel) where {T<:Number, N} = PhysicalParameter(size(m), spacing(m), origin(m); DT=T)
similar(x::Array, m::AbstractModel) = similar(x, size(m))

ndims(m::AbstractModel) = ndims(m.m.data)

_repr(m::AbstractModel) = "Model (n=$(size(m)), d=$(spacing(m)), o=$(origin(m))) with parameters $(_mparams(m))"
display(m::AbstractModel) = println(_repr(m))
show(io::IO, m::AbstractModel) = print(io, _repr(m))
show(io::IO, ::MIME{Symbol("text/plain")}, m::AbstractModel) = print(io, _repr(m))

# Pad gradient if aperture doesn't match full domain
_project_to_physical_domain(p, ::Any) = p

function _project_to_physical_domain(p::PhysicalParameter, model::AbstractModel)
    size(p) == size(model) && (return p)
    pp = similar(p, model)
    pp .+= p
    return pp
end
