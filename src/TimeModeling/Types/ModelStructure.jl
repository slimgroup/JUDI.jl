# Model structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
# Author Mathias Louboutin
# Date: 2017-2022
#
export Model, PhysicalParameter, get_dt, size, origin, spacing, nbl

abstract type AbstractModel{T, N} end

struct DiscreteGrid{T<:Real, N}
    n::NTuple{N, Int64}
    d::NTuple{N, <:T}
    o::NTuple{N, <:T}
    nb::Int64 # number of absorbing boundaries points on each side
end

size(G::DiscreteGrid) = G.n
origin(G::DiscreteGrid) = G.o
spacing(G::DiscreteGrid) = G.d
nbl(G::DiscreteGrid) = G.nb

size(G::DiscreteGrid, i::Int) = G.n[i]
origin(G::DiscreteGrid, i::Int) = G.o[i]
spacing(G::DiscreteGrid, i::Int) = G.d[i]

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

    PhysicalParameter(v::Array{T}, A::PhysicalParameter{T, N}) where {T<:Real, N} Creates a PhysicalParameter from the Array `v` with n, d, o from `A`

    PhysicalParameter(v::Array{T, N}, n::NTuple{N, T}, d::NTuple{N, T}, o::Tuple) where `v` is a vector or nd-array that is reshaped into shape `n`

    PhysicalParameter(v::T, n::NTuple{N, T}, d::NTuple{N, T}, o::Tuple) Creates a constant (single number) PhyicalParameter

"""
mutable struct PhysicalParameter{T, N} <: DenseArray{T, N}
    n::NTuple{N, Int64}
    d::NTuple{N, T}
    o::NTuple{N, T}
    data::Union{Array{T, N}, T}
    PhysicalParameter(n::NTuple{N, <:Number}, d::NTuple{N, <:Number}, o::NTuple{N, <:Number}, data::Union{Array{T, N}, T}) where {T, N} = 
        new{T, N}(Int.(n), T.(d), T.(o), data)
end

mutable struct PhysicalParameterException <: Exception
    msg :: String
end

PhysicalParameter(v::BitArray{N}, args...) where N = v
PhysicalParameter(v::Array{Bool, N}, ::NTuple, ::NTuple) where N = v

function PhysicalParameter(v::Array{T, N}, d::NTuple, o::NTuple) where {T<:Real, N}
    n = size(v)
    length(n) != length(o) && throw(PhysicalParameterException("Input array should be $(length(o))-dimensional"))
    return PhysicalParameter(n, d, o, v)
end

PhysicalParameter(n::NTuple{N}, d::NTuple{N}, o::NTuple{N}; DT::Type{dT}=eltype(d)) where {dT<:Real, N} = PhysicalParameter(n, d, o, zeros(dT, n))
PhysicalParameter(v::Array{T, N}, A::PhysicalParameter{ADT, N}) where {T<:Real, ADT<:Real, N} = PhysicalParameter(A.n, A.d, A.o, reshape(v, A.n))

function PhysicalParameter(v::Array{T, N1}, n::NTuple{N, Int}, d::NTuple{N, T}, o::NTuple{N, T}) where {T<:Real, N, N1}
    length(v) != prod(n) && throw(PhysicalParameterException("Incompatible number of element in input $(length(v)) with n=$(n)"))
    N1 == 1 && (v = reshape(v, n))
    return PhysicalParameter(n, d, o, v)
end

PhysicalParameter(v::Real, n::NTuple{N}, d::NTuple{N}, o::NTuple{N}) where {N} = PhysicalParameter(n, d, o, v)
PhysicalParameter(v::Integer, n::NTuple{N}, d::NTuple{N}, o::NTuple{N}) where {N} = PhysicalParameter(n, d, o, Float32.(v))
PhysicalParameter(v::Array{T, N}, n::NTuple{N}, d::NTuple{N}, o::NTuple{N}) where {T<:Real, N} = PhysicalParameter(n, d, o, v)

PhysicalParameter(p::PhysicalParameter{T, N}, ::NTuple{N, Int}, ::NTuple{N, T}, ::NTuple{N, T}) where {T<:Real, N} = p 
PhysicalParameter(p::PhysicalParameter{T, N}) where {T<:Real, N} = p 
PhysicalParameter(p::PhysicalParameter{T, N}, v::Array{T, Nv}) where {T<:Real, N, Nv} = PhysicalParameter(p.n, p.d, p.o, reshape(v, p.n))

# transpose and such.....
conj(x::PhysicalParameter{T, N}) where {T<:Real, N} = x
transpose(x::PhysicalParameter{T, N}) where {T<:Real, N} = PhysicalParameter(x.n[N:-1:1], x.d[N:-1:1], x.o[N:-1:1], permutedims(x.data, N:-1:1))
adjoint(x::PhysicalParameter{T, N}) where {T<:Real, N} = transpose(x)

# Basic overloads
size(A::PhysicalParameter{T, N}) where {T<:Real, N} = A.n
length(A::PhysicalParameter{T, N}) where {T<:Real, N} = prod(A.n)

function norm(A::PhysicalParameter{T, N}, order::Real=2) where {T<:Real, N}
    return norm(vec(A.data), order)
end

dot(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N} = dot(vec(A.data), vec(B.data))
dot(A::PhysicalParameter{T, N}, B::Array{T, N}) where {T<:Real, N} = dot(vec(A.data), vec(B))
dot(A::Array{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N} = dot(vec(A), vec(B.data))

_repr(A::PhysicalParameter{T, N}) where {T<:Real, N} = "$(typeof(A)) of size $(A.n) with origin $(A.o) and spacing $(A.d)"
display(A::PhysicalParameter{T, N}) where {T<:Real, N} = println(_repr(A))
show(io::IO, A::PhysicalParameter{T, N}) where {T<:Real, N} = print(io, _repr(A))
summary(io::IO, A::PhysicalParameter{T, N}) where {T<:Real, N} = print(io, _repr(A))
showarg(io::IO, A::PhysicalParameter, toplevel) = print(io, _repr(A))
show(io::IO, ::MIME{Symbol("text/plain")}, A::PhysicalParameter{T, N}) where {T<:Real, N} = println(io, _repr(A))

# Indexing
firstindex(A::PhysicalParameter{T, N}) where {T<:Real, N} = 1
lastindex(A::PhysicalParameter{T, N}) where {T<:Real, N} = length(A)
lastindex(A::PhysicalParameter{T, N}, dim::Int) where {T<:Real, N} = A.n[dim]

function promote_shape(p::PhysicalParameter{T, N}, A::Array{T, Na}) where {T, N, Na}
    (size(A) != p.n && N>1) && return promote_shape(p.data, A)
    (length(A) == prod(p.n) && N==1) && return size(A)
    return promote_shape(A, A)
end

promote_shape(A::Array{T, Na}, p::PhysicalParameter{T, N}) where {T<:Real, N, Na}  = promote_shape(p, A)
reshape(p::PhysicalParameter{T, N}, n::Tuple{Vararg{Int64,N}}) where {T<:Real, N} = (n == p.n ? p : reshape(p.data, n))
reshape(p::PhysicalParameter, pr::PhysicalParameter) = (p.n == pr.n ? p : throw(ArgumentError("Incompatible PhysicalParameter sizes ($(p.n), $(po.n))")))

dotview(m::PhysicalParameter, i) = Base.dotview(m.data, i)

getindex(A::PhysicalParameter{T, N}, i::Int) where {T<:Real, N} = A.data[i]
getindex(A::PhysicalParameter{T, N}, ::Colon) where {T<:Real, N} = A.data[:]

elsize(A::PhysicalParameter)  = elsize(A.data)

get_step(r::StepRange) = r.step
get_step(r) = 1

function getindex(A::PhysicalParameter{T, N}, I::Vararg{Union{Int, BitArray, Function, StepRange{Int}, UnitRange{Int}}, Ni}) where {N, Ni, T<:Real}
    new_v = getindex(A.data, I...)
    length(size(new_v)) != length(A.n) && (return new_v)
    s = [i == Colon() ? 0 : i[1]-1 for i=I]
    st = [get_step(i) for i=I]
    new_o = [ao+i*d for (ao, i, d)=zip(A.o, s, A.d)]
    new_d = [d*s for (d, s)=zip(A.d, st)]
    PhysicalParameter(size(new_v), tuple(new_d...), tuple(new_o...), new_v)
end

setindex!(A::PhysicalParameter{T, N}, v, I::Vararg{Union{Int, Function, UnitRange{Int}}, Ni}) where {T<:Real, N, Ni}  = setindex!(A.data, v, I...)
setindex!(A::PhysicalParameter{T, N}, v, i::Int) where {T<:Real, N} = (A.data[i] = v)

# Constructiors by copy
similar(x::PhysicalParameter{T, N}) where {T<:Real, N} = PhysicalParameter(x.n, x.d, x.o, fill!(similar(x.data), 0))

function similar(p::PhysicalParameter{T, N}, ::Type{nT}, s::AbstractSize) where {T, N, nT}
    nn = tuple(last.(sort(collect(s.dims), by = x->x[1]))...)
    return PhysicalParameter(nn, p.d, p.o, zeros(nT, nn))
end

copy(x::PhysicalParameter{T, N}) where {T<:Real, N} = PhysicalParameter(x.n, x.d, x.o, copy(x.data))
unsafe_convert(::Type{Ptr{T}}, p::PhysicalParameter{T, N}) where {T<:Real, N} = unsafe_convert(Ptr{T}, p.data)
Base.Vector{T}(m::PhysicalParameter) where T = Vector{T}(m.data[:])

# Equality
==(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N} = (A.data == B.data && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}; kwargs...) where {T<:Real, N} = (isapprox(A.data, B.data) && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter{T, N}, B::AbstractArray{T, N}; kwargs...) where {T<:Real, N} = isapprox(A.data, B)
isapprox(A::AbstractArray{T, N}, B::PhysicalParameter{T, N}; kwargs...) where {T<:Real, N} = isapprox(A, B.data)

# # Arithmetic operations
compare(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N} =  (A.o == B.o &&  A.d == B.d &&  A.n == B.n)


function combine(op, A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N}
    A.d == B.d || throw(PhysicalParameterException("Incompatible grids: ($(A.d), $(B.d))"))
    o = min.(A.o, B.o)
    sa = floor.(Int, (A.o .- o) ./ A.d) .+ 1
    ea = sa .+ A.n .- 1
    sb = floor.(Int, (B.o .- o) ./ B.d) .+ 1
    eb = sb .+ B.n .- 1
    mn = max.(ea, eb)
    ia = [s:e for (s, e) in zip(sa, ea)]
    ib = [s:e for (s, e) in zip(sb, eb)]
    if isnothing(op)
        @assert A.n == mn
        A.data[ib...] .= B.data
        return nothing
    else
        out = zeros(T, mn)
        out[ia...] .= A.data
        broadcast!(op, view(out, ib...),  view(out, ib...), B.data)
        return PhysicalParameter(mn, A.d, o, out)
    end
end

combine!(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N} = combine(nothing, A, B)

for op in [:+, :-, :*, :/]
    @eval function $(op)(A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N}
        if compare(A, B)
            return PhysicalParameter(broadcast($(op), A.data, B.data), A)
        elseif A.d == B.d
            # same grid but difference origin/shape, merging
            return combine($(op), A, B)
        else
            throw(PhysicalParameterException("Incompatible grids: ($(A.d), $(B.d))"))
        end
    end
    @eval $(op)(A::PhysicalParameter{T, N}, B::T2) where {T<:Real, T2<:Number, N} = PhysicalParameter(broadcast($(op), A.data, T(B)), A)
    @eval $(op)(A::T2, B::PhysicalParameter{T, N}) where {T<:Real, T2<:Number, N} = PhysicalParameter(broadcast($(op), T(A), B.data), B)
end

function *(A::Union{joMatrix, joLinearFunction, joLinearOperator, joCoreBlock}, p::PhysicalParameter{RDT, N}) where {RDT<:Real, N}
    @warn "JOLI linear operator, returning julia Array"
    return A*vec(p.data)
end

# Brodacsting
BroadcastStyle(::Type{<:PhysicalParameter}) = Broadcast.ArrayStyle{PhysicalParameter}()

function similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}, ::Type{ElType}) where ElType
    # Scan the inputs
    A = find_bc(bc, PhysicalParameter)
    # Use the char field of A to create the output
    newT = ElType <: Nothing ? eltype(A) : ElType
    Ad = zeros(newT, axes(A.data))
    PhysicalParameter(Ad, A.d, A.o)
end

similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}) = similar(bc, nothing)

function materialize!(A::PhysicalParameter{T, N}, ev::PhysicalParameter{T, N}) where {T<:Real, N}
    if compare(A, ev)
        A.data .= ev.data
    else
        A.n = ev.n
        A.d = ev.d
        A.o = ev.o
        A.data = copy(ev.data)
    end
    nothing
end

materialize!(A::PhysicalParameter{T, N}, B::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{PhysicalParameter}}) where {T<:Real, N} = materialize!(A, B.f(B.args...))
materialize!(A::PhysicalParameter{T, N}, B::Broadcast.Broadcasted{Broadcast.DefaultArrayStyle{Na}}) where {T<:Real, N, Na} = materialize!(A.data, reshape(materialize(B), A.n))
materialize!(A::AbstractArray{T, N}, B::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}) where {T<:Real, N} = materialize!(A, reshape(materialize(B).data, size(A)))

for op in [:+, :-, :*, :/, :\]
    @eval begin
        broadcasted(::typeof($op), A::PhysicalParameter{T, N}, B::DenseVector{T}) where {T<:Real, N} = PhysicalParameter(A.n, A.d, A.o, materialize(broadcasted($(op), A.data, reshape(B, A.n))))
        broadcasted(::typeof($op), B::DenseVector{T}, A::PhysicalParameter{T, N}) where {T<:Real, N} = PhysicalParameter(A.n, A.d, A.o, materialize(broadcasted($(op), reshape(B, A.n), A.data)))
        broadcasted(::typeof($op), A::PhysicalParameter{T, N}, B::DenseArray{T, N}) where {T<:Real, N} = PhysicalParameter(A.n, A.d, A.o, materialize(broadcasted($(op), A.data, B)))
        broadcasted(::typeof($op), B::DenseArray{T, N}, A::PhysicalParameter{T, N}) where {T<:Real, N} = PhysicalParameter(A.n, A.d, A.o, materialize(broadcasted($(op), B, A.data)))
        broadcasted(::typeof($op), A::PhysicalParameter{T, N}, B::PhysicalParameter{T, N}) where {T<:Real, N} = $(op)(A, B)
        broadcasted(::typeof($op), A::PhysicalParameter{T, N}, B::T2) where {T<:Real, T2<:Number, N} = PhysicalParameter(broadcast($(op), A.data, T(B)), A)
        broadcasted(::typeof($op), A::T2, B::PhysicalParameter{T, N}) where {T<:Real, T2<:Number, N} = PhysicalParameter(broadcast($(op), T(A), B.data), B)
    end
end

# For ploting
NpyArray(p::PhysicalParameter{T, N}, revdims::Bool) where {T<:Real, N} = NpyArray(p.data, revdims)

###################################################################################################
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
    lam::ModelParam{T, N}
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

_params(m::IsoModel) = ((:m, m.m), (:rho, m.rho))
_params(m::TTIModel) = ((:m, m.m), (:rho, m.rho), (:epsilon, m.epsilon), (:delta, m.delta), (:theta, m.theta), (:phi, m.phi))
_params(m::IsoElModel) = ((:lam, m.lam), (:mu, m.mu), (:b, m.b))
_params(m::ViscIsoModel) = ((:m, m.m), (:rho, m.rho), (:qp, m.qp))

_mparams(m::AbstractModel) = first.(_params(m))

###################################################################################################
# Constructors
_scalar(::Nothing, ::Type{T}, def=1) where T = T(def)
_scalar(v::Number, ::Type{T}, def=1) where T = T(v)

"""
    Model(n, d, o, m; epsilon=nothing, delta=nothing, theta=nothing,
            phi=nothing, rho=nothing, qp=nothing, vs=nothing, nb=40)


The parameters `n`, `d`, `o` and `m` are mandatory, whith `nb` and other physical parameters being optional input arguments.

where

`m`: velocity model in slowness squared (s^2/km^2)

`epsilon`: Epsilon thomsen parameter ( between -1 and 1)

`delta`: Delta thomsen parameter ( between -1 and 1 and delta < epsilon)

`theta`: Anisotopy dip in radian

`phi`: Anisotropy asymuth in radian

`rho`: density (g / m^3)

`qp`: P-wave attenuation for visco-acoustic models

`vs`: S-wave velocity for elastic models.

`nb`: Number of ABC points

"""
function Model(d, o, m::Array{mT, N}; epsilon=nothing, delta=nothing, theta=nothing,
               phi=nothing, rho=nothing, qp=nothing, vs=nothing, nb=40) where {mT<:Real, N}

    # Currently force single precision
    m = convert(Array{Float32, N}, m)
    T = Float32
    # Convert dimension to internal types
    n = size(m)
    d = tuple(Float32.(d)...)
    o = tuple(Float32.(o)...)
    G = DiscreteGrid{T, N}(n, d, o, nb)

    size(m) == n || throw(ArgumentError("Grid size $n and squared slowness size $(size(m)) don't match"))

    # Elastic
    if !isnothing(vs)
        rho = isa(rho, Array) ? rho : _scalar(rho, T)
        if any(!isnothing(p) for p in [epsilon, delta, theta, phi])
            @warn "Thomsen parameters no supported for elastic (vs) ignoring them"
        end
        lambda = PhysicalParameter(convert(Array{T, N}, (m.^(-1) .- T(2) .* vs.^2) .* rho), n, d, o)
        mu = PhysicalParameter(convert(Array{T, N}, vs.^2 .* rho), n, d, o)
        b = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, 1 ./ rho), n, d, o) : _scalar(rho, T)
        return IsoElModel{T, N}(G, lambda, mu, b)
    end

    ## Visco
    if !isnothing(qp)
        if any(!isnothing(p) for p in [epsilon, delta, theta, phi])
            @warn "Thomsen parameters no supported for elastic (vs) ignoring them"
        end
        qp = isa(qp, Array) ? PhysicalParameter(convert(Array{T, N}, qp), n, d, o)  : _scalar(qp, T)
        m = PhysicalParameter(m, n, d, o)
        rho = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, rho), n, d, o) : _scalar(rho, T)
        return ViscIsoModel{T, N}(G, m, rho, qp)
    end

    ## TTI
    if !isnothing(epsilon) || !isnothing(delta) || !isnothing(theta) || !isnothing(phi)
        if any(!isnothing(p) for p in [vs, qp])
            @warn "Elastic (vs) and attenuation (qp) not supported for TTI/VTI"
        end
        m = PhysicalParameter(m, n, d, o)
        rho = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, rho), n, d, o) : _scalar(rho, T)
        epsilon = isa(epsilon, Array) ? PhysicalParameter(convert(Array{T, N}, epsilon), n, d, o) : _scalar(epsilon, T, 0)
        delta = isa(delta, Array) ? PhysicalParameter(convert(Array{T, N}, delta), n, d, o) : _scalar(delta, T, 0)
        # For safety remove delta values unsupported (delta > epsilon)
        _clip_delta!(delta, epsilon)
        theta = isa(theta, Array) ? PhysicalParameter(convert(Array{T, N}, theta), n, d, o) : _scalar(theta, T, 0)
        phi = isa(phi, Array) ? PhysicalParameter(convert(Array{T, N}, phi), n, d, o) : _scalar(phi, T, 0)
        return TTIModel{T, N}(G, m, rho, epsilon, delta, theta, phi)
    end

    # None of the advanced models, return isotropic acoustic
    m = PhysicalParameter(m, n, d, o)
    rho = isa(rho, Array) ? PhysicalParameter(convert(Array{T, N}, rho), n, d, o) : _scalar(rho, T)
    return IsoModel{T, N}(G, m, rho)
end

Model(n, d, o, m::Array, rho::Array; nb=40) = Model(d, o, reshape(m, n...); rho=reshape(rho, n...), nb=nb)
Model(n, d, o, m::Array, rho::Array, qp::Array; nb=40) = Model(d, o, reshape(m, n...); rho=reshape(rho, n...), qp=reshape(qp, n...), nb=nb)
Model(n, d, o, m::Array; kw...) = Model(d, o, reshape(m, n...); kw...)

size(m::MT) where {MT<:AbstractModel} = size(m.G)
origin(m::MT) where {MT<:AbstractModel} = origin(m.G)
spacing(m::MT) where {MT<:AbstractModel} = spacing(m.G)
nbl(m::MT) where {MT<:AbstractModel} = nbl(m.G)

size(m::MT, i::Int) where {MT<:AbstractModel} = size(m.G, i)
origin(m::MT, i::Int) where {MT<:AbstractModel} = origin(m.G, i)
spacing(m::MT, i::Int) where {MT<:AbstractModel} = spacing(m.G, i)

eltype(::AbstractModel{T, N}) where {T, N} = T

get_dt(m::AbstractModel; dt=nothing) = calculate_dt(m; dt=dt)

similar(::PhysicalParameter{T, N}, m::AbstractModel) where {T<:Real, N} = PhysicalParameter(size(m), spacing(m), origin(m); DT=T)
similar(x::Array, m::AbstractModel) = similar(x, size(m))

PhysicalParameter(p::AbstractArray{T, N}, m::AbstractModel{T, N}) where {T, N} = PhysicalParameter(p, spacing(m), origin(m))

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

# Utils
_clip_delta!(::T, epsilon) where {T<:Number} = nothing
_clip_delta!(delta::PhysicalParameter{T}, epsilon::T) where {T<:Number} = delta.data[delta.data .>= epsilon] .= T(.99) * epsilon
_clip_delta!(delta::PhysicalParameter{T}, epsilon::PhysicalParameter{T}) where {T<:Number} = delta.data[delta.data .>= epsilon.data] .= T(.99) * epsilon[delta.data .>= epsilon.data]