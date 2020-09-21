# Model structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

const IntTuple = Union{Tuple{Integer,Integer}, Tuple{Integer,Integer,Integer},Array{Int64,1},Array{Int32,1}}
const RealTuple = Union{Tuple{Real,Real}, Tuple{Real,Real,Real},Array{Float64,1},Array{Float32,1}}

export Model, PhysicalParameter, get_dt


###################################################################################################
# PhysicalParameter abstract vector

mutable struct PhysicalParameter{vDT} <: AbstractVector{vDT}
    n::Tuple
    d::Tuple
    o::Tuple
    data::Union{Array, vDT}
end

mutable struct PhysicalParameterException <: Exception
    msg :: String
end

function PhysicalParameter(v::Array{vDT}, d::Tuple, o::Tuple) where {vDT}
    length(size(v)) != length(o) && throw(PhysicalParameterException("Input array should be $(length(o))-dimensional"))
    return PhysicalParameter{vDT}(size(v), d, o, v)
end

function PhysicalParameter(n::Tuple, d::Tuple, o::Tuple; vDT=Float32)
    return PhysicalParameter{vDT}(n, d, o, zeros(vDT, n))
end

function PhysicalParameter(v::Array{vDT}, A::PhysicalParameter) where {vDT}
    return PhysicalParameter{vDT}(A.n, A.d, A.o, v)
end

function PhysicalParameter(v::Union{Array{vDT}, vDT}, n::Tuple, d::Tuple, o::Tuple) where {vDT}
    return PhysicalParameter{vDT}(n, d, o, v)
end

# transpose and such.....
conj(x::PhysicalParameter{vDT}) where vDT = x
transpose(x::PhysicalParameter{vDT}) where vDT = x
adjoint(x::PhysicalParameter{vDT}) where vDT = x

# Basic overloads
function norm(A::PhysicalParameter, order::Real=2)
    return norm(vec(A.data), order)
end

dot(A::PhysicalParameter, B::PhysicalParameter) = dot(vec(A.data), vec(B.data))
dot(A::PhysicalParameter, B::Array) = dot(vec(A.data), vec(B))
dot(A::Array, B::PhysicalParameter) = dot(vec(A), vec(B.data))

display(A::PhysicalParameter) = println("$(typeof(A)) of size $(A.n) with origin $(A.o) and spacing $(A.d)")
show(A::PhysicalParameter) = show(A.data)

size(A::PhysicalParameter) = (prod(A.n), 1)
getindex(A::PhysicalParameter, i::Int) = A.data[i]
getindex(A::PhysicalParameter, I::Vararg{Union{Int, UnitRange{Int}}, N}) where {N} = getindex(A.data, I...)

setindex!(A::PhysicalParameter, v, I::Vararg{Union{Int, UnitRange{Int}}, N}) where {N} = setindex!(A.data, v, I...)
setindex!(A::PhysicalParameter, v, i::Int) = (A.data[i] = v)

similar(x::PhysicalParameter{vDT}) where {vDT} = vDT(0) .* x
copy(x::PhysicalParameter{vDT}) where {vDT} = PhysicalParameter{vDT}(x.n, x.d, x.o, x.data)

isequal(A::PhysicalParameter, B::PhysicalParameter) = (A.data == B.data && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter, B::PhysicalParameter; kwargs...) = (isapprox(A.data, B.data) && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter, B::AbstractArray; kwargs...) = isapprox(A.data, B)
isapprox(A::AbstractArray, B::PhysicalParameter; kwargs...) = isapprox(A, B.data)

# Arithmetic operations
function compare(A::PhysicalParameter, B::PhysicalParameter)
    A.o != B.o && throw(PhysicalParameterException("Incompatible origins $(A.o) and $(B.o)"))
    A.d != B.d && throw(PhysicalParameterException("Incompatible spacing $(A.d) and $(B.d)"))
    A.n != B.n && throw(PhysicalParameterException("Incompatible sizes $(A.n) and $(B.n)"))
end

function +(A::PhysicalParameter{vDT}, B::PhysicalParameter{vDT}) where {vDT}
    compare(A, B)
    return PhysicalParameter(A.data + B.data, A)
end

function -(A::PhysicalParameter{vDT}, B::PhysicalParameter{vDT}) where {vDT}
    compare(A, B)
    return PhysicalParameter(A.data - B.data, A)
end

function *(A::PhysicalParameter{vDT}, B::PhysicalParameter{vDT}) where {vDT}
    compare(A, B)
    return PhysicalParameter(A.data .* B.data, A)
end

function /(A::PhysicalParameter{vDT}, B::PhysicalParameter{vDT}) where {vDT}
    compare(A, B)
    return PhysicalParameter(A.data ./ B.data, A)
end


+(A::PhysicalParameter{vDT}, b::Number) where {vDT} = PhysicalParameter(A.data .+ b, A)
+(b::Number, A::PhysicalParameter{vDT}) where {vDT} = PhysicalParameter(A.data .+ b, A)
-(A::PhysicalParameter{vDT}, b::Number) where {vDT} = PhysicalParameter(A.data .- b, A)
-(A::PhysicalParameter{vDT}) where {vDT} = PhysicalParameter(-A.data, A)
-(b::Number, A::PhysicalParameter{vDT}) where {vDT} = PhysicalParameter(b .- A.data, A)
*(A::PhysicalParameter{vDT}, b::Number) where {vDT} = PhysicalParameter(A.data .* b, A)
*(b::Number, A::PhysicalParameter{vDT}) where {vDT} = PhysicalParameter(A.data .* b, A)
/(A::PhysicalParameter{vDT}, b::Number) where {vDT} = PhysicalParameter(A.data ./ b, A)
/(b::Number, A::PhysicalParameter{vDT}) where {vDT} = PhysicalParameter(b ./ A.data, A)


# Brodacsting
BroadcastStyle(::Type{PhysicalParameter}) = Base.Broadcast.DefaultArrayStyle{1}()
ndims(::Type{PhysicalParameter{vDT}}) where {vDT} = 1

### +/- ####
broadcasted(::typeof(+), x::PhysicalParameter, y::PhysicalParameter) = x + y
broadcasted(::typeof(+), x::PhysicalParameter, y::Number) = x + y
broadcasted(::typeof(+), x::Number, y::PhysicalParameter) = x + y
broadcasted(::typeof(-), x::PhysicalParameter, y::PhysicalParameter) = x - y
broadcasted(::typeof(-), x::PhysicalParameter, y::Number) = x - y
broadcasted(::typeof(-), x::Number, y::PhysicalParameter) = x - y
broadcasted(::typeof(*), x::PhysicalParameter, y::PhysicalParameter) = x * y
broadcasted(::typeof(*), x::PhysicalParameter, y::Number) = x * y
broadcasted(::typeof(*), x::Number, y::PhysicalParameter) = x * y
broadcasted(::typeof(/), x::PhysicalParameter, y::PhysicalParameter) = x / y
broadcasted(::typeof(/), x::PhysicalParameter, y::Number) = x / y
broadcasted(::typeof(/), x::Number, y::PhysicalParameter) = x / y

broadcast!(identity, x::PhysicalParameter, y::PhysicalParameter) = copy!(x, y)
broadcasted(identity, x::PhysicalParameter) = x

# Materialize for broadcasting
function materialize!(x::PhysicalParameter, y::PhysicalParameter)
    x.data .= y.data
    x.d = y.d
    x.o = y.o
    x.n = y.n
end

function copy!(x::PhysicalParameter, y::PhysicalParameter)
    compare(x, y)
    x.data .= y.data
    x.d = y.d
    x.o = y.o
    x.n = y.n
end

# For ploting
array2py(p::PhysicalParameter{vDT}, i::Int64, I::CartesianIndex{N}) where {vDT, N} = array2py(p.data, i, I)

###################################################################################################
# Isotropic acoustic

"""
    Model
        n::IntTuple
        d::RealTuple
        o::RealTuple
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
mutable struct Model
    n::IntTuple
    d::RealTuple
    o::RealTuple
    nb::Integer # number of absorbing boundaries points on each side
    params::Dict
end

###################################################################################################
# Constructors

function Model(n::IntTuple, d::RealTuple, o::RealTuple, m;
               epsilon=nothing, delta=nothing, theta=nothing,
               phi=nothing, rho=nothing, nb=40)

    params = Dict(:m => PhysicalParameter(m, d, o))
    for (name, val) in zip([:rho, :epsilon, :delta, :theta, :phi], [rho, epsilon, delta, theta, phi])
        ~isnothing(val) && (params[name] = PhysicalParameter(val, n, d, o))
    end
    
    return Model(n, d, o, nb, params)
end

function Model(n::IntTuple, d::RealTuple, o::RealTuple, m, rho; nb=40)
    return Model(n, d, o, m; rho=rho, nb=nb)
end

get_dt(m::Model) = calculate_dt(m)
getindex(m::Model, sym::Symbol) = m.params[sym]

function Base.getproperty(obj::Model, sym::Symbol)
    if sym == :params
        return getfield(obj, sym)
    elseif sym in keys(obj.params)
        return obj.params[sym]
    else
         # fallback to getfield
        return getfield(obj, sym)
    end
end
