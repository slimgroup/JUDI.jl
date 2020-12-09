# Model structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

const IntTuple = Union{Tuple{Integer,Integer}, Tuple{Integer,Integer,Integer},Array{Int64,1},Array{Int32,1}}
const RealTuple = Union{Tuple{Real,Real}, Tuple{Real,Real,Real},Array{Float64,1},Array{Float32,1}}

export Model, PhysicalParameter, get_dt


###################################################################################################
# PhysicalParameter abstract vector

"""
    PhysicalParameter
        n::IntTuple
        d::RealTuple
        o::RealTuple
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

    PhysicalParameter(v::Array{vDT}, d, o) where `v` is an n-dimensional array and n=size(v)

    PhysicalParameter(n, d, o; vDT=Float32) Creates a zero PhysicalParameter

    PhysicalParameter(v::Array{vDT}, A::PhysicalParameter) Creates a PhysicalParameter from the Array `v` with n, d, o from `A`

    PhysicalParameter(v::Array{vDT, N}, n::Tuple, d::Tuple, o::Tuple) where `v` is a vector or nd-array that is reshaped into shape `n`

    PhysicalParameter(v::vDT, n::Tuple, d::Tuple, o::Tuple) Creates a constant (single number) PhyicalParameter

"""
mutable struct PhysicalParameter{vDT} <: AbstractVector{vDT}
    n::Tuple
    d::Tuple
    o::Tuple
    data::Union{Array, vDT}
end

mutable struct PhysicalParameterException <: Exception
    msg :: String
end

PhysicalParameter(v::BitArray{N}, args...) where N = v

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

function PhysicalParameter(v::Array{vDT, N}, n::Tuple, d::Tuple, o::Tuple) where {vDT, N}
    length(v) != prod(n) && throw(PhysicalParameterException("Incompatible number of element in input $(length(v)) with n=$(n)"))
    N == 1 && (v = reshape(v, n))
    return PhysicalParameter{vDT}(n, d, o, v)
end

function PhysicalParameter(v::vDT, n::Tuple, d::Tuple, o::Tuple) where vDT<:Number
    return PhysicalParameter{vDT}(n, d, o, v)
end

PhysicalParameter(p::PhysicalParameter, n::Tuple, d::Tuple, o::Tuple) = p 
PhysicalParameter(p::PhysicalParameter) = p 

# transpose and such.....
conj(x::PhysicalParameter{vDT}) where vDT = x
transpose(x::PhysicalParameter{vDT}) where vDT = PhysicalParameter{vDT}(x.n[end:-1:1], x.d[end:-1:1], x.o[end:-1:1], permutedims(x.data, length(x.n):-1:1))
adjoint(x::PhysicalParameter{vDT}) where vDT = transpose(x)

# Basic overloads
size(A::PhysicalParameter) = (prod(A.n), 1)
length(A::PhysicalParameter) = prod(A.n)

function norm(A::PhysicalParameter, order::Real=2)
    return norm(vec(A.data), order)
end

dot(A::PhysicalParameter, B::PhysicalParameter) = dot(vec(A.data), vec(B.data))
dot(A::PhysicalParameter, B::Array) = dot(vec(A.data), vec(B))
dot(A::Array, B::PhysicalParameter) = dot(vec(A), vec(B.data))

display(A::PhysicalParameter) = println("$(typeof(A)) of size $(A.n) with origin $(A.o) and spacing $(A.d)")
show(A::PhysicalParameter) = show(A.data)
showarg(io::IO, A::PhysicalParameter, toplevel) = print(io, typeof(A), " with size $(A.n), spacing $(A.d) and origin $(A.o)")

# Indexing
firstindex(A::PhysicalParameter) = 1
lastindex(A::PhysicalParameter) = length(A)
lastindex(A::PhysicalParameter, dim::Int) = A.n[dim]

function promote_shape(p::PhysicalParameter, A::Array{vDT, N}) where {vDT, N}
    (size(A) != p.n && N>1) && return promote_shape(p.data, A)
    (length(A) == prod(p.n) && N==1) && return size(A)
    return promote_shape(A, A)
end

promote_shape(A::Array{vDT, N}, p::PhysicalParameter) where {vDT, N} = promote_shape(p, A)
reshape(p::PhysicalParameter, n::Tuple{Vararg{Int64,N}}) where N = (prod(n)==prod(p.n) && return p)

dotview(A::PhysicalParameter{vDT}, I::Vararg{Union{Function, Int, UnitRange{Int}}, N}) where {vDT, N} = dotview(A.data, I...)

getindex(A::PhysicalParameter, i::Int) = A.data[i]
getindex(A::PhysicalParameter, I::Vararg{Union{Int, BitArray, Function, UnitRange{Int}}, N}) where {N} = getindex(A.data, I...)
Base.dotview(m::PhysicalParameter, i) = Base.dotview(m.data, i)

setindex!(A::PhysicalParameter, v, I::Vararg{Union{Int, Function, UnitRange{Int}}, N}) where {N} = setindex!(A.data, v, I...)
setindex!(A::PhysicalParameter, v, i::Int) = (A.data[i] = v)

# Constructiors by copy
similar(x::PhysicalParameter{vDT}) where {vDT} = vDT(0) .* x
copy(x::PhysicalParameter{vDT}) where {vDT} = PhysicalParameter{vDT}(x.n, x.d, x.o, x.data)

# Equality
isequal(A::PhysicalParameter, B::PhysicalParameter) = (A.data == B.data && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter, B::PhysicalParameter; kwargs...) = (isapprox(A.data, B.data) && A.o == B.o && A.d == B.d)
isapprox(A::PhysicalParameter, B::AbstractArray; kwargs...) = isapprox(A.data, B)
isapprox(A::AbstractArray, B::PhysicalParameter; kwargs...) = isapprox(A, B.data)

# # Arithmetic operations
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

# Brodacsting
BroadcastStyle(::Type{<:PhysicalParameter}) = Broadcast.ArrayStyle{PhysicalParameter}()

function similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ArrayAndChar:
    A = find_aac(bc)
    # Use the char field of A to create the output
    ElType == Bool ? Ad = similar(Array{ElType}, axes(A.data)) : Ad = similar(A.data)
    PhysicalParameter(Ad, A.d, A.o)
end

"`A = find_aac(As)` returns the first PhysicalParameter among the arguments."
find_aac(bc::Base.Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(args::Tuple) = find_aac(find_aac(args[1]), Base.tail(args))
find_aac(x) = x
find_aac(::Tuple{}) = nothing
find_aac(a::PhysicalParameter, rest) = a
find_aac(::Any, rest) = find_aac(rest)

function broadcasted(f::Function, A::AbstractArray{vDT, N}, p::PhysicalParameter{RDT}) where {vDT, RDT, N}
    if size(A) != p.n
        if length(A) == length(p)
            return PhysicalParameter(materialize(broadcasted(f, A, vec(p.data))), p.n, p.d, p.o)
        end
        throw(PhysicalParameterException("Incompatible sizes $(size(A)) and $(p.n)"))
    end
    return PhysicalParameter(materialize(broadcasted(f, A, p.data)), p.d, p.o)
end

function broadcasted(f::Function, p::PhysicalParameter{RDT}, A::AbstractArray{vDT, N}) where {vDT, RDT, N}
    if size(A) != p.n
        if length(A) == length(p)
            return PhysicalParameter(materialize(broadcasted(f, vec(p.data), A)), p.n, p.d, p.o)
        end
        throw(PhysicalParameterException("Incompatible sizes $(size(A)) and $(p.n)"))
    end
    return PhysicalParameter(materialize(broadcasted(f, p.data, A)), p.d, p.o)
end

function broadcasted(f::Function, p1::PhysicalParameter{RDT}, p2::PhysicalParameter{RDT}) where {RDT}
    p1.n != p2.n && throw(PhysicalParameterException("Incompatible sizes $(p1.n) and $(p2.n)"))
    return PhysicalParameter(materialize(broadcasted(f, p1.data, p2.data)), p1.d, p1.o)
end

broadcasted(f::Function, p::PhysicalParameter, bc::Base.Broadcast.Broadcasted) = broadcasted(f, p, materialize(bc))
broadcasted(f::Function, bc::Base.Broadcast.Broadcasted, p::PhysicalParameter) = broadcasted(f, materialize(bc), p)

function *(A::Union{joMatrix, joLinearFunction, joLinearOperator, joCoreBlock}, p::PhysicalParameter{RDT}) where {RDT}
    @warn "JOLI linear operator, returning julia Array"
    return A*vec(p.data)
end

materialize!(p::PhysicalParameter{RDT}, b::Array{<:Number, N}) where{RDT, N} = (p.data[:] .= b)
materialize!(p::PhysicalParameter{RDT}, b::Base.Broadcast.Broadcasted{Base.Broadcast.DefaultArrayStyle{N}}) where{RDT, N} = materialize!(p, collect(b))
materialize!(p::Array, b::Broadcast.Broadcasted{Broadcast.ArrayStyle{PhysicalParameter}}) = materialize!(p, reshape(collect(b), size(p)))

# Linalg Extras
mul!(x::PhysicalParameter, F::Union{joAbstractLinearOperator, joLinearFunction, Array}, y::Array) = mul!(view(x.data, :), F, y)
mul!(x::PhysicalParameter, F::Union{joAbstractLinearOperator, joLinearFunction, Array}, y::PhysicalParameter) = mul!(view(x.data, :), F, view(y.data, :))
mul!(x::Array, F::Union{joAbstractLinearOperator, joLinearFunction, Array}, y::PhysicalParameter) = mul!(x, F, y[1:end])

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

    params = Dict(:m => PhysicalParameter(Float32.(m), d, o))
    for (name, val) in zip([:rho, :epsilon, :delta, :theta, :phi], [rho, epsilon, delta, theta, phi])
        ~isnothing(val) && (params[name] = PhysicalParameter(Float32.(val), n, d, o))
    end
    
    return Model(n, d, o, nb, params)
end

function Model(n::IntTuple, d::RealTuple, o::RealTuple, m, rho; nb=40)
    return Model(n, d, o, m; rho=rho, nb=nb)
end

get_dt(m::Model) = calculate_dt(m)
getindex(m::Model, sym::Symbol) = m.params[sym]

Base.setproperty!(m::Model, s::Symbol, p::PhysicalParameter{Float32}) = (m.params[s] = p)

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
