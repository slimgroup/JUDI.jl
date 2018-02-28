# Model structure 
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

const IntTuple = Union{Tuple{Integer,Integer}, Tuple{Integer,Integer,Integer},Array{Int64,1},Array{Int32,1}}
const RealTuple = Union{Tuple{Real,Real}, Tuple{Real,Real,Real},Array{Float64,1},Array{Float32,1}}

export Model

# Object for velocity/slowness models
mutable struct Model
    n::IntTuple
    d::RealTuple
    o::RealTuple
    nb::Integer # number of absorbing boundaries points on each side
    m   # slowness squared
end


"""
    Model
        n::IntTuple
        d::RealTuple
        o::RealTuple
        nb::Integer
        m::Array
      
Model structure for seismic velocity models. 

`n`: number of gridpoints in (x,y,z) for 3D or (x,z) for 2D

`d`: grid spacing in (x,y,z) or (x,z) (in meters)

`o`: origin of coordinate system in (x,y,z) or (x,z) (in meters)

`nb`: number of absorbing boundary points in each direction

`m`: velocity model in slowness squared (s^2/km^2)


Constructor
===========

The parameters `n`, `d`, `o` and `m` are mandatory, whith nb being an optional input argument.

    Model(n, d, o, m; nb=40)


"""
Model(n::IntTuple,d::RealTuple,o::RealTuple,m;nb=40) = Model(n,d,o,nb,m)


