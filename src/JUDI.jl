# Module with functions for time-domain modeling and inversion using OPESCI/devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
# Updated, December 2020, Mathias Louboutin, mlouboutin3@gatech.edu

__precompile__()
module JUDI

export JUDIPATH, set_verbosity
JUDIPATH = dirname(pathof(JUDI))

# Dependencies
using LinearAlgebra
using Distributed
using DSP, FFTW, Dierckx
using PyCall
using JOLI, SegyIO

# Import Base functions to dispatch on JUDI types
import Base.*, Base./, Base.+, Base.-
import Base.copy!, Base.copy, Base.convert
import Base.sum, Base.ndims, Base.reshape, Base.fill!, Base.axes, Base.dotview
import Base.eltype, Base.length, Base.size, Base.iterate, Base.show, Base.display, Base.showarg
import Base.maximum, Base.minimum, Base.push!
import Base.Broadcast.broadcasted, Base.BroadcastStyle, Base.Broadcast.DefaultArrayStyle, Base.Broadcast, Base.broadcast!
import Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex
import Base.similar, Base.isapprox, Base.isequal
import Base.materialize!, Base.materialize
import Base.promote_shape, Base.diff, Base.cumsum, Base.cumsum!
import Base.getproperty

# Import Linear Lagebra functions to dispatch on JUDI types
import LinearAlgebra.transpose, LinearAlgebra.conj, LinearAlgebra.vcat, LinearAlgebra.adjoint
import LinearAlgebra.vec, LinearAlgebra.dot, LinearAlgebra.norm, LinearAlgebra.abs
import LinearAlgebra.rmul!, LinearAlgebra.lmul!, LinearAlgebra.rdiv!, LinearAlgebra.ldiv!
import LinearAlgebra.mul!, Base.isfinite

# JOLI
import JOLI: jo_convert

# Import pycall array to python for easy plotting
import PyCall.array2py

# Set python paths
const pm = PyNULL()
const ac = PyNULL()

# Constants
_worker_pool = default_worker_pool
_TFuture = Future
_verbose = false

set_verbosity(x::Bool) = begin global _verbose = x; end
judilog(msg) = _verbose ? println(msg) : nothing

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    copy!(pm, pyimport("models"))
    copy!(ac, pyimport("interface"))
end

# JUDI time modeling
include("TimeModeling/TimeModeling.jl")

module TimeModeling
    using ..JUDI
    # Define backward compatible types so that old JLD files load properly
    # Old JLD file expected to be arrays, SeisCon are expected to be read and built
    # from SEGY files
    const judiVector{T} = JUDI.judiVector{T, Matrix{T}} where T
    const GeometryIC = JUDI.GeometryIC{Float32}
    const GeometryOOC = JUDI.GeometryOOC{Float32}
end

end
