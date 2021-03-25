# Module with functions for time-domain modeling and inversion using OPESCI/devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
# Updated, December 2020, Mathias Louboutin, mlouboutin3@gatech.edu

__precompile__()
module JUDI

export JUDIPATH
JUDIPATH = dirname(pathof(JUDI))

# Dependencies
using Pkg, Printf, LinearAlgebra
using Distributed
using DSP, FFTW, Dierckx
using PyCall
using JOLI, SegyIO

# Import Base functions to dispatch on JUDI types
import Base.*, Base./, Base.+, Base.-
import Base.copy!, Base.copy
import Base.sum, Base.ndims, Base.reshape, Base.fill!, Base.axes, Base.dotview
import Base.eltype, Base.length, Base.size, Base.iterate, Base.show, Base.display, Base.showarg
import Base.maximum, Base.minimum
import Base.Broadcast.broadcasted, Base.BroadcastStyle, Base.Broadcast.DefaultArrayStyle, Base.Broadcast, Base.broadcast!
import Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex
import Base.similar, Base.isapprox, Base.isequal
import Base.materialize!, Base.materialize
import Base.promote_shape, Base.diff, Base.cumsum, Base.cumsum!

# Import Linear Lagebra functions to dispatch on JUDI types
import LinearAlgebra.transpose, LinearAlgebra.conj, LinearAlgebra.vcat, LinearAlgebra.adjoint
import LinearAlgebra.vec, LinearAlgebra.dot, LinearAlgebra.norm, LinearAlgebra.abs
import LinearAlgebra.rmul!, LinearAlgebra.lmul!, LinearAlgebra.rdiv!, LinearAlgebra.ldiv!
import LinearAlgebra.mul!, Base.isfinite

# Import pycall array to python for easy plotting
import PyCall.array2py

# Set python paths
const pm = PyNULL()
const ac = PyNULL()

function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    copy!(pm, pyimport("models"))
    copy!(ac, pyimport("interface"))
end

# JUDI time modeling
include("TimeModeling/TimeModeling.jl")

# # Backward compatibility for JUDI
module TimeModeling
    using Reexport
    @reexport using ..JUDI
    Base.@warn "JUDI.TimeModeling is deprecated, use `using JUDI` instead"
end


end
