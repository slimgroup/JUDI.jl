# Module with functions for time-domain modeling and inversion using OPESCI/devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
# Updated, December 2020, Mathias Louboutin, mlouboutin3@gatech.edu

__precompile__()
module JUDI

export JUDIPATH, set_verbosity, ftp_data
JUDIPATH = dirname(pathof(JUDI))

# Dependencies
using LinearAlgebra, Random
using Distributed
using DSP, FFTW, Dierckx
using PyCall
using JOLI, SegyIO
using ChainRulesCore
using Requires
using OrderedCollections

# Import Base functions to dispatch on JUDI types
import Base.depwarn
import Base.*, Base./, Base.+, Base.-, Base.==, Base.\
import Base.copy!, Base.copy, Base.copyto!, Base.deepcopy, Base.summary
import Base.sum, Base.ndims, Base.reshape, Base.fill!, Base.axes, Base.dotview
import Base.eltype, Base.length, Base.size, Base.iterate, Base.show, Base.display, Base.showarg
import Base.maximum, Base.minimum, Base.push!
import Base.Broadcast.ArrayStyle, Base.Broadcast.extrude
import Base.Broadcast.broadcasted, Base.BroadcastStyle, Base.Broadcast.DefaultArrayStyle, Base.Broadcast, Base.broadcast!
import Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex
import Base.similar, Base.isapprox, Base.isequal
import Base.materialize!, Base.materialize, Base.Broadcast.instantiate
import Base.promote_shape, Base.diff, Base.cumsum, Base.cumsum!
import Base.getproperty, Base.unsafe_convert, Base.convert

# Import Linear Lagebra functions to dispatch on JUDI types
import LinearAlgebra.transpose, LinearAlgebra.conj, LinearAlgebra.vcat, LinearAlgebra.adjoint
import LinearAlgebra.vec, LinearAlgebra.dot, LinearAlgebra.norm, LinearAlgebra.abs
import LinearAlgebra.rmul!, LinearAlgebra.lmul!, LinearAlgebra.rdiv!, LinearAlgebra.ldiv!
import LinearAlgebra.mul!, Base.isfinite, Base.inv

# JOLI
import JOLI: jo_convert

# FFTW
import FFTW: fft, ifft

# Import pycall array to python for easy plotting
import PyCall.NpyArray

# Import AD rrule
import ChainRulesCore: rrule

# Set python paths
const pm = PyNULL()
const ac = PyNULL()
const pyut = PyNULL()

# Create a lock for pycall FOR THREAD/TASK SAFETY
# See discussion at
# https://github.com/JuliaPy/PyCall.jl/issues/882

const PYLOCK = Ref{ReentrantLock}()

# acquire the lock before any code calls Python
pylock(f::Function) = Base.lock(PYLOCK[]) do
    prev_gc = GC.enable(false)
    try 
        return f()
    finally
        GC.enable(prev_gc) # recover previous state
    end
end

# Constants
function _worker_pool()
    p = default_worker_pool()
    pool = length(p) < 2 ? nothing : p
    return pool
end

_TFuture = Future
_verbose = false
_devices = []

# Utility for data loading
JUDI_DATA = joinpath(JUDIPATH, "../data")
ftp_data(ftp::String, name::String) = Base.Downloads().download("$(ftp)/$(name)", "$(JUDI.JUDI_DATA)/$(name)")
ftp_data(ftp::String) = Base.Downloads().download(ftp, "$(JUDI.JUDI_DATA)/$(split(ftp, "/")[end])")

# Some usefull types
const RangeOrVec = Union{AbstractRange, Vector}

set_verbosity(x::Bool) = begin global _verbose = x; end
judilog(msg) = _verbose ? println(msg) : nothing

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

# Backward Compatibility
include("compat.jl")

# Automatic Differentiation
include("rrules.jl")

# Initialize
function __init__()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    copy!(pm, pyimport("models"))
    copy!(ac, pyimport("interface"))
    copy!(pyut, pyimport("utils"))
    # Initialize lock at session start
    PYLOCK[] = ReentrantLock()

    if get(ENV, "DEVITO_PLATFORM", "") == "nvidiaX"
        @info "Initializing openacc/openmp offloading"
        devito_model(Model((21, 21, 21), (10., 10., 10.), (0., 0., 0.), randn(Float32, 21, 21, 21)), Options())
        global _devices = parse.(Int, get(ENV, "CUDA_VISIBLE_DEVICES", "-1"))
    end

    @require Zygote="e88e6eb3-aa80-5325-afca-941959d7151f" begin
        Zygote.unbroadcast(x::AbstractArray, x̄::LazyPropagation) = Zygote.unbroadcast(x, eval_prop(x̄))
    end

    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        Flux.Zygote.unbroadcast(x::AbstractArray, x̄::LazyPropagation) = Zygote.unbroadcast(x, eval_prop(x̄))
        Flux.cpu(x::LazyPropagation) = Flux.cpu(eval_prop(x))
        Flux.gpu(x::LazyPropagation) = Flux.gpu(eval_prop(x))
        Flux.CUDA.cu(F::LazyPropagation) = Flux.CUDA.cu(eval_prop(F))
    end
end

end
