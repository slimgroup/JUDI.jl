# Test 2D modeling
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using JUDI
using Test, Printf, Aqua
using SegyIO, LinearAlgebra, Distributed, JOLI
using TimerOutputs: TimerOutputs, @timeit

set_verbosity(false)
# Collect timing and allocations information to show in a clear way.
const TIMEROUTPUT = TimerOutputs.TimerOutput()
timeit_include(path::AbstractString) = @timeit TIMEROUTPUT path include(path)

# Utilities
const success_log = Dict(true => "\e[32m SUCCESS \e[0m", false => "\e[31m FAILED \e[0m")

# Test set
const GROUP = get(ENV, "GROUP", "JUDI")

# JUDI seismic utils
include("seismic_utils.jl")

const nlayer = 2
const tti = contains(GROUP, "TTI")
const fs = contains(GROUP, "FS")
const viscoacoustic = contains(GROUP, "VISCO")

# Utility macro to run block of code with a single omp threa
macro single_threaded(expr)
    return quote
        nthreads = ENV["OMP_NUM_THREADS"]
        ENV["OMP_NUM_THREADS"] = "1"
        local val = $(esc(expr))
        ENV["OMP_NUM_THREADS"] = nthreads
        val
    end
end

# Testing Utilities
include("testing_utils.jl")


base = ["test_geometry.jl",
        "test_judiVector.jl",
        "test_composite.jl",
        "test_judiWeights.jl",
        "test_judiWavefield.jl",
        "test_linear_operators.jl",
        "test_physicalparam.jl",
        "test_compat.jl"]

devito = ["test_all_options.jl",
          "test_linearity.jl",
          "test_preconditioners.jl",
          "test_adjoint.jl",
          "test_jacobian.jl",
          "test_gradients.jl",
          "test_multi_exp.jl",
          "test_rrules.jl"]

extras = ["test_modeling.jl", "test_basics.jl", "test_linear_algebra.jl"]

issues = ["test_issues.jl"]

# custom
if endswith(GROUP, ".jl")
    timeit_include(GROUP)
end

# Basic JUDI objects tests, no Devito
if GROUP == "JUDI" || GROUP == "All"
    for t=base
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
    # Test resolved issues Due to some data type incomaptibilities only test 1.7
    if VERSION >= v"1.7"
        for t=issues
            timeit_include(t)
            try Base.GC.gc(); catch; gc() end
        end
    end
end

# Generic mdeling tests
if GROUP == "BASICS" || GROUP == "All"
    println("JUDI generic modelling tests")
    for t=extras
        timeit_include(t)
        @everywhere try Base.GC.gc(); catch; gc() end
    end
end

# Isotropic Acoustic tests
if GROUP == "ISO_OP" || GROUP == "All"
    println("JUDI iso-acoustic operators tests (parallel)")
    # Basic test of LA/CG/LSQR needs
    for t=devito
        timeit_include(t)
        @everywhere try Base.GC.gc(); catch; gc() end
    end
end

# Isotropic Acoustic tests with a free surface
if GROUP == "ISO_OP_FS" || GROUP == "All"
    println("JUDI iso-acoustic operators with free surface tests")
    for t=devito
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
end

# Anisotropic Acoustic tests
if GROUP == "TTI_OP" || GROUP == "All"
    println("JUDI TTI operators tests")
    # TTI tests
    for t=devito
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
end

# Anisotropic Acoustic tests with free surface
if GROUP == "TTI_OP_FS" || GROUP == "All"
    println("JUDI TTI operators with free surface tests")
    for t=devito
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
end

# Viscoacoustic tests
if GROUP == "VISCO_AC_OP" || GROUP == "All"
    println("JUDI Viscoacoustic operators tests")
    # Viscoacoustic tests
    for t=devito
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
end

# Code quality check
if GROUP == "AQUA" || GROUP == "All" || GROUP == "JUDI"
    @testset "code quality" begin
        # Prevent ambiguities from PyCall and other packages
        Aqua.test_all(JUDI; ambiguities=false)
        Aqua.test_ambiguities(JUDI; Aqua.askwargs(true)...)
    end
end

# Testing memory and runtime summary
show(TIMEROUTPUT; compact=true, sortby=:firstexec)
