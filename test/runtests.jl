# Test 2D modeling
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using JUDI
using ArgParse, Test, Printf
using SegyIO, LinearAlgebra, Distributed, JOLI
using TimerOutputs: TimerOutputs, @timeit

# Collect timing and allocations information to show in a clear way.
const TIMEROUTPUT = TimerOutputs.TimerOutput()
timeit_include(path::AbstractString) = @timeit TIMEROUTPUT path include(path)

# Utilities
const success_log = Dict(true => "SUCCESS", false => "FAILED")
# Test set
const GROUP = get(ENV, "GROUP", "JUDI")

include("utils.jl")

base = ["test_abstract_vectors.jl",
        "test_geometry.jl",
        "test_judiVector.jl",
        "test_composite.jl",
        "test_judiWeights.jl",
        "test_judiWavefield.jl",
        "test_linear_operators.jl",
        "test_physicalparam.jl"]

devito = ["test_linearity.jl",
          "test_adjoint.jl",
          "test_all_options.jl",
          "test_jacobian.jl",
          "test_jacobian_extended.jl",
          "test_gradient_fwi.jl",
          "test_gradient_lsrtm.jl",
          "test_multi_exp.jl"]
        #   "test_gradient_twri.jl"]

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
    VERSION >= v"1.7" && push!(Base.ARGS, "-p 2")
    for t=extras
        timeit_include(t)
        @everywhere try Base.GC.gc(); catch; gc() end
    end
end

# Isotropic Acoustic tests
if GROUP == "ISO_OP" || GROUP == "All"
    println("JUDI iso-acoustic operators tests (parallel)")
    # Basic test of LA/CG/LSQR needs
    VERSION >= v"1.7" && push!(Base.ARGS, "-p 2")
    for t=devito
        timeit_include(t)
        @everywhere try Base.GC.gc(); catch; gc() end
    end
end

# Isotropic Acoustic tests with a free surface
if GROUP == "ISO_OP_FS" || GROUP == "All"
    println("JUDI iso-acoustic operators with free surface tests")
    push!(Base.ARGS, "--fs")
    for t=devito
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
end

# Anisotropic Acoustic tests
if GROUP == "TTI_OP" || GROUP == "All"
    println("JUDI TTI operators tests")
    # TTI tests
    push!(Base.ARGS, "--tti")
    for t=devito
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
end

# Anisotropic Acoustic tests with free surface
if GROUP == "TTI_OP_FS" || GROUP == "All"
    println("JUDI TTI operators with free surface tests")
    push!(Base.ARGS, "--tti")
    push!(Base.ARGS, "--fs")
    for t=devito
        timeit_include(t)
        try Base.GC.gc(); catch; gc() end
    end
end

# Viscoacoustic tests
if GROUP == "VISCO_AC_OP" || GROUP == "All"
    println("JUDI Viscoacoustic operators tests")
    # Viscoacoustic tests
    push!(Base.ARGS, "--viscoacoustic")
    visco = ["test_jacobian_extended.jl"]
    for t=devito
        if (~any(x->x==t, visco))
            @time include(t)
            try Base.GC.gc(); catch; gc() end
        end
    end
end

show(TIMEROUTPUT; compact=true, sortby=:firstexec)
