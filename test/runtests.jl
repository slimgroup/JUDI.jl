# Test 2D modeling
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using JUDI, ArgParse, Test
using SegyIO, LinearAlgebra, Printf, Distributed, JOLI

const GROUP = get(ENV, "GROUP", "JUDI")

include("utils.jl")

if endswith(GROUP, ".jl")
    @time include(GROUP)
end

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
          "test_gradient_lsrtm.jl"]
        #   "test_gradient_twri.jl"]

extras = ["test_modeling.jl", "test_basics.jl", "test_linear_algebra.jl"]

# Basic JUDI objects tests, no Devito
if GROUP == "JUDI" || GROUP == "All"
    for t=base
        @time include(t)
    end
end

# Generic mdeling tests
if GROUP == "BASICS" || GROUP == "All"
    println("JUDI generic modelling tests")
    VERSION >= v"1.5" && push!(Base.ARGS, "-p 2")
    for t=extras
        @time include(t)
    end
end

# Isotropic Acoustic tests
if GROUP == "ISO_OP" || GROUP == "All"
    println("JUDI iso-acoustic operators tests (parallel)")
    # Basic test of LA/CG/LSQR needs
    VERSION >= v"1.5" && push!(Base.ARGS, "-p 2")
    for t=devito
        @time include(t)
    end
end

# Isotropic Acoustic tests with a free surface
if GROUP == "ISO_OP_FS" || GROUP == "All"
    println("JUDI iso-acoustic operators with free surface tests")
    push!(Base.ARGS, "--fs")
    for t=devito
        @time include(t)
    end
end

# Anisotropic Acoustic tests
if GROUP == "TTI_OP" || GROUP == "All"
    println("JUDI TTI operators tests")
    # TTI tests
    push!(Base.ARGS, "--tti")
    for t=devito
        @time include(t)
    end
end

# Anisotropic Acoustic tests with free surface
if GROUP == "TTI_OP_FS" || GROUP == "All"
    println("JUDI TTI operators with free surface tests")
    push!(Base.ARGS, "--tti")
    push!(Base.ARGS, "--fs")
    for t=devito
        @time include(t)
    end
end
