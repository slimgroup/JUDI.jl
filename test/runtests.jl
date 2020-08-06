# Test 2D modeling
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using Test

const GROUP = get(ENV, "GROUP", "JUDI")

include("test_utils.jl")

if endswith(GROUP, ".jl")
    include(GROUP)
end

# Basic JUDI objects tests, no Devito
if GROUP == "JUDI" || GROUP == "All"
    include("test_abstract_vectors.jl")
    include("test_geometry.jl")
    include("test_judiVector.jl")
    include("test_composite.jl")
    include("test_judiWeights.jl")
    include("test_judiWavefield.jl")
    include("test_linear_operators.jl")
end

# Isotropic Acoustic tests
if GROUP == "ISO_OP" || GROUP == "All"
    println("JUDI iso-acoustic operators tests (parallel)")
    # Basic utility test
    include("basic_tests.jl")
    # Basic test of LA/CG/LSQR needs
    push!(Base.ARGS, "-p 2")
    include("test_linear_algebra.jl")
    # Iso-acoustic tests
    include("modelingTest.jl")
    include("linearityTest.jl")
    include("test_jacobian.jl")
    include("test_jacobian_extended.jl")
    include("adjointTest.jl")
    include("fwiGradientTest.jl")
    include("test_all_options.jl")
end

# Isotropic Acoustic tests with a free surface
if GROUP == "ISO_OP_FS" || GROUP == "All"
    println("JUDI iso-acoustic operators with free surface tests")
    push!(Base.ARGS, "--fs")
    include("linearityTest.jl")
    include("test_jacobian.jl")
    include("test_jacobian_extended.jl")
    include("adjointTest.jl")
    include("test_all_options.jl")
end

# Anisotropic Acoustic tests
if GROUP == "TTI_OP" || GROUP == "All"
    println("JUDI TTI operators tests")
    # TTI tests
    push!(Base.ARGS, "--tti")
    include("linearityTest.jl")
    include("test_jacobian.jl")
    include("test_jacobian_extended.jl")
    include("adjointTest.jl")
    include("fwiGradientTest.jl")
    include("test_all_options.jl")
end

# Anisotropic Acoustic tests with free surface
if GROUP == "TTI_OP_FS" || GROUP == "All"
    println("JUDI TTI operators with free surfacetests")
    push!(Base.ARGS, "--tti")
    push!(Base.ARGS, "--fs")
    include("linearityTest.jl")
    include("test_jacobian.jl")
    include("test_jacobian_extended.jl")
    include("adjointTest.jl")
    include("test_all_options.jl")
end
