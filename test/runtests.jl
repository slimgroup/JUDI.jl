# Test 2D modeling
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using Test

const GROUP = get(ENV, "GROUP", "JUDI")

# Basic JUDI objects tests, no Devito
if GROUP == "JUDI" || GROUP == "All"
    @testset "JUDI Unit tests" begin
        include("test_abstract_vectors.jl")
        include("test_geometry.jl")
        include("test_judiVector.jl")
        include("test_composite.jl")
        include("test_judiWeights.jl")
        include("test_linear_operators.jl")
    end
end

# Isotropic Acoustic tests
if GROUP == "ISO_OP" || GROUP == "All"
    include("test_utils.jl")
    @testset "JUDI iso-acoustic operators tests (parallel)" begin
        push!(Base.ARGS, "-p 2")
        # Basic utility test
        include("basic_tests.jl")
        # Iso-acoustic tests
        include("modelingTest.jl")
        include("linearityTest.jl")
        include("test_jacobian.jl")
        include("test_jacobian_extended.jl")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
        include("test_all_options.jl")
    end
end

# Isotropic Acoustic tests with a free surface
if GROUP == "ISO_OP_FS" || GROUP == "All"
    include("test_utils.jl")
    @testset "JUDI iso-acoustic operators with free surface tests" begin
        push!(Base.ARGS, "--fs")
        include("linearityTest.jl")
        include("test_jacobian.jl")
        include("test_jacobian_extended.jl")
        include("adjointTest.jl")
        include("test_all_options.jl")
    end
end

# Anisotropic Acoustic tests
if GROUP == "TTI_OP" || GROUP == "All"
    include("test_utils.jl")
    @testset "JUDI TTI operators tests" begin
        # TTI tests
        push!(Base.ARGS, "--tti")
        include("linearityTest.jl")
        include("test_jacobian.jl")
        include("test_jacobian_extended.jl")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
        include("test_all_options.jl")
    end
end

# Anisotropic Acoustic tests with free surface
if GROUP == "TTI_OP_FS" || GROUP == "All"
    include("test_utils.jl")
    @testset "JUDI TTI operators with free surfacetests" begin
        push!(Base.ARGS, "--tti")
        push!(Base.ARGS, "--fs")
        include("linearityTest.jl")
        include("test_jacobian.jl")
        include("test_jacobian_extended.jl")
        include("adjointTest.jl")
        include("test_all_options.jl")
    end
end
