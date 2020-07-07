using Test

const GROUP = get(ENV, "GROUP", "JUDI")

if GROUP == "JUDI" || GROUP == "All"
    @testset "JUDI Unit tests" begin
        include("test_abstract_vectors.jl")
        include("test_geometry.jl")
        include("test_judiVector.jl")
        include("test_linear_operators.jl")
    end
end

if GROUP == "ISO_OP" || GROUP == "All"
    include("test_utils.jl")
    @testset "JUDI iso-acoustic operators tests (parallel)" begin
        push!(Base.ARGS, "-p 2")
        # Basic utility test
        include("basic_tests.jl")
        # Iso-acoustic adjoint tests
        include("linearityTest.jl")
        include("test_jacobian.jl")
        include("test_jacobian_extended.jl")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
        include("test_all_options.jl")
    end
end

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

if GROUP == "TTI_OP" || GROUP == "All"
    include("test_utils.jl")
    @testset "JUDI TTI operators tests" begin
        # TTI adjoint tests
        push!(Base.ARGS, "--tti")
        include("linearityTest.jl")
        include("test_jacobian.jl")
        include("test_jacobian_extended.jl")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
        include("test_all_options.jl")
    end
end

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