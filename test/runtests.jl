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
    @testset "JUDI iso-acoustic operators tests" begin
        include("linearityTest.jl")
        include("test_jacobian.jl")
        include("test_jacobian_extended.jl")
        # Iso-acoustic adjoint tests
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
        push!(Base.ARGS, "--fs")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
    end
end

if GROUP == "TTI_OP" || GROUP == "All"
    @testset "JUDI TTI operators tests" begin
        # TTI adjoint tests
        push!(Base.ARGS, "--tti")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
        push!(Base.ARGS, "--fs")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
    end
end