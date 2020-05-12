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

if GROUP == "OP" || GROUP == "All"
    @testset "JUDI operators tests" begin
        include("linearityTest.jl")
        include("adjointTest.jl")
        include("fwiGradientTest.jl")
        include("test_jacobian_extended.jl")
        include("test_jacobian_extended.jl")
        push!(Base.ARGS, "--tti")
        include("adjointTest.jl")
    end
end