using Pkg; Pkg.activate("JUDI")
using Test
@testset "JUDI Unit tests" begin
    include("test_abstract_vectors.jl")
    include("test_geometry.jl")
    include("test_judiVector.jl")
    include("test_linear_operators.jl")
end
