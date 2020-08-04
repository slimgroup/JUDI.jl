# Unit tests for judiWeights
# Rafael Orozco (rorozco@gatech.edu)
# July 2021

using JUDI.TimeModeling, Test, LinearAlgebra
import LinearAlgebra.BLAS.axpy!


# number of sources/receivers
nsrc = 1
nrec = 120
weight_size_x = 4
weight_size_y = 4
ftol = 1f-6

################################################# test constructors ####################################################

@testset "judiWeights Unit Tests" begin

    # set up judiWeights


    # Extended source weights
    weights = Array{Array}(undef, nsrc)
    for j=1:nsrc
        weights[j] = randn(Float32, weight_size_x,weight_size_y) #model size (100,100)
    end
    w = judiWeights(weights)

    @test isequal(w.nsrc, nsrc)
    @test isequal(typeof(w.weights), Array{Array, 1})
    @test isequal(size(w), (weight_size_x*weight_size_y,1))



#################################################### test operations ###################################################

    # conj, transpose, adjoint
    @test isequal(size(w), size(conj(w)))

    @test isequal(reverse(size(w)), size(transpose(w)))

    @test isequal(reverse(size(w)), size(adjoint(w)))

    # +, -, *, /
    @test iszero(norm(2*w - (w + w)))
    @test iszero(norm(w - (w + w)/2))

    #test unary operator
    @test iszero(norm((-w) - (-1*w))) 

    # vcat
    w_vcat = [w; w]
    @test isequal(length(w_vcat), 2*length(w))
    @test isequal(w_vcat.nsrc, 2*w.nsrc)

    # dot, norm, abs
    @test isapprox(norm(w), sqrt(dot(w, w)))
    @test isapprox(abs.(w.weights[1]), abs(w).weights[1]) 

    # Test the norm
    d_ones = judiWeights(2f0 .* ones(Float32, weight_size_x, weight_size_y))
    @test isapprox(norm(d_ones, 2), sqrt(weight_size_x*weight_size_y*4))
    @test isapprox(norm(d_ones, 1), weight_size_x*weight_size_y*2)
    @test isapprox(norm(d_ones, Inf), 2)

    # vector space axioms
    u = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    v = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    w = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    a = randn(1)[1]
    b = randn(1)[1]

    @test isapprox(u + (v + w), (u + v) + w; rtol=ftol)
    @test isapprox(u + v, v + u; rtol=ftol)
    @test isapprox(u, u + 0; rtol=ftol)
    @test iszero(norm(u + u*(-1)))
    @test isapprox(-u, (-1f0)*u; rtol=ftol)
    @test isapprox(a .* (b .* u), (a * b) .* u; rtol=ftol)
    @test isapprox(u, u .* 1; rtol=ftol)
    @test isapprox(a .* (u + v), a .* u + a .* v; rtol=ftol)
    @test isapprox((a + b) .* v, a .* v + b.* v; rtol=ftol)


    # broadcast multiplication
    u = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    v = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    u_scale = deepcopy(u)
    v_scale = deepcopy(v)
    
    u_scale .*= 2f0
    @test isapprox(u_scale, 2f0 * u; rtol=ftol)
    v_scale .+= 2f0
    @test isapprox(v_scale, 2f0 + v; rtol=ftol)
    u_scale ./= 2f0
    @test isapprox(u_scale, u; rtol=ftol)
    u_scale .= 2f0 .* u_scale .+ v_scale
    @test isapprox(u_scale, 2f0 * u + 2f0 + v; rtol=ftol)
   

end
