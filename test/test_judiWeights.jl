# Unit tests for judiWeights
# Rafael Orozco (rorozco@gatech.edu)
# July 2021
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using JUDI.TimeModeling, Test, LinearAlgebra, JOLI
import LinearAlgebra.BLAS.axpy!


# number of sources/receivers
nrec = 120
weight_size_x = 4
weight_size_y = 4
ftol = 1f-6

################################################# test constructors ####################################################

@testset "judiWeights Unit Tests with $(nsrc) sources" for nsrc=[1, 2]

    # set up judiWeights


    # Extended source weights
    weights = Array{Array}(undef, nsrc)
    for j=1:nsrc
        weights[j] = randn(Float32, weight_size_x,weight_size_y) #model size (100,100)
    end
    w = judiWeights(weights)

    @test isequal(w.nsrc, nsrc)
    @test isequal(typeof(w.weights), Array{Array, 1})
    @test isequal(size(w), (weight_size_x*weight_size_y*nsrc, 1))
    @test isfinite(w)
#################################################### test operations ###################################################
    # Indexing/reshape/...
    @test isapprox(w[1].weights[1], w.weights[1])
    @test isapprox(subsample(w, 1).weights[1], w.weights[1])

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
    d_ones = judiWeights(2f0 .* ones(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    @test isapprox(norm(d_ones, 2), sqrt(nsrc*weight_size_x*weight_size_y*4))
    @test isapprox(norm(d_ones, 2), sqrt(dot(d_ones, d_ones)))
    @test isapprox(norm(d_ones, 1), nsrc*weight_size_x*weight_size_y*2)
    @test isapprox(norm(d_ones, Inf), 2)

    # vector space axioms
    u = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    v = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    w = judiWeights(randn(Float32, weight_size_x, weight_size_y); nsrc=nsrc)
    a = .5f0 + rand(Float32)
    b = .5f0 + rand(Float32)

    @test isapprox(u + (v + w), (u + v) + w; rtol=ftol)
    @test isapprox(u + v, v + u; rtol=ftol)
    @test isapprox(u, u + 0; rtol=ftol)
    @test iszero(norm(u + u*(-1)))
    @test isapprox(-u, (-1f0)*u; rtol=ftol)
    @test isapprox(a .* (b .* u), (a * b) .* u; rtol=ftol)
    @test isapprox(u, u .* 1; rtol=ftol)
    @test isapprox(a .* (u + v), a .* u + a .* v; rtol=1f-5)
    @test isapprox((a + b) .* v, a .* v + b.* v; rtol=1f-5)


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
    u_scale .= u .+ v
    @test isapprox(u_scale, u + v)
    u_scale .= u .- v
    @test isapprox(u_scale, u - v)
    u_scale .= u .* v
    @test isapprox(u_scale.weights[1], u.weights[1].*v.weights[1])
    u_scale .= u ./ v
    @test isapprox(u_scale.weights[1], u.weights[1]./v.weights[1])

    # Test copies/similar
    w1 = deepcopy(w)
    @test isapprox(w1, w)
    w1 = similar(w)
    @test w1.nsrc == w.nsrc
    @test isapprox(w1.weights, 0f0 .* w.weights)
    w1 .= w
    @test w1.nsrc == w.nsrc
    @test isapprox(w1.weights, w.weights)

end
