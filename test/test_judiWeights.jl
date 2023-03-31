# Unit tests for judiWeights
# Rafael Orozco (rorozco@gatech.edu)
# July 2021
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

# number of sources/receivers
nrec = 120
weight_size_x = 4
weight_size_y = 4
ftol = 1f-6

################################################# test constructors ####################################################

@testset "judiWeights Unit Tests with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiWeights (nsrc=$(nsrc))" begin
        # Extended source weights
        w = judiWeights([randn(Float64,weight_size_x,weight_size_y) for i = 1:nsrc])

        @test isequal(w.nsrc, nsrc)
        @test isequal(typeof(w.weights), Array{Array{Float32,2},1})
        @test isequal(size(w), (nsrc,))
        @test isfinite(w)
        
        w_cell = judiWeights(convertToCell([randn(Float32,weight_size_x,weight_size_y) for i = 1:nsrc]))

        @test isequal(w_cell.nsrc, nsrc)
        @test isequal(typeof(w_cell.weights), Array{Array{Float32, 2},1})
        @test isequal(size(w_cell), (nsrc,))
        @test isfinite(w_cell)
        
        w_multi = judiWeights(randn(Float64,weight_size_x,weight_size_y); nsrc=3)

        @test isequal(w_multi.nsrc, 3)
        @test isequal(typeof(w_multi.weights), Array{Array{Float32, 2},1})
        @test isequal(size(w_multi), (3,))
        @test isfinite(w_multi)
        @test isapprox(w_multi[1],w_multi[2])
        @test isapprox(w_multi[2],w_multi[3])

        I2 = ones(Float32, 1, nsrc*weight_size_x*weight_size_y)
        w2 = I2*w
        @test isapprox(w2[1], sum(sum(w.weights)))

        I2 = joOnes(1, nsrc*weight_size_x*weight_size_y; DDT=Float32, RDT=Float32)
        w2 = I2*w
        @test isapprox(w2[1], sum(sum(w.weights)))

        I3 = [I2;I2]
        w3 = I3*w
        @test isapprox(w3[1], sum(sum(w.weights)))
        @test isapprox(w3[2], sum(sum(w.weights)))

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

        # Copies and iter utilities
        w2 = copy(w)
        @test isequal(w2.weights, w.weights)
        @test isequal(w2.nsrc, w.nsrc)
        @test firstindex(w) == 1
        @test lastindex(w) == nsrc
        @test ndims(w) == 1

        w2 = similar(w, Float32, 1:nsrc)
        @test isequal(w2.nsrc, w.nsrc)
        @test firstindex(w) == 1
        @test lastindex(w) == nsrc
        @test ndims(w) == 1

        w3 = similar(w, Float32, 1)
        @test isequal(w3.nsrc, 1)

        copy!(w2, w)
        @test isequal(w2.weights, w.weights)
        @test isequal(w2.nsrc, w.nsrc)
        @test firstindex(w) == 1
        @test lastindex(w) == nsrc
        @test ndims(w) == 1

        # vcat
        w_vcat = [w; w]
        @test isequal(length(w_vcat), 2*length(w))
        @test isequal(w_vcat.nsrc, 2*w.nsrc)

        # dot, norm, abs
        @test isapprox(norm(w), sqrt(dot(w, w)))
        @test isapprox(abs.(w.weights[1]), abs(w).weights[1])
        
        # max, min
        w_max_min = judiWeights([ones(Float32,weight_size_x,weight_size_y) for i = 1:nsrc])
        w_max_min.weights[1][1,1] = 1f3
        w_max_min.weights[nsrc][end,end] = 1f-3
        @test isapprox(maximum(w_max_min),1f3)
        @test isapprox(minimum(w_max_min),1f-3)

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
        w1 .= w
        @test w1.nsrc == w.nsrc
        @test isapprox(w1.weights, w.weights)

        # SimSources
        if nsrc == 2
            M = randn(Float32, 1, nsrc)
            sw = M*w
            @test sw.nsrc == 1
            @test isapprox(sw.data[1], M[1]*w.data[1] + M[2]*w.data[2])
        end
    end
end
