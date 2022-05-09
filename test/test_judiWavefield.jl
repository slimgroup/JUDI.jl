# Unit tests for judiWeights
# Rafael Orozco (rorozco@gatech.edu)
# July 2021
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

# number of sources/receivers
nsrc = 1
nrec = 120
nx = 4
ny = 4
nt = 10
dt = 2f0
ftol = 1f-6

################################################# test constructors ####################################################

@testset "judiWavefield Unit Tests with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiWavefield (nsrc=$(nsrc))" begin
        # Extended source weights
        wf = Array{Array{Float32, 3}, 1}(undef, nsrc)
        for j=1:nsrc
            wf[j] = randn(Float32, nt, ny, ny)
        end
        w = judiWavefield(dt, wf)
        w1 = similar(w) .+ 1f0

        @test isequal(length(w.data), nsrc)
        @test isequal(length(w.data), nsrc)
        @test isequal(w.nsrc, nsrc)
        @test isequal(typeof(w.data), Array{Array{Float32, 3}, 1})
        @test isequal(size(w), (nsrc,))
        @test isfinite(w)

        #################################################### test operations ###################################################

        # conj, transpose, adjoint
        @test isequal(size(w), size(conj(w)))
        @test isequal(reverse(size(w)), size(transpose(w)))
        @test isequal(reverse(size(w)), size(adjoint(w)))

        # +, -, *, /
        @test iszero(norm(2*w - (w + w)))
        @test iszero(norm(w - (w + w)/2))
        @test iszero(norm(1f0 - w1))
        @test isequal(norm(1f0 + w1, 1), 2f0 * norm(w1, 1))
        #test unary operator
        @test iszero(norm((-w) - (-1*w))) 

        # vcat
        w_vcat = [w; w]
        @test isequal(length(w_vcat), 2*length(w))
        @test isequal(w_vcat.nsrc, 2*nsrc)
        @test isequal(length(w_vcat.data), 2*nsrc)

        # dot, norm, abs
        @test isapprox(norm(w), sqrt(dot(w, w)))
        @test isapprox(abs.(w.data[1]), abs(w).data[1]) 

        # Test the norm
        d_ones = judiWavefield(nsrc, dt, 2f0 .* ones(Float32, nt, nx, ny))
        @test isapprox(norm(d_ones, 2), sqrt(dt*nt*nx*ny*4*nsrc))
        @test isapprox(norm(d_ones, 1), dt*nt*nx*ny*2*nsrc)
        @test isapprox(norm(d_ones, Inf), 2)

        # vector space axioms
        u = judiWavefield(nsrc, dt, randn(Float32, nt, nx, ny))
        v = judiWavefield(nsrc, dt, randn(Float32, nt, nx, ny))
        w = judiWavefield(nsrc, dt, randn(Float32, nt, nx, ny))
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

        # test fft
        fw = fft(w)
        fwf = ifft(fw)
        @test isapprox(dot(fwf, w), real(dot(fw, fw)); rtol=ftol)
        @test isapprox(fwf, w; rtol=ftol)
    end
end
