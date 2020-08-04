# Unit tests for judiComposite
# Mathias Louboutin (mlouboutin3@gatech.edu)
# July 2021

using JUDI.TimeModeling, Test, LinearAlgebra

function example_rec_geometry(; nsrc=2, nrec=120)
    xrec = range(50f0, stop=1150f0, length=nrec)
    yrec = 0f0
    zrec = range(50f0, stop=50f0, length=nrec)
    return Geometry(xrec, yrec, zrec; dt=4f0, t=1000f0, nsrc=nsrc)
end

# number of sources/receivers
nsrc = 2
nrec = 120
ns = 251

ftol = 1f-6

@testset "judiVStack Unit Tests" begin

    # set up judiVector fr,om array
    dsize = (nsrc*nrec*ns, 1)
    rec_geometry = example_rec_geometry(nsrc=nsrc, nrec=nrec)
    data = randn(Float32, ns, nrec)
    d_obs = judiVector(rec_geometry, data)
    w0 = judiWeights(rand(10, 10); nsrc=nsrc)

    # Composite objs
    c1 = [d_obs; w0]
    c1_b = deepcopy(c1)
    c2 = [w0; d_obs]
    c2_b = deepcopy(c2)

    @test isapprox(c1.components[1], d_obs)
    @test isapprox(c2.components[2], d_obs)
    @test isapprox(c2.components[1], w0)
    @test isapprox(c1.components[2], w0)

    @test isapprox(c1.components[1], c1_b.components[1])
    @test isapprox(c1.components[2], c1_b.components[2])
    @test isapprox(c2.components[1], c2_b.components[1])
    @test isapprox(c2.components[2], c2_b.components[2])
    
    @test size(c1, 1) == length(d_obs) + length(w0)
    @test size(c2, 1) == length(d_obs) + length(w0)
    @test size(c1, 2) == 1
    @test size(c2, 2) == 1

    # Transpose
    c1_a = transpose(c1)
    @test size(c1_a, 1) == 1
    @test size(c1_a, 2) == length(d_obs) + length(w0)
    @test isapprox(c1_a.components[1],d_obs)
    @test isapprox(c1_a.components[2], w0)

    # Adjoint
    c1_a = adjoint(c1)
    @test size(c1_a, 1) == 1
    @test size(c1_a, 2) == length(d_obs) + length(w0)
    @test isapprox(c1_a.components[1], d_obs)
    @test isapprox(c1_a.components[2], w0)

    # Conj
    c1_a = conj(c1)
    @test size(c1_a, 2) == 1
    @test size(c1_a, 1) == length(d_obs) + length(w0)
    @test isapprox(c1_a.components[1], d_obs)
    @test isapprox(c1_a.components[2], w0)

    # Test arithithmetic
    u =  [d_obs; w0]
    v =  [2f0 * d_obs; w0 + 1f0]
    w =  [d_obs + 1f0; 2f0 * w0]
    a = randn(Float32, 1)[1]
    b = randn(Float32, 1)[1]

    @test isapprox(u + (v + w), (u + v) + w; rtol=ftol)
    @test isapprox(2f0*u, 2f0.*u; rtol=ftol)
    @test isapprox(u + v, v + u; rtol=ftol)
    @test isapprox(u, u + 0; rtol=ftol)
    @test iszero(norm(u + u*(-1)))
    @test isapprox(-u, -1f0 * u; rtol=ftol)
    @test isapprox(a .* (b .* u), (a * b) .* u; rtol=ftol)
    @test isapprox(u, u .* 1; rtol=ftol)
    @test isapprox(a .* (u + v), a .* u + a .* v; rtol=ftol)
    @test isapprox((a + b) .* v, a .* v + b .* v; rtol=ftol)

    # Test the norm
    u_scale = deepcopy(u)
    @test isapprox(norm(u_scale, 2), sqrt(norm(d_obs, 2)^2 + norm(w0, 2)^2))
    @test isapprox(norm(u_scale, 1), norm(d_obs, 1) + norm(w0, 1))
    @test isapprox(norm(u_scale, Inf), max(norm(d_obs, Inf), norm(w0, Inf)))
# Test broadcasting
    u_scale = deepcopy(u)
    v_scale = deepcopy(v)

    u_scale .*= 2f0
    @test isapprox(u_scale, 2f0 * u)
    v_scale .+= 2f0
    @test isapprox(v_scale, 2f0 + v)
    u_scale ./= 2f0
    @test isapprox(u_scale, u)
    u_scale .= 2f0 .* u_scale .+ v_scale
    @test isapprox(u_scale, 2f0 * u + 2f0 + v)

    # Test multi-vstack
    u1 = [u; 2f0*d_obs]
    u2 = [3f0*d_obs; u]

    @test size(u2) == size(u1)
    @test size(u2, 1) == u.m + length(d_obs)

    @test isapprox(u1.components[1], d_obs)
    @test isapprox(u1.components[2], w0)
    @test isapprox(u1.components[3], 2f0*d_obs)

    @test isapprox(u2.components[1], 3f0*d_obs)
    @test isapprox(u2.components[2], d_obs)
    @test isapprox(u2.components[3], w0)

    v1 = [u; 2f0 * w0]
    v2 = [3f0 * w0; u]

    @test size(v2) == size(v1)
    @test size(v2, 1) == u.m + length(w0)

    @test isapprox(v1.components[1], d_obs)
    @test isapprox(v1.components[2], w0)
    @test isapprox(v1.components[3], 2f0*w0)

    @test isapprox(v2.components[1], 3f0*w0)
    @test isapprox(v2.components[2], d_obs)
    @test isapprox(v2.components[3], w0)

    w1 = [c1; 2f0.*c2]
    w2 = [c2./2f0; c1]

    @test size(w1) == size(w2)
    @test size(w2, 1) == c1.m + c2.m

    @test isapprox(w1.components[1], d_obs)
    @test isapprox(w1.components[2], w0)
    @test isapprox(w1.components[3], 2f0*w0)
    @test isapprox(w1.components[4], 2f0*d_obs)

    @test isapprox(w2.components[1], w0 / 2f0)
    @test isapprox(w2.components[2], d_obs / 2f0)
    @test isapprox(w2.components[3], d_obs)
    @test isapprox(w2.components[4], w0)

end