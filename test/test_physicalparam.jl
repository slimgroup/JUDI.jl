using JUDI.TimeModeling, LinearAlgebra, Test

ftol = 1f-5

@testset "PhysicalParameter Unit Tests in $(nd) dimensions" for nd=[2, 3]
    n = Tuple(11 for i=1:nd)
    d = Tuple(1. for i=1:nd)
    o = Tuple(0. for i=1:nd)
    a = randn(Float32, n...)
    p = PhysicalParameter(a, d, o)

    @test size(p) == (prod(n), 1)
    @test size(conj(p)) == (prod(n), 1)
    @test size(adjoint(p)) == (prod(n), 1)
    @test size(transpose(p)) == (prod(n), 1)
    @test isequal(p.data, a)
    @test p.d == d
    @test p.n == n
    @test p.o == o

    p2 = similar(p)
    @test p2.n == p.n
    @test norm(p2) == 0

    p2 = copy(p)
    @test p2.n == p.n
    @test norm(p2) == norm(p)
    
    p2 = 0f0 * p + 1f0
    @test norm(p2, 1) == prod(n)


    # broadcast multiplication
    u = p
    v = p2
    u_scale = deepcopy(p)
    v_scale = deepcopy(p2)
    c = ones(Float32, v.n)

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
    @test isapprox(u_scale.data[1], u.data[1].*v.data[1])
    u_scale .= u ./ v
    @test isapprox(u_scale.data[1], u.data[1]./v.data[1])

    @test isapprox(u.*c, u)
    @test isapprox(c.*u, u)
    @test isapprox(u./c, u)
    @test isapprox(c./u, 1 ./u)
    @test isapprox(c .+ u, 1 .+ u)
    @test isapprox(c .- u, 1 .- u)
    @test isapprox(u .+ c, 1 .+ u)
    @test isapprox(u .- c, u .- 1)

    # Arithmetic 
    v = PhysicalParameter(randn(Float32, n...), d, o)
    w = PhysicalParameter(randn(Float32, n...), d, o)
    a = .5f0 + rand(Float32)
    b = .5f0 + rand(Float32)
    c2 = randn(size(v, 1), size(v,1))

    @test isapprox(u + (v + w), (u + v) + w; rtol=ftol)
    @test isapprox(u + v, v + u; rtol=ftol)
    @test isapprox(u, u + 0; rtol=ftol)
    @test iszero(norm(u + u*(-1)))
    @test isapprox(-u, (-1f0)*u; rtol=ftol)
    @test isapprox(a .* (b .* u), (a * b) .* u; rtol=ftol)
    @test isapprox(u, u .* 1; rtol=ftol)
    @test isapprox(a .* (u + v), a .* u + a .* v; rtol=1f-5)
    @test isapprox((a + b) .* v, a .* v + b.* v; rtol=1f-5)
    @test isapprox(c2*v, c2*vec(v.data))
    @test isapprox(c2\v, c2\vec(v.data))

end