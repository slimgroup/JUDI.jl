# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

ftol = 1f-5

@testset "PhysicalParameter Unit Tests in $(nd) dimensions" for nd=[2, 3]
    n = Tuple(11+i for i=1:nd)
    d = Tuple(1f0+i for i=1:nd)
    o = Tuple(0f0+i for i=1:nd)
    a = randn(Float32, n...)
    p = PhysicalParameter(a, d, o)
    @test PhysicalParameter(p) == p
    @test PhysicalParameter(p, n, d, o) == p
    @test PhysicalParameter(p.data, n, d, o) == p
    @test PhysicalParameter(vec(p.data), n, d, o) == p

    @test size(p) == (prod(n), 1)
    @test size(conj(p)) == (prod(n), 1)

    @test size(adjoint(p)) == (prod(n), 1)
    @test adjoint(p).n == p.n[end:-1:1]
    @test adjoint(p).d == p.d[end:-1:1]
    @test adjoint(p).o == p.o[end:-1:1]
    @test isapprox(adjoint(p).data, permutedims(p.data, nd:-1:1))

    @test size(transpose(p)) == (prod(n), 1)
    @test transpose(p).n == p.n[end:-1:1]
    @test transpose(p).d == p.d[end:-1:1]
    @test transpose(p).o == p.o[end:-1:1]
    @test isapprox(transpose(p).data, permutedims(p.data, nd:-1:1))

    ps = similar(p, Model(n, d, o, a))
    @test norm(ps) == 0
    @test ps.n == n
    @test ps.d == d
    @test ps.o == o

    @test isequal(p.data, a)
    p .= zeros(n...)
    p .= a
    @test isequal(p.data, a)
    @test p.d == d
    @test p.n == n
    @test p.o == o

    # indexing
    @test firstindex(p) == 1
    @test lastindex(p) == prod(n)
    @test p == p
    @test isapprox(p, a)
    @test isapprox(a, p)

    # Subsampling
    inds = [3:11 for i=1:nd]
    b = p[inds...]
    @test b.o == tuple([o+d*2 for (o,d)=zip(p.o,p.d)]...)
    @test b.n == tuple([9 for i=1:nd]...)
    @test b.d == p.d

    inds = [3:2:11 for i=1:nd]
    b = p[inds...]
    @test b.o == tuple([o+d*2 for (o,d)=zip(p.o,p.d)]...)
    @test b.n == tuple([5 for i=1:nd]...)
    @test b.d == tuple([d*2 for d=p.d]...)

    # copies
    p2 = similar(p)
    @test p2.n == p.n
    @test norm(p2) == 0

    p2 = copy(p)
    @test p2.n == p.n
    @test norm(p2) == norm(p)
    
    p2 = 0f0 .* p .+ 1f0
    @test norm(p2, 1) == prod(n)

    # Some basics
    pl = PhysicalParameter(n, d, o)
    @test norm(pl) == 0
    pl = PhysicalParameter(0, n, d, o)
    @test pl.data == 0
    @test isapprox(dot(p, p), norm(p)^2)
    @test isapprox(dot(ones(n), p), sum(p.data))
    @test isapprox(dot(p, ones(n)), sum(p.data))

    # broadcast multiplication
    u = p
    v = p2
    u_scale = deepcopy(p)
    v_scale = deepcopy(p2)
    c = ones(Float32, v.n)

    u_scale .*= 2f0
    @test isapprox(u_scale, 2f0 * u; rtol=ftol)
    v_scale .+= 2f0
    @test isapprox(v_scale, 2f0 .+ v; rtol=ftol)
    u_scale ./= 2f0
    @test isapprox(u_scale, u; rtol=ftol)
    u_scale .= 2f0 .* u_scale .+ v_scale
    @test isapprox(u_scale, 2f0 * u .+ 2f0 + v; rtol=ftol)
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
    @test isapprox(u, u .+ 0; rtol=ftol)
    @test iszero(norm(u + u.*(-1)))
    @test isapprox(-u, (-1f0).*u; rtol=ftol)
    @test isapprox(a .* (b .* u), (a * b) .* u; rtol=ftol)
    @test isapprox(u, u .* 1; rtol=ftol)
    @test isapprox(a .* (u + v), a .* u + a .* v; rtol=1f-5)
    @test isapprox((a + b) .* v, a .* v + b.* v; rtol=1f-5)
    @test isapprox(c2*v, c2*vec(v.data))
    @test isapprox(c2\v, c2\vec(v.data))
    @test isapprox(c + v, 1 .+ v)
    @test isapprox(c - v, 1 .- v)
    @test isapprox(v + c, 1 .+ v)
    @test isapprox(v - c, v .- 1)
    @test isapprox(v * v, v.^2)
    @test isapprox(v / v, 0f0.*v .+1)

    a = zeros(Float32, n...)
    a .= v
    @test isapprox(a, v.data)
    v = PhysicalParameter(ones(Float32, n...), d, o)
    v[v .> 0] .= -1
    @test all(v .== -1)

    # Indexing
    u = PhysicalParameter(randn(n), d, o)
    u[:] .= 0f0
    @test norm(u) == 0f0

    u[1] = 1f0
    @test norm(u) == 1f0
    @test u.data[1] == 1f0

    u[1:10] .= 1:10
    @test norm(u, 1) == 55
    @test u[1:10] == 1:10
    @test u.data[1:10] == 1:10
    @test u[11] == 0f0

    if nd == 2
        tmp = randn(u[1:10, :].n)
        u[1:10, :] .= tmp
        @test u.data[1:10, :] == tmp
        @test size(u[:, 1]) == (n[1],)
        @test size(u[1, :]) == (n[2],)
        @test u[2:end, 2:3].n == (n[1]-1, 2)
        u[:, 1] .= ones(Float32, n[1])
        @test all(u.data[:, 1] .== 1f0)
    else
        tmp = randn(u[1:10, :, :].n)
        u[1:10, :, :] .= tmp
        @test u.data[1:10, :, :] == tmp
        @test size(u[:, :, 1]) == (n[1], n[2])
        @test size(u[:, 1, :]) == (n[1], n[3])
        @test size(u[1, :, :]) == (n[2], n[3])
        @test size(u[1, 2:3, 1:end]) == (2, n[3])
        @test u[1:2, 2:3, 1:end].n == (2, 2, n[3])
        u[:, 1:end-2, 1] .= ones(Float32, n[1])
        @test all(u.data[:, 1:end-2, 1] .== 1f0)
    end

    # Test that works with JOLI
    A = joEye(size(v, 1))
    u2 = A*u
    @test isequal(u2, vec(u.data))
    u2 = [A;A]*u
    @test length(u2) == 2 * length(u)
    @test u2[1:length(u)] == vec(u.data)
    @test u2[length(u)+1:end] == vec(u.data)

end