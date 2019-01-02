# Unit tests for judiRHS and judiWavefield (without PDE solves)
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#

using JUDI.TimeModeling, SeisIO, Test, LinearAlgebra

# Example structures

example_info(; n=(120,100), nsrc=2, ntComp=1000) = Info(prod(n), nsrc, ntComp)
example_model(; n=(120,100), d=(10f0, 10f0), o=(0f0, 0f0), m=randn(Float32, n)) = Model(n, d, o, m)

function example_rec_geometry(; nsrc=2, nrec=120)
    xrec = range(50f0, stop=1150f0, length=nrec)
    yrec = 0f0
    zrec = range(50f0, stop=50f0, length=nrec)
    return Geometry(xrec, yrec, zrec; dt=4f0, t=1000f0, nsrc=nsrc)
end

function example_src_geometry(; nsrc=2)
    xrec = range(100f0, stop=1000f0, length=nsrc)
    yrec = 0f0
    zrec = range(50f0, stop=50f0, length=nsrc)
    return Geometry(xrec, yrec, zrec; dt=4f0, t=1000f0, nsrc=nsrc)
end


########################################################### judiRHS ####################################################

@testset "judiRHS Unit test" begin

    # Constructor
    nsrc = 2
    info = example_info(nsrc=nsrc)
    rec_geometry = example_rec_geometry(nsrc=nsrc)
    data = Array{Array}(undef, nsrc)
    for j=1:nsrc
        data[j] = randn(Float32, rec_geometry.nt[j], length(rec_geometry.xloc[j]))
    end
    rhs = judiRHS(info, rec_geometry, data)

    @test isequal(typeof(rhs), judiRHS{Float32})
    @test isequal(rhs.geometry, rec_geometry)

    # conj, transpose, adjoint
    @test isequal(size(rhs), size(conj(rhs)))
    @test isequal(reverse(size(rhs)), size(transpose(rhs)))
    @test isequal(reverse(size(rhs)), size(adjoint(rhs)))

    # +, -
    info = example_info(nsrc=nsrc)
    rec_geometry = example_rec_geometry(nsrc=nsrc)
    src_geometry = example_src_geometry(nsrc=nsrc)
    data1 = Array{Array}(undef, nsrc)
    data2 = Array{Array}(undef, nsrc)
    for j=1:nsrc
        data1[j] = randn(Float32, rec_geometry.nt[j], length(rec_geometry.xloc[j]))
        data2[j] = randn(Float32, src_geometry.nt[j], length(src_geometry.xloc[j]))
    end
    rhs1 = judiRHS(info, rec_geometry, data1)
    rhs2 = judiRHS(info, src_geometry, data2)

    rhs_sum = rhs1 + rhs2
    rhs_sub = rhs1 - rhs2

    @test isequal(size(rhs_sum), size(rhs1))
    @test isequal(size(rhs_sub), size(rhs1))

    @test isequal(length(rhs_sum.geometry.xloc[1]), length(rhs1.geometry.xloc[1]) + length(rhs2.geometry.xloc[1]))
    @test isequal(length(rhs_sub.geometry.xloc[1]), length(rhs1.geometry.xloc[1]) + length(rhs2.geometry.xloc[1]))

    @test isequal(size(rhs_sum.data[1])[2], size(rhs1.data[1])[2] + size(rhs2.data[1])[2])
    @test isequal(size(rhs_sub.data[1])[2], size(rhs1.data[1])[2] + size(rhs2.data[1])[2])

    # get index
    rhs_sub = rhs[1]
    @test isequal(rhs_sub.info.nsrc, 1)
    @test isequal(typeof(rhs_sub.geometry), GeometryIC)
    @test isequal(typeof(rhs.data), Array{Array, 1})
    @test isequal(length(rhs_sub), Int(length(rhs)/2))

    rhs_sub = rhs[1:2]
    @test isequal(rhs_sub.info.nsrc, 2)
    @test isequal(typeof(rhs_sub.geometry), GeometryIC)
    @test isequal(typeof(rhs.data), Array{Array, 1})
    @test isequal(length(rhs_sub), length(rhs))

end

######################################################### judiWavefield ################################################
