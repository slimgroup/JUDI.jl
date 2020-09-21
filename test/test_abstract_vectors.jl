# Unit tests for judiRHS and judiWavefield (without PDE solves)
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using JUDI.TimeModeling, SegyIO, Test, LinearAlgebra

########################################################### judiRHS ####################################################

@testset "judiRHS Unit Tests with $(nsrc) sources" for nsrc=[1, 2]

    # Constructor
    info = example_info(nsrc=nsrc)
    rec_geometry = example_rec_geometry(nsrc=nsrc)
    data = Array{Array}(undef, nsrc)
    for j=1:nsrc
        data[j] = randn(Float32, rec_geometry.nt[j], length(rec_geometry.xloc[j]))
    end
    datacell = process_input_data(vec(hcat(data...)), rec_geometry, info)
    @test isequal(datacell, data)

    rhs = judiRHS(info, rec_geometry, data)
    Pr = judiProjection(info, rec_geometry)

    @test isequal(typeof(rhs), judiRHS{Float32})
    @test isequal(rhs.geometry, rec_geometry)

    rhs2 = Pr'*vec(hcat(data...))
    @test isequal(typeof(rhs2), judiRHS{Float32})
    @test isequal(rhs2.geometry, rec_geometry)
    @test isequal(rhs2.geometry, rhs.geometry)
    @test isequal(rhs2.data, rhs.data)

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
    @test isequal(length(rhs_sub), Int(length(rhs)/nsrc))

    inds = nsrc > 1 ? (1:nsrc) : 1
    rhs_sub = rhs[inds]
    @test isequal(rhs_sub.info.nsrc, nsrc)
    @test isequal(typeof(rhs_sub.geometry), GeometryIC)
    @test isequal(typeof(rhs.data), Array{Array, 1})
    @test isequal(length(rhs_sub), length(rhs))

end
