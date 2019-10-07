# Unit tests for judiVector
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#

using JUDI.TimeModeling, SegyIO, Test, LinearAlgebra
import LinearAlgebra.BLAS.axpy!

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

################################################# test constructors ####################################################

@testset "judiVector Unit Tests" begin

    # set up judiVector fr,om array
    dsize = (nsrc*nrec*ns, 1)
    rec_geometry = example_rec_geometry(nsrc=nsrc, nrec=nrec)
    data = randn(Float32, ns, nrec)
    d_obs = judiVector(rec_geometry, data)

    @test isequal(d_obs.nsrc, nsrc)
    @test isequal(typeof(d_obs.data), Array{Array, 1})
    @test isequal(typeof(d_obs.geometry), GeometryIC)
    @test iszero(norm(d_obs.data[1] - d_obs.data[2]))
    @test isequal(size(d_obs), dsize)

    # set up judiVector from cell array
    data = Array{Array}(undef, nsrc)
    for j=1:nsrc
        data[j] = randn(Float32, 251, nrec)
    end
    d_obs =  judiVector(rec_geometry, data)

    @test isequal(d_obs.nsrc, nsrc)
    @test isequal(typeof(d_obs.data), Array{Array, 1})
    @test isequal(typeof(d_obs.geometry), GeometryIC)
    @test iszero(norm(d_obs.data - d_obs.data))
    @test isequal(size(d_obs), dsize)

    # contructor for in-core data container
    block = segy_read("../data/unit_test_shot_records.segy")
    d_block = judiVector(block; segy_depth_key="RecGroupElevation")
    dsize = (prod(size(block.data)), 1)

    @test isequal(d_block.nsrc, nsrc)
    @test isequal(typeof(d_block.data), Array{Array, 1})
    @test isequal(typeof(d_block.geometry), GeometryIC)
    @test isequal(size(d_block), dsize)

    # contructor for in-core data container and given geometry
    rec_geometry = Geometry(block; key="receiver", segy_depth_key="RecGroupElevation")
    d_block = judiVector(rec_geometry, block)

    @test isequal(d_block.nsrc, nsrc)
    @test isequal(typeof(d_block.data), Array{Array, 1})
    @test isequal(typeof(d_block.geometry), GeometryIC)
    @test isequal(rec_geometry, d_block.geometry)
    @test isequal(size(d_block), dsize)

    # contructor for out-of-core data container from single container
    container = segy_scan("../data/", "unit_test_shot_records", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
    d_cont = judiVector(container; segy_depth_key="RecGroupElevation")

    @test isequal(d_cont.nsrc, nsrc)
    @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
    @test isequal(typeof(d_cont.geometry), GeometryOOC)
    @test isequal(size(d_cont), dsize)

    # contructor for single out-of-core data container and given geometry
    d_cont = judiVector(rec_geometry, container)

    @test isequal(d_cont.nsrc, nsrc)
    @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
    @test isequal(typeof(d_cont.geometry), GeometryIC)
    @test isequal(rec_geometry, d_cont.geometry)
    @test isequal(size(d_cont), dsize)

    # contructor for out-of-core data container from cell array of containers
    container_cell = Array{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        container_cell[j] = split(container, j)
    end
    d_cont =  judiVector(container_cell; segy_depth_key="RecGroupElevation")

    @test isequal(d_cont.nsrc, nsrc)
    @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
    @test isequal(typeof(d_cont.geometry), GeometryOOC)
    @test isequal(size(d_cont), dsize)

    # contructor for out-of-core data container from cell array of containers and given geometry
    d_cont =  judiVector(rec_geometry, container_cell)

    @test isequal(d_cont.nsrc, nsrc)
    @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
    @test isequal(typeof(d_cont.geometry), GeometryIC)
    @test isequal(rec_geometry, d_cont.geometry)
    @test isequal(size(d_cont), dsize)


#################################################### test operations ###################################################

    # conj, transpose, adjoint
    @test isequal(size(d_obs), size(conj(d_obs)))
    @test isequal(size(d_block), size(conj(d_block)))
    @test isequal(size(d_cont), size(conj(d_cont)))

    @test isequal(reverse(size(d_obs)), size(transpose(d_obs)))
    @test isequal(reverse(size(d_block)), size(transpose(d_block)))
    @test isequal(reverse(size(d_cont)), size(transpose(d_cont)))

    @test isequal(reverse(size(d_obs)), size(adjoint(d_obs)))
    @test isequal(reverse(size(d_block)), size(adjoint(d_block)))
    @test isequal(reverse(size(d_cont)), size(adjoint(d_cont)))

    # +, -, *, /
    @test iszero(norm(2*d_obs - (d_obs + d_obs)))
    @test iszero(norm(d_obs - (d_obs + d_obs)/2))

    @test iszero(norm(2*d_block - (d_block + d_block)))
    @test iszero(norm(d_block - (d_block + d_block)/2))

    @test iszero(norm(2*d_cont - (d_cont + d_cont)))    # creates in-core judiVector
    @test iszero(norm(1*d_cont - (d_cont + d_cont)/2))

    # vcat
    d_vcat = [d_block; d_block]
    @test isequal(length(d_vcat), 2*length(d_block))
    @test isequal(d_vcat.nsrc, 2*d_block.nsrc)
    @test isequal(d_vcat.geometry.xloc[1], d_block.geometry.xloc[1])

    # dot, norm, abs
    @test isapprox(norm(d_block), sqrt(dot(d_block, d_block)))
    @test isapprox(norm(d_cont), sqrt(dot(d_cont, d_cont)))
    @test isapprox(abs.(d_block.data[1]), abs(d_block).data[1]) # need to add iterate for JUDI vector

    # vector space axioms
    u = judiVector(rec_geometry, randn(Float32, ns, nrec))
    v = judiVector(rec_geometry, randn(Float32, ns, nrec))
    w = judiVector(rec_geometry, randn(Float32, ns, nrec))
    a = randn(1)[1]
    b = randn(1)[1]

    @test isapprox(u + (v + w), (u + v) + w; rtol=eps(1f0))
    @test isapprox(u + v, v + u; rtol=eps(1f0))
    #@test isapprox(u, u + 0; rtol=eps(1f0))
    @test iszero(norm(u + u*(-1)))
    @test isapprox(a .* (b .* u), (a * b) .* u; rtol=eps(1f0))
    @test isapprox(u, u .* 1; rtol=eps(1f0))
    @test isapprox(a .* (u + v), a .* u + a .* v; rtol=eps(1f0))
    @test isapprox((a + b) .* v, a .* v + b.* v; rtol=eps(1f0))

    # subsamling
    d_block_sub = d_block[1]
    @test isequal(d_block_sub.nsrc, 1)
    @test isequal(typeof(d_block_sub.geometry), GeometryIC)
    @test isequal(typeof(d_block_sub.data), Array{Array, 1})

    d_block_sub = d_block[1:2]
    @test isequal(d_block_sub.nsrc, 2)
    @test isequal(typeof(d_block_sub.geometry), GeometryIC)
    @test isequal(typeof(d_block_sub.data), Array{Array, 1})

    d_cont =  judiVector(container_cell; segy_depth_key="RecGroupElevation")
    d_cont_sub = d_cont[1]
    @test isequal(d_cont_sub.nsrc, 1)
    @test isequal(typeof(d_cont_sub.geometry), GeometryOOC)
    @test isequal(typeof(d_cont_sub.data), Array{SegyIO.SeisCon, 1})

    d_cont_sub = d_cont[1:2]
    @test isequal(d_cont_sub.nsrc, 2)
    @test isequal(typeof(d_cont_sub.geometry), GeometryOOC)
    @test isequal(typeof(d_cont_sub.data), Array{SegyIO.SeisCon, 1})

    # Conversion to SegyIO.Block
    src_geometry = Geometry(block; key="source", segy_depth_key="SourceSurfaceElevation")
    wavelet = randn(Float32, src_geometry.nt[1])
    q = judiVector(src_geometry, wavelet)
    block_out =  judiVector_to_SeisBlock(d_block, q; source_depth_key="SourceSurfaceElevation", receiver_depth_key="RecGroupElevation")

    @test isapprox(block.data, block_out.data)
    @test isapprox(get_header(block, "SourceX"), get_header(block_out, "SourceX"); rtol=1f-6)
    @test isapprox(get_header(block, "GroupX"), get_header(block_out, "GroupX"); rtol=1f-6)
    @test isapprox(get_header(block, "RecGroupElevation"), get_header(block_out, "RecGroupElevation"); rtol=1f-6)
    @test isequal(get_header(block, "ns"), get_header(block_out, "ns"))
    @test isequal(get_header(block, "dt"), get_header(block_out, "dt"))

    # Time interpolation (inplace)
    dt_orig = 2f0
    dt_new = 1f0
    nt_orig = 501
    nt_new = 1001
    d_resample = deepcopy(d_block)
    time_resample!(d_resample, dt_new; order=2)

    @test isequal(d_resample.geometry.dt[1], dt_new)
    @test isequal(d_resample.geometry.nt[1], nt_new)
    @test isequal(size(d_resample.data[1])[1], nt_new)

    time_resample!(d_resample, dt_orig; order=2)
    @test isapprox(d_resample, d_block)

    # Time interpolation (w/ deepcopy)
    d_resample = time_resample(d_block, dt_new; order=2)

    @test isequal(d_block.geometry.dt[1], dt_orig)
    @test isequal(d_block.geometry.nt[1], nt_orig)
    @test isequal(size(d_block.data[1])[1], nt_orig)
    @test isequal(d_resample.geometry.dt[1], dt_new)
    @test isequal(d_resample.geometry.nt[1], nt_new)
    @test isequal(size(d_resample.data[1])[1], nt_new)

    d_recover = time_resample(d_resample, dt_orig; order=2)
    @test isapprox(d_recover, d_block)

    # Time interpolation (linear operator)
    I = judiTimeInterpolation(d_block.geometry, dt_orig, dt_new)
    d_resample = I*d_block

    @test isequal(d_block.geometry.dt[1], dt_orig)
    @test isequal(d_block.geometry.nt[1], nt_orig)
    @test isequal(size(d_block.data[1])[1], nt_orig)
    @test isequal(d_resample.geometry.dt[1], dt_new)
    @test isequal(d_resample.geometry.nt[1], nt_new)
    @test isequal(size(d_resample.data[1])[1], nt_new)

    d_recover = transpose(I)*d_resample
    @test isapprox(d_recover, d_block)

    # scale
    a = randn(Float32, 1)[1]
    d_scale = deepcopy(d_block)

    # broadcast multiplication
    u = judiVector(rec_geometry, randn(Float32, ns, nrec))
    v = judiVector(rec_geometry, randn(Float32, ns, nrec))
    u_scale = deepcopy(u)
    v_scale = deepcopy(v)
    a = randn(1)[1]

    # broadcast identity
    u = judiVector(rec_geometry, randn(Float32, ns, nrec))
    v = judiVector(rec_geometry, randn(Float32, ns, nrec))
    u_id = deepcopy(u)
    v_id = deepcopy(v)
    broadcast!(identity, u_id, v_id)    # copy v_id into u_id

    @test isapprox(v, v_id)
    @test isapprox(v, u_id)

    # broadcast scaling + addition
    u = judiVector(rec_geometry, randn(Float32, ns, nrec))
    v = judiVector(rec_geometry, randn(Float32, ns, nrec))
    w = judiVector(rec_geometry, randn(Float32, ns, nrec))
    u_add = deepcopy(u)
    v_add = deepcopy(v)
    w_add = deepcopy(w)
    a = randn(1)[1]

    # in-place overwrite
    u = judiVector(rec_geometry, randn(Float32, ns, nrec))
    v = judiVector(rec_geometry, randn(Float32, ns, nrec))
    u_cp = deepcopy(u)
    v_cp = deepcopy(v)
    copy!(v_cp, u_cp)

    @test isapprox(u, u_cp)
    @test isapprox(u, v_cp)

    # similar
    d_zero = similar(d_block, Float32)

    @test isequal(d_zero.geometry, d_block.geometry)
    @test isequal(size(d_zero), size(d_block))
    @test iszero(d_zero.data[1])

    # retrieve out-of-core data
    d_get = get_data(d_cont)
    @test isapprox(d_block, d_get)

end
