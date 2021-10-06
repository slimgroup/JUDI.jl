# Unit tests for judiVector
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

datapath = joinpath(dirname(pathof(JUDI)))*"/../data/"

# number of sources/receivers
nrec = 120
ns = 251
ftol = 1e-6

################################################# test constructors ####################################################

@testset "judiVector Unit Tests with $(nsrc) sources" for nsrc=[1, 2]

    # set up judiVector from array
    info = example_info(nsrc=nsrc)
    dsize = (nsrc*nrec*ns, 1)
    rec_geometry = example_rec_geometry(nsrc=nsrc, nrec=nrec)
    data = randn(Float32, ns, nrec)
    d_obs = judiVector(rec_geometry, data)
    @test typeof(d_obs) == judiVector{Float32, Array{Float32, 2}}
    @test isequal(process_input_data(d_obs, rec_geometry, info), d_obs.data)

    @test isequal(d_obs.nsrc, nsrc)
    @test isequal(typeof(d_obs.data), Array{Array{Float32, 2}, 1})
    @test isequal(typeof(d_obs.geometry), GeometryIC{Float32})
    @test iszero(norm(d_obs.data[1] - d_obs.data[end]))
    @test isequal(size(d_obs), dsize)

    # set up judiVector from cell array
    data = Array{Array{Float32, 2}, 1}(undef, nsrc)
    for j=1:nsrc
        data[j] = randn(Float32, ns, nrec)
    end
    d_obs =  judiVector(rec_geometry, data)

    @test isequal(d_obs.nsrc, nsrc)
    @test isequal(typeof(d_obs.data), Array{Array{Float32, 2}, 1})
    @test isequal(typeof(d_obs.geometry), GeometryIC{Float32})
    @test iszero(norm(d_obs.data - d_obs.data))
    @test isequal(size(d_obs), dsize)
    @test isapprox(convert_to_array(d_obs), vcat([vec(d) for d in data]...); rtol=ftol)

    # contructor for in-core data container
    block = segy_read(datapath*"unit_test_shot_records_$(nsrc).segy"; warn_user=false)
    d_block = judiVector(block; segy_depth_key="RecGroupElevation")
    dsize = (prod(size(block.data)), 1)

    @test isequal(d_block.nsrc, nsrc)
    @test isequal(typeof(d_block.data), Array{Array{Float32, 2}, 1})
    @test isequal(typeof(d_block.geometry), GeometryIC{Float32})
    @test isequal(size(d_block), dsize)


    # contructor for in-core data container and given geometry
    rec_geometry = Geometry(block; key="receiver", segy_depth_key="RecGroupElevation")
    d_block = judiVector(rec_geometry, block)

    @test isequal(d_block.nsrc, nsrc)
    @test isequal(typeof(d_block.data), Array{Array{Float32, 2}, 1})
    @test isequal(typeof(d_block.geometry), GeometryIC{Float32})
    @test isequal(rec_geometry, d_block.geometry)
    @test isequal(size(d_block), dsize)

    # contructor for out-of-core data container from single container
    container = segy_scan(datapath, "unit_test_shot_records_$(nsrc)", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
    d_cont = judiVector(container; segy_depth_key="RecGroupElevation")

    @test typeof(d_cont) == judiVector{Float32, SeisCon}
    @test isequal(d_cont.nsrc, nsrc)
    @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
    @test isequal(typeof(d_cont.geometry), GeometryOOC{Float32})
    @test isequal(size(d_cont), dsize)

    # contructor for single out-of-core data container and given geometry
    d_cont = judiVector(rec_geometry, container)

    @test isequal(d_cont.nsrc, nsrc)
    @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
    @test isequal(typeof(d_cont.geometry), GeometryIC{Float32})
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
    @test isequal(typeof(d_cont.geometry), GeometryOOC{Float32})
    @test isequal(size(d_cont), dsize)

    # contructor for out-of-core data container from cell array of containers and given geometry
    d_cont =  judiVector(rec_geometry, container_cell)

    @test isequal(d_cont.nsrc, nsrc)
    @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
    @test isequal(typeof(d_cont.geometry), GeometryIC{Float32})
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

    # lmul!, rmul!, ldiv!, rdiv!
    data_ones = Array{Array{Float32, 2}, 1}(undef, nsrc)
    for j=1:nsrc
        data_ones[j] = ones(Float32, ns, nrec)
    end
    d1 =  judiVector(rec_geometry, data_ones)
    lmul!(2f0, d1)
    @test all([all(d1.data[i] .== 2f0) for i = 1:d1.nsrc])
    rmul!(d1, 3f0)
    @test all([all(d1.data[i] .== 6f0) for i = 1:d1.nsrc])
    ldiv!(2f0,d1)
    @test all([all(d1.data[i] .== 3f0) for i = 1:d1.nsrc])
    rdiv!(d1, 3f0)
    @test all([all(d1.data[i] .== 1f0) for i = 1:d1.nsrc])

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
    a = .5f0 + rand(Float32)
    b = .5f0 + rand(Float32)

    @test isapprox(u + (v + w), (u + v) + w; rtol=ftol)
    @test isapprox(u + v, v + u; rtol=ftol)
    @test isapprox(-u, -1f0 * u; rtol=ftol)
    #@test isapprox(u, u + 0; rtol=ftol)
    @test iszero(norm(u + u*(-1)))
    @test isapprox(a .* (b .* u), (a * b) .* u; rtol=ftol)
    @test isapprox(u, u .* 1; rtol=ftol)
    @test isapprox(a .* (u + v), a .* u + a .* v; rtol=1f-5)
    @test isapprox((a + b) .* v, a .* v + b .* v; rtol=1f-5)

    # subsamling
    d_block_sub = d_block[1]
    @test isequal(d_block_sub.nsrc, 1)
    @test isequal(typeof(d_block_sub.geometry), GeometryIC{Float32})
    @test isequal(typeof(d_block_sub.data), Array{Array{Float32, 2}, 1})

    inds = nsrc > 1 ? (1:nsrc) : 1
    d_block_sub = d_block[inds]
    @test isequal(d_block_sub.nsrc, nsrc)
    @test isequal(typeof(d_block_sub.geometry), GeometryIC{Float32})
    @test isequal(typeof(d_block_sub.data), Array{Array{Float32, 2}, 1})

    d_cont =  judiVector(container_cell; segy_depth_key="RecGroupElevation")
    d_cont_sub = d_cont[1]
    @test isequal(d_cont_sub.nsrc, 1)
    @test isequal(typeof(d_cont_sub.geometry), GeometryOOC{Float32})
    @test isequal(typeof(d_cont_sub.data), Array{SegyIO.SeisCon, 1})

    d_cont_sub = d_cont[inds]
    @test isequal(d_cont_sub.nsrc, nsrc)
    @test isequal(typeof(d_cont_sub.geometry), GeometryOOC{Float32})
    @test isequal(typeof(d_cont_sub.data), Array{SegyIO.SeisCon, 1})

    # Conversion to SegyIO.Block
    src_geometry = Geometry(block; key="source", segy_depth_key="SourceSurfaceElevation")
    wavelet = randn(Float32, src_geometry.nt[1], 1)
    q = judiVector(src_geometry, wavelet)
    block_out =  judiVector_to_SeisBlock(d_block, q; source_depth_key="SourceSurfaceElevation", receiver_depth_key="RecGroupElevation")

    @test isapprox(block.data, block_out.data)
    @test isapprox(get_header(block, "SourceX"), get_header(block_out, "SourceX"); rtol=1f-6)
    @test isapprox(get_header(block, "GroupX"), get_header(block_out, "GroupX"); rtol=1f-6)
    @test isapprox(get_header(block, "RecGroupElevation"), get_header(block_out, "RecGroupElevation"); rtol=1f-6)
    @test isequal(get_header(block, "ns"), get_header(block_out, "ns"))
    @test isequal(get_header(block, "dt"), get_header(block_out, "dt"))

    q_block = src_to_SeisBlock(q)
    q1 = judiVector(q_block)
    @test isapprox(q1, q)

    block_2d = judiVector_to_SeisBlock(d_obs, q)
    d_obs1 = judiVector(block_2d)
    @test isapprox(d_obs1.data, d_obs.data)

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
    a = .5f0 + rand(Float32)
    d_scale = deepcopy(d_block)

    # Test norms
    d_ones = judiVector(rec_geometry, 2f0 .* ones(Float32, ns, nrec))
    @test isapprox(norm(d_ones, 2), sqrt(rec_geometry.dt[1]*nsrc*ns*nrec*4))
    @test isapprox(norm(d_ones, 1), rec_geometry.dt[1]*nsrc*ns*nrec*2)
    @test isapprox(norm(d_ones, Inf), 2)

    # Indexing and utilities
    @test isfinite(d_obs)
    @test ndims(judiVector{Float32, Array{Float32, 2}}) == 1
    @test isapprox(sum(d_obs), sum(sum([vec(d) for d in d_obs])))
    d0 = copy(d_obs)
    fill!(d0, 0f0)
    @test iszero(norm(d0))

    @test firstindex(d_obs) == 1
    @test lastindex(d_obs) == nsrc
    @test axes(d_obs) == Base.OneTo(nsrc)
    @test ndims(d_obs) == 2

    d0[1] = d_obs.data[1]
    @test isapprox(d0.data[1], d_obs.data[1])

    # broadcast multiplication
    u = judiVector(rec_geometry, randn(Float32, ns, nrec))
    v = judiVector(rec_geometry, randn(Float32, ns, nrec))
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
    @test isapprox(u_scale.data[1], u.data[1].*v.data[1])
    u_scale .= u ./ v
    @test isapprox(u_scale.data[1], u.data[1]./v.data[1])

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
    a = randn(Float32, 1)[1]

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

    # Test copies/similar
    w1 = deepcopy(d_obs)
    @test isapprox(w1, d_obs)
    w1 = similar(d_obs)
    @test w1.nsrc == d_obs.nsrc
    @test isapprox(w1.data, 0f0 .* d_obs.data)
    w1 .= d_obs
    @test w1.nsrc == d_obs.nsrc
    @test isapprox(w1.data, d_obs.data)


    # Test transducer
    q = judiVector(Geometry(0f0, 0f0, 0f0; dt=2, t=1000), randn(251))
    tr = transducer(q, (10, 10), 30, pi/2)
    @test length(tr.geometry.xloc[1]) == 22
    @test tr.geometry.xloc[1][1:11] == range(-30., 30., length=11)
    @test tr.geometry.xloc[1][12:end] == range(-30., 30., length=11)
    @test all(tr.geometry.zloc[1][12:end] .== -10f0)
    @test all(tr.geometry.zloc[1][1:11] .== 0f0)

    q = judiVector(Geometry(0f0, 0f0, 0f0; dt=2, t=1000), randn(251))
    tr = transducer(q, (10, 10), 30, pi)
    @test length(tr.geometry.xloc[1]) == 22
    @test isapprox(tr.geometry.zloc[1][1:11], range(30., -30., length=11); atol=1f-14, rtol=1f-14)
    @test isapprox(tr.geometry.zloc[1][12:end], range(30., -30., length=11); atol=1f-14, rtol=1f-14)
    @test isapprox(tr.geometry.xloc[1][12:end], -10f0*ones(11); atol=1f-14, rtol=1f-14)
    @test isapprox(tr.geometry.xloc[1][1:11], zeros(11); atol=1f-14, rtol=1f-14)

    # Test integral & derivative
    refarray = Array{Array{Float32, 2}, 1}(undef, nsrc)
    for j=1:nsrc
        refarray[j] = randn(Float32, ns, nrec)
    end
    d_orig = judiVector(rec_geometry, refarray)

    dt = rec_geometry.dt[1]

    d_cumsum = cumsum(d_orig)
    for i = 1:d_orig.nsrc
        @test isapprox(dt * cumsum(refarray[i],dims=1), d_cumsum.data[i])
    end

    d_diff = diff(d_orig)
    for i = 1:d_orig.nsrc
        @test isapprox(1/dt * refarray[i][1,:], d_diff.data[i][1,:])
        @test isapprox(d_diff.data[i][2:end,:], 1/dt * diff(refarray[i],dims=1))
    end

    @test isapprox(cumsum(d_orig,dims=1),cumsum(d_orig))
    @test isapprox(diff(d_orig,dims=1),diff(d_orig))

    d_cumsum_rec = cumsum(d_orig,dims=2)
    for i = 1:d_orig.nsrc
        @test isapprox(cumsum(refarray[i],dims=2), d_cumsum_rec.data[i])
    end

    d_diff_rec = diff(d_orig,dims=2)
    for i = 1:d_orig.nsrc
        @test isapprox(refarray[i][:,1], d_diff_rec.data[i][:,1])
        @test isapprox(d_diff_rec.data[i][:,2:end], diff(refarray[i],dims=2))
    end

end
