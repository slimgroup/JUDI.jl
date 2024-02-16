# Unit tests for judiVector
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

datapath = joinpath(dirname(pathof(JUDI)))*"/../data/"

# number of sources/receivers
nrec = 120
ftol = 1e-6

################################################# test constructors ####################################################

@testset "judiVector Unit Tests with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiVector (nsrc=$(nsrc))" begin
        # set up judiVector from array
        dsize = (nsrc,)
        rec_geometry = example_rec_geometry(nsrc=nsrc, nrec=nrec)
        data = randn(Float32, rec_geometry.nt[1], nrec)
        d_obs = judiVector(rec_geometry, data)
        @test typeof(d_obs) == judiVector{Float32, Array{Float32, 2}}
        @test isequal(process_input_data(d_obs, rec_geometry), d_obs)
        @test isequal(process_input_data(d_obs), d_obs.data)

        @test isequal(d_obs.nsrc, nsrc)
        @test isequal(typeof(d_obs.data), Array{Array{Float32, 2}, 1})
        @test isequal(typeof(d_obs.geometry), GeometryIC{Float32})
        @test iszero(norm(d_obs.data[1] - d_obs.data[end]))
        @test isequal(size(d_obs), dsize)

        # set up judiVector from cell array
        data = Array{Array{Float32, 2}, 1}(undef, nsrc)
        for j=1:nsrc
            data[j] = randn(Float32, rec_geometry.nt[1], nrec)
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
        dsize = (nsrc,)

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


        ########## Test t0 #######
        d_contt0 = judiVector(container; segy_depth_key="RecGroupElevation", t0=50f0)

        @test all(get_t0(d_contt0.geometry) .== 50f0)
        @test all(get_t(d_contt0.geometry) .== (get_t(d_cont.geometry) .+ 50f0))
        @test all(get_nt(d_contt0.geometry) .== get_nt(d_cont.geometry))

        # Time resampling adds back the t0 for consistent modeling
        data0 = get_data(d_contt0[1]).data[1]
        newdt = div(get_dt(d_contt0.geometry, 1), 2)
        dinterp = time_resample(data0, Geometry(d_contt0.geometry[1]), newdt)
        @test size(dinterp, 1) == 2*size(data0, 1) - 1

        if nsrc == 2
            @test_throws JUDI.judiMultiSourceException  JUDI._maybe_pad_t0(d_cont, d_contt0)
        end
        _, dinterpt0 = JUDI._maybe_pad_t0(d_cont[1], d_contt0[1])
        @test size(dinterpt0, 1) == size(data0, 1) + div(50f0, get_dt(d_contt0.geometry, 1))

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
            data_ones[j] = ones(Float32, rec_geometry.nt[1], nrec)
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
        u = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        v = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        w = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        a = .5f0 + rand(Float32)
        b = .5f0 + rand(Float32)

        @test isapprox(u + (v + w), (u + v) + w; rtol=ftol)
        @test isapprox(u + v, v + u; rtol=ftol)
        @test isapprox(-u, -1f0 * u; rtol=ftol)
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
        wavelet = randn(Float32, src_geometry.nt[1])
        q = judiVector(src_geometry, wavelet)
        block_out =  judiVector_to_SeisBlock(d_block, q; source_depth_key="SourceSurfaceElevation", receiver_depth_key="RecGroupElevation")

        @test isapprox(block.data, block_out.data)
        @test isapprox(get_header(block, "SourceX"), get_header(block_out, "SourceX"); rtol=1f-6)
        @test isapprox(get_header(block, "GroupX"), get_header(block_out, "GroupX"); rtol=1f-6)
        @test isapprox(get_header(block, "RecGroupElevation"), get_header(block_out, "RecGroupElevation"); rtol=1f-6)
        @test isequal(get_header(block, "ns"), get_header(block_out, "ns"))
        @test isequal(get_header(block, "dt"), get_header(block_out, "dt"))

        block_obs =  judiVector_to_SeisBlock(d_obs, q; source_depth_key="SourceSurfaceElevation", receiver_depth_key="RecGroupElevation")
        d_obs1 = judiVector(block_obs)
        @test isapprox(d_obs1.data, d_obs.data)

        block_q = src_to_SeisBlock(q)
        q_1 = judiVector(block_q)
        for i = 1:nsrc
            vec(q_1.data[i]) == vec(q.data[i])
        end

        # scale
        a = .5f0 + rand(Float32)
        d_scale = deepcopy(d_block)

        # Test norms
        d_ones = judiVector(rec_geometry, 2f0 .* ones(Float32, rec_geometry.nt[1], nrec))
        @test isapprox(norm(d_ones, 2), sqrt(rec_geometry.dt[1]*nsrc*rec_geometry.nt[1]*nrec*4))
        @test isapprox(norm(d_ones, 1), rec_geometry.dt[1]*nsrc*rec_geometry.nt[1]*nrec*2)
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
        @test axes(d_obs) == (Base.OneTo(nsrc),)
        @test ndims(d_obs) == 1

        d0[1] = d_obs.data[1]
        @test isapprox(d0.data[1], d_obs.data[1])

        # broadcast multiplication
        u = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        v = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
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
        u = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        v = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        u_id = deepcopy(u)
        v_id = deepcopy(v)
        broadcast!(identity, u_id, v_id)    # copy v_id into u_id

        @test isapprox(v, v_id)
        @test isapprox(v, u_id)

        # in-place overwrite
        u = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        v = judiVector(rec_geometry, randn(Float32, rec_geometry.nt[1], nrec))
        u_cp = deepcopy(u)
        v_cp = deepcopy(v)
        copy!(v_cp, u_cp)

        @test isapprox(u, u_cp)
        @test isapprox(u, v_cp)

        # similar
        d_zero = similar(d_block, Float32)

        @test isequal(d_zero.geometry, d_block.geometry)
        @test isequal(size(d_zero), size(d_block))

        # retrieve out-of-core data
        d_get = get_data(d_cont)
        @test isapprox(d_block, d_get)

        # Test copies/similar
        w1 = deepcopy(d_obs)
        @test isapprox(w1, d_obs)
        w1 = similar(d_obs)
        @test w1.nsrc == d_obs.nsrc
        w1 .= d_obs
        @test w1.nsrc == d_obs.nsrc
        @test isapprox(w1.data, d_obs.data)

        # Test transducer
        q = judiVector(Geometry(0f0, 0f0, 0f0; dt=2, t=1000), randn(Float32, 501))
        tr = transducer(q, (10, 10), 30, pi/2)
        @test length(tr.geometry.xloc[1]) == 22
        @test tr.geometry.xloc[1][1:11] == range(-30., 30., length=11)
        @test tr.geometry.xloc[1][12:end] == range(-30., 30., length=11)
        @test all(tr.geometry.zloc[1][12:end] .== -10f0)
        @test all(tr.geometry.zloc[1][1:11] .== 0f0)

        q = judiVector(Geometry(0f0, 0f0, 0f0; dt=2, t=1000), randn(Float32, 501))
        tr = transducer(q, (10, 10), 30, pi)
        @test length(tr.geometry.xloc[1]) == 22
        @test isapprox(tr.geometry.zloc[1][1:11], range(30., -30., length=11); atol=1f-14, rtol=1f-14)
        @test isapprox(tr.geometry.zloc[1][12:end], range(30., -30., length=11); atol=1f-14, rtol=1f-14)
        @test isapprox(tr.geometry.xloc[1][12:end], -10f0*ones(11); atol=1f-14, rtol=1f-14)
        @test isapprox(tr.geometry.xloc[1][1:11], zeros(11); atol=1f-14, rtol=1f-14)

        # Test exception if number of samples in geometry doesn't match ns of data
        @test_throws JUDI.judiMultiSourceException judiVector(Geometry(0f0, 0f0, 0f0; dt=2, t=1000), randn(Float32, 10))
        @test_throws JUDI.judiMultiSourceException judiVector(rec_geometry, randn(Float32, 10))        

        # Test integral & derivative
        refarray = Array{Array{Float32, 2}, 1}(undef, nsrc)
        for j=1:nsrc
            refarray[j] = randn(Float32, rec_geometry.nt[1], nrec)
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

        # test simsources with fixed rec geom
        if nsrc == 2
            dic = judiVector(rec_geometry, refarray)
            for jvec in [dic, d_obs, d_cont]
                for nsim=1:3
                    M1 = randn(Float32, nsim, 2)
                    ds = M1 * jvec
                    @test ds.nsrc == nsim
                    for s=1:nsim
                        @test ds.data[s] ≈ mapreduce((x, y)->x*y, +, M1[s,:], get_data(jvec).data)
                    end
                    gtest = Geometry(jvec.geometry[1])
                    @test all(ds.geometry[i] == gtest for i=1:nsim)
                end
            end

            # test simsources with "marine" rec geom
            refarray = randn(Float32, 251, 2)

            dic = judiVector(example_src_geometry(), [refarray[:, 1:1], refarray[:, 2:2]])
            for nsim=1:3
                M1 = randn(Float32, nsim, 2)
                ds = M1 * dic
                @test ds.nsrc == nsim
                @test all(ds.geometry.nrec .== 2)
                for s=1:nsim
                    @test ds.data[s] ≈ hcat(M1[s,1]*refarray[:, 1],M1[s, 2]*refarray[:, 2])
                end
                @test ds.geometry[1].xloc[1][1] == dic.geometry[1].xloc[1][1]
                @test ds.geometry[1].xloc[1][2] == dic.geometry[2].xloc[1][1]
            end

            # Test "simsource" without reduction
            refarray = randn(Float32, 251, 4)
            dic = judiVector(example_src_geometry(), [refarray[:, 1:1], refarray[:, 2:2]])
            M1 = randn(Float32, 2)
            @test all(dic.geometry.nrec .== 1)
            sd1 = simsource(M1, dic; reduction=nothing)
            @test sd1.nsrc == 2
            @test all(sd1.geometry.nrec .== 2)
            @test norm(sd1.data[1][:, 2]) == 0
            @test norm(sd1.data[2][:, 1]) == 0
            @test isapprox(sd1.data[1][:, 1], M1[1]*refarray[:, 1])
            @test isapprox(sd1.data[2][:, 2], M1[2]*refarray[:, 2])

            dic = judiVector(example_rec_geometry(nsrc=2, nrec=2), [refarray[:, 1:2], refarray[:, 3:4]])
            @test all(dic.geometry.nrec .== 2)
            sd1 = simsource(M1[:], dic; reduction=nothing)
            @test sd1.nsrc == 2
            @test all(sd1.geometry.nrec .== 2)
            @test isapprox(sd1.data[1], M1[1]*refarray[:, 1:2])
            @test isapprox(sd1.data[2], M1[2]*refarray[:, 3:4])

            # Test minimal supershot (only keep common coordinates)
            geometry = Geometry([[1f0, 2f0], [.5f0, 2f0]], [[0f0], [0f0]], [[0f0, 0.25f0], [0f0, 0.25f0]];
                                 dt=4f0, t=1000f0)
            refarray = randn(Float32, 251, 4)
            dic = judiVector(geometry, [refarray[:, 1:2], refarray[:, 3:4]])
            M1 = randn(Float32, 1, 2)
            dsim = simsource(M1, dic; minimal=true)
            @test dsim.nsrc == 1
            @test dsim.geometry.nrec[1] == 1
            @test dsim.geometry.xloc[1] == [2f0]
            @test dsim.geometry.zloc[1] == [.25f0]

            # Check no common receiver errors
            geometry = Geometry([[1f0, 2f0], [.5f0, 1.5f0]], [[0f0], [0f0]], [[0f0, 0.25f0], [0f0, 0.25f0]];
                        dt=4f0, t=1000f0)
            refarray = randn(Float32, 251, 4)
            dic = judiVector(geometry, [refarray[:, 1:2], refarray[:, 3:4]])
            M1 = randn(Float32, 1, 2)
            @test_throws ArgumentError dsim = simsource(M1, dic; minimal=true)

        end
    end
end
