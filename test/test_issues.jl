using JLD2

@testset "Test issue #83" begin
    @timeit TIMEROUTPUT "Issue 83" begin
        nxrec = 120
        nyrec = 100
        xrec = range(50f0, stop=1150f0, length=nxrec)
        yrec = range(100f0, stop=900f0, length=nyrec)
        zrec = 50f0

        # Construct 3D grid from basis vectors
        (x3d, y3d, z3d) = setup_3D_grid(xrec, yrec, zrec)

        for k=1:nyrec
            s = (k - 1) * nxrec + 1
            e = s + nxrec - 1
            @test all(x3d[k:nxrec:end] .== xrec[k])
            @test all(y3d[s:e] .== yrec[k])
        end
        @test all(z3d .== zrec)
    end
end

@testset "Test backward comp issue" begin
    @timeit TIMEROUTPUT "Backward compatibility" begin
        datapath = joinpath(dirname(pathof(JUDI)))*"/../data/"

        # Load file with old judiVector type and julia <1.7 StepRangeLen
        @load "$(datapath)backward_comp.jld" dat

        @test typeof(dat) == judiVector{Float32, Matrix{Float32}}
        @test typeof(dat.geometry) == GeometryIC{Float32}
        @test typeof(dat.geometry.xloc) == Vector{Vector{Float32}}
    end
end

@testset "Test issue #130" begin
    @timeit TIMEROUTPUT "Issue 130" begin
        model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
        q, srcGeometry, recGeometry, f0 = setup_geom(model; nsrc=1)
        opt = Options(save_data_to_disk=true, file_path=pwd(),	# path to files
                      file_name="test_wri")	# saves files as file_name_xsrc_ysrc.segy
        F = judiModeling(model, srcGeometry, recGeometry; options=opt)
        dobs = F*q
        f, gm, gy = twri_objective(model0, q, dobs, nothing; options=opt, optionswri=TWRIOptions(params=:all))
        @test typeof(f) <: Number
        @test typeof(gy) <: judiVector
        @test typeof(gm) <: PhysicalParameter
    end
end

@testset "Test issue #140" begin
    @timeit TIMEROUTPUT "Issue 140" begin
        n = (120, 100)   # (x,y,z) or (x,z)
        d = (10., 10.)
        o = (0., 0.)
        
        v = ones(Float32,n) .+ 0.5f0
        v[:,Int(round(end/2)):end] .= 3.5f0
        rho = ones(Float32,n) ;
        rho[1,1] = .09;# error but .1 makes rho go to b and then it is happy
        m = (1f0 ./ v).^2
        nsrc = 2	# number of sources
        model = Model(n, d, o, m;rho=rho)
        
        q, srcGeometry, recGeometry, f0 = setup_geom(model; nsrc=nsrc)
        F = judiModeling(model, srcGeometry, recGeometry)
        dobs = F*q
        @test ~isnan(norm(dobs))
    end
end

@testset "Tests limit_m issue 156" begin
    @timeit TIMEROUTPUT "Issue 156" begin
        model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
        q, srcGeometry, recGeometry, f0 = setup_geom(model; nsrc=1)
        # Restrict rec to middle of the model
        recGeometry.xloc[1] .= range(11*model.d[1], stop=(model.n[1] - 12)*model.d[1],
                                     length=length(recGeometry.xloc[1]))
        # Data
        F = judiModeling(model, srcGeometry, recGeometry)
        dobs = F*q
        # Run gradient and check output size
        opt = Options(limit_m=true, buffer_size=0f0)
        F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
    
        # fwi wrapper
        g_ap = JUDI.multi_src_fg(model0, q , dobs, nothing, opt, false, false, mse)[2]
        @test g_ap.n == (model.n .- (22, 0))
        @test g_ap.o[1] == model.d[1]*11
    
        g1 = fwi_objective(model0, q, dobs; options=opt)[2]
        @test g1.n  == model.n
        @test norm(g1.data[1:10, :]) == 0
        @test norm(g1.data[end-10:end, :]) == 0

        # lsrtm wrapper
        g_ap = JUDI.multi_src_fg(model0, q , dobs, dm, opt, false, true, mse)[2]
        @test g_ap.n == (model.n .- (22, 0))
        @test g_ap.o[1] == model.d[1]*11

        g2 = lsrtm_objective(model0, q, dobs, dm; options=opt)[2]
        @test g2.n  == model.n
        @test norm(g2.data[1:10, :]) == 0
        @test norm(g2.data[end-10:end, :]) == 0

        # Lin op
        Jp = judiJacobian(F0, q)'
        g_ap = JUDI.propagate(Jp, dobs)
        @test g_ap.n == (model.n .- (22, 0))
        @test g_ap.o[1] == model.d[1]*11

        g3 = Jp * dobs
        @test g3.n  == model.n
        @test norm(g3.data[1:10, :]) == 0
        @test norm(g3.data[end-10:end, :]) == 0
    end
end