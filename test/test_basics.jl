# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#

ftol = 1f-6

function test_model(ndim; tti=false, elas=false, visco=false)
    n = Tuple(121 for i=1:ndim)
    # non zero origin to check 'limit_model_to_receiver_area'
    o = (10, 0, 0)[1:ndim]
    d = Tuple(10. for i=1:ndim)
    m = .5f0 .+ rand(Float32, n...)
    if tti
        epsilon = .1f0 * m
        delta = .05f0 * m
        theta = .23f0 * m
        phi = .12f0 * m
        model = Model(n, d, o, m; nb=10, epsilon=epsilon, delta=delta, theta=theta, phi=phi)
        @test all(k in JUDI._mparams(model) for k in [:m, :epsilon, :delta, :theta, :phi])
        @test isa(model, JUDI.TTIModel)

        delta2 = 1.1f0 * model.epsilon
        @test sum(delta2 .>= epsilon) == length(delta2)
        JUDI._clip_delta!(delta2, model.epsilon)
        @test sum(delta2 .>= epsilon) == 0
    elseif elas
        vs = .1f0 * m
        model = Model(n, d, o, m; nb=10, vs=vs,)
        @test all(k in JUDI._mparams(model) for k in [:lam, :mu, :b])
        @test isa(model, JUDI.IsoElModel)
    elseif visco
        qp = .1f0 * m
        model = Model(n, d, o, m; nb=10, qp=qp)
        @test all(k in JUDI._mparams(model) for k in [:m, :qp, :rho])
        @test isa(model, JUDI.ViscIsoModel)
    else
        model = Model(n, d, o, m; nb=10)
        @test [JUDI._mparams(model)...] == [:m, :rho]
        @test isa(model, JUDI.IsoModel)
    end
    return model
end

function test_density(ndim)
    n = Tuple(121 for i=1:ndim)
    # non zero origin to check 'limit_model_to_receiver_area'
    o = (10, 0, 0)[1:ndim]
    d = Tuple(10. for i=1:ndim)
    m = .5f0 .+ rand(Float32, n...)
    rho = rand(Float32, n) .+ 1f0
    model = Model(n, d, o, m, rho; nb=0)
    @test :rho in JUDI._mparams(model)
    modelpy = devito_model(model, Options())
    @test isapprox(modelpy.irho.data, 1 ./ model.rho)

    rho[61] = 1000
    model = Model(n, d, o, m, rho; nb=0)
    @test :rho in JUDI._mparams(model)
    modelpy = devito_model(model, Options())
    @test isapprox(modelpy.rho.data, model.rho)
end


# Test padding and its adjoint
function test_padding(ndim)

    model = test_model(ndim; tti=false)
    modelPy = devito_model(model, Options())

    m0 = model.m
    mcut = remove_padding(deepcopy(modelPy.m.data), modelPy.padsizes; true_adjoint=true)
    mdata = deepcopy(modelPy.m.data)

    @show dot(m0, mcut)/dot(mdata, mdata)
    @test isapprox(dot(m0, mcut), dot(mdata, mdata))
end

function test_limit_m(ndim, tti)
    model = test_model(ndim; tti=tti)

    @test get_dt(model) == calculate_dt(model)
    mloc = model.m
    model.m .*= 0f0
    @test norm(model.m) == 0
    model.m .= mloc
    @test model.m == mloc

    srcGeometry = example_src_geometry()
    recGeometry = example_rec_geometry(cut=true)
    buffer = 100f0
    dm = rand(Float32, model.n...)
    dmb = PhysicalParameter(0 .* dm, model.n, model.d, model.o)
    new_mod, dm_n = limit_model_to_receiver_area(srcGeometry, recGeometry, deepcopy(model), buffer; pert=dm)

    # check new_mod coordinates
    # as long as func 'example_rec_geometry' uses '0' for 'y' we check only 'x' limits
    min_x = min(minimum(recGeometry.xloc[1]), minimum(srcGeometry.xloc[1]))
    max_x = max(maximum(recGeometry.xloc[1]), maximum(srcGeometry.xloc[1]))

    @test isapprox(new_mod.o[1], min_x-buffer; rtol=ftol)
    @test isapprox(new_mod.o[1] + new_mod.d[1]*(new_mod.n[1]-1), max_x+buffer; rtol=ftol)

    # check inds
    # 5:115 because of origin[1] = 10 (if origin[1] = 0 then 6:116)
    inds = ndim == 3 ? [5:115, 1:11, 1:121] : [5:115, 1:121]
    @test new_mod.n[1] == 111
    @test new_mod.n[end] == 121
    if ndim == 3
        @test new_mod.n[2] == 11
    end

    @test isapprox(new_mod.m, model.m[inds...]; rtol=ftol)
    @test isapprox(dm_n, vec(dm[inds...]); rtol=ftol)
    if tti
        @test isapprox(new_mod.epsilon, model.epsilon[inds...]; rtol=ftol)
        @test isapprox(new_mod.delta, model.delta[inds...]; rtol=ftol)
        @test isapprox(new_mod.theta, model.theta[inds...]; rtol=ftol)
        @test isapprox(new_mod.phi, model.phi[inds...]; rtol=ftol)
    end

    dmt = PhysicalParameter(dm_n, new_mod.n, model.d, new_mod.o)
    ex_dm = dmb + dmt
    @test ex_dm.o == model.o
    @test ex_dm.n == model.n
    @test ex_dm.d == model.d
    @test norm(ex_dm) == norm(dm_n)
end

function setup_3d()
    xrec = Array{Any}(undef, 2)
    yrec = Array{Any}(undef, 2)
    zrec = Array{Any}(undef, 2)
    for i=1:2
        xrec[i] = i .+ collect(0:10)
        yrec[i] = i .+ collect(0:10)
        zrec[i] = i
    end
    x3d, y3d, z3d = setup_3D_grid(xrec[1],yrec[1],zrec[1])
    for k=1:11
        @test all(y3d[(11*(k-1))+1:(11*(k-1))+11] .== k)
        @test all(x3d[k:11:end] .== k)
    end
    @test all(z3d[:] .== 1)
    x3d, y3d, z3d = setup_3D_grid(xrec,yrec,zrec)
    for i=1:2
        for k=1:11
            @test all(y3d[i][(11*(k-1))+1:(11*(k-1))+11] .== k + i -1)
            @test all(x3d[i][k:11:end] .== k + i - 1)
        end
        @test all(z3d[i][:] .== i)
    end
end

function test_ftp()
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI/overthrust_model_2D.h5")
    @test isfile("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5")
    rm("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5")
    ftp_data("ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/2DFWI", "overthrust_model_2D.h5")
    @test isfile("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5")
    rm("$(JUDI.JUDI_DATA)/overthrust_model_2D.h5")
end

@testset "Test basic utilities" begin
    @timeit TIMEROUTPUT "Basic setup utilities" begin
        test_ftp()
        setup_3d()
        for ndim=[2, 3]
            test_padding(ndim)
            test_density(ndim)
        end
        opt = Options(frequencies=[[2.5, 4.5], [3.5, 5.5]])
        @test subsample(opt, 1).frequencies == [2.5, 4.5]
        @test subsample(opt, 2).frequencies == [3.5, 5.5]

        for ndim=[2, 3]
            for tti=[true, false]
                test_limit_m(ndim, tti)
            end
        end

        # Test model
        for ndim=[2, 3]
            for (tti, elas, visco) in [(false, false, false), (true, false, false), (false, true, false), (false, false, true)]
                model = test_model(ndim; tti=tti, elas=elas, visco=visco)

                # Default dt
                modelPy = devito_model(model, Options())
                @test isapprox(modelPy.critical_dt, calculate_dt(model))
                @test get_dt(model) == calculate_dt(model)
                @test isapprox(calculate_dt(model; dt=.5f0), .5f0)

                # Input dt
                modelPy = devito_model(model, Options(dt_comp=.5f0))
                @test modelPy.critical_dt == .5f0

                # Verify nt
                srcGeometry = example_src_geometry()
                recGeometry = example_rec_geometry(cut=true)
                nt1 = get_computational_nt(srcGeometry, recGeometry, model)
                nt2 = get_computational_nt(srcGeometry, model)
                nt3 = get_computational_nt(srcGeometry, recGeometry, model; dt=1f0)
                nt4 = get_computational_nt(srcGeometry, model; dt=1f0)
                dtComp = calculate_dt(model)
                @test all(nt1 .== length(0:dtComp:(dtComp*ceil(1000f0/dtComp))))
                @test all(nt1 .== nt2)
                @test all(nt3 .== 1001)
                @test all(nt4 .== 1001)
            end
        end
    end
end