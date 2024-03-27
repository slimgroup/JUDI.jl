# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: November 2022

model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
wb = find_water_bottom(model.m .- maximum(model.m))
q, srcGeometry, recGeometry = setup_geom(model; nsrc=2)

ftol = 1f-5

# Propagator if needed
FM = judiModeling(model, srcGeometry, recGeometry)
J = judiJacobian(FM, q)
dobs = FM * q
dobs_out = 0 .* dobs
dm = model0.m - model.m


@testset "Preconditioners test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
    @timeit TIMEROUTPUT "Data Preconditioners tests" begin
        F = judiFilter(recGeometry, .002, .030)
        Mdr = judiDataMute(srcGeometry, recGeometry; mode=:reflection)
        Mdt = judiDataMute(srcGeometry, recGeometry; mode=:turning)
        Mdg = judiTimeGain(recGeometry, 2f0)
        Mm = judiTopmute(model.n, 10, 1)
        order = .25f0
        Dt = judiTimeDerivative(recGeometry, order)
        It = judiTimeIntegration(recGeometry, order)

        # Time differential only
        @test inv(It) == Dt
        @test inv(Dt) == It
        dinv = inv(It) * It * dobs
        @test isapprox(dinv, dobs; atol=0f0, rtol=ftol)
        dinv = inv(Dt) * Dt * dobs
        @test isapprox(dinv, dobs; atol=0f0, rtol=ftol)
        dinv = Dt * It * dobs
        @test isapprox(dinv, dobs; atol=0f0, rtol=ftol)

        # Time gain inverse is just 1/pow
        @test inv(Mdg) == judiTimeGain(recGeometry, -2f0)

        # conj/transpose
        for Pc in [F, Mdr, Mdt, Mdg, Dt, It]
            @test conj(Pc) == Pc
            @test transpose(Pc) == Pc
        end

        # DataPrecon getindex
        for Pc in [F, Mdr, Mdt, Mdg, Dt, It]
            @test Pc[1] * dobs[1] == (Pc * dobs)[1]
            @test Pc[1] * dobs.data[1][:] ≈ (Pc * dobs).data[1][:] rtol=ftol
        end

        # Time resample
        newt = 0f0:5f0:recGeometry.t[1]
        for Pc in [F, Mdr, Mdt, Mdg, Dt, It]
            @test_throws AssertionError time_resample(Pc, newt)
            @test time_resample(Pc[1], newt).recGeom.taxis[1] == newt
        end
        multiP = time_resample((F[1], Mdr[1]), newt)
        @test isa(multiP, JUDI.MultiPreconditioner)
        @test isa(multiP.precs[1], JUDI.FrequencyFilter)
        @test isa(multiP.precs[2], JUDI.DataMute)
        @test all(Pi.recGeom.taxis[1] == newt for Pi in multiP.precs)
    
        multiP = time_resample([F[1], Mdr[1]], newt)
        @test isa(multiP, JUDI.MultiPreconditioner)
        @test isa(multiP.precs[1], JUDI.FrequencyFilter)
        @test isa(multiP.precs[2], JUDI.DataMute)
        @test all(Pi.recGeom.taxis[1] == newt for Pi in multiP.precs)

        # Test in place DataPrecon
        for Pc in [F, Mdr, Mdt, Mdg, Dt, It]
            mul!(dobs_out, Pc, dobs)
            @test isapprox(dobs_out, Pc*dobs; rtol=ftol)
            mul!(dobs_out, Pc', dobs)
            @test isapprox(dobs_out, Pc'*dobs; rtol=ftol)

            # Check JUDI-JOLI compat
            mul!(dobs_out, Pc*J, dm)
            mul!(dobs, Pc*J, vec(dm.data))
            dlin = Pc*J*dm
            @test isapprox(dobs_out, dlin; rtol=ftol)
            @test isapprox(dobs_out, dobs; rtol=ftol)
        end

        # check JOLI operator w/ judiVector
        DFT = joDFT(dm.n...; DDT=Float32, RDT=ComplexF32)
        dm1 = adjoint(J*DFT') * dobs
        dm2 = similar(dm1)
        mul!(dm2, adjoint(J*DFT'), dobs)
        @test isapprox(dm1, dm2; rtol=ftol)

        dm1 = Mm * J' * Mdr * dobs
        @test length(dm1) == prod(model0.n)
        @test dm1[1] == 0

        # test out-of-place
        dobs1 = deepcopy(dobs)
        for Op in [F, F', Mdr , Mdr', Mdt, Mdt', Mdg, Mdg', Dt, Dt', It, It']
            m = Op*dobs 
            # Test that dobs wasn't modified
            @test isapprox(dobs, dobs1, rtol=eps())
            # Test that it did compute something 
            @test m != dobs
        end
    end


    @timeit TIMEROUTPUT "Model Preconditioners tests" begin

        # JUDI precon make sure it runs
        Ds = judiDepthScaling(model)
        Mm = judiTopmute(model.n, 20, 1)
        Mm2 = judiTopmute(model.n, 20 * ones(model.n[1]), 1)

        # conj/transpose
        for Pc in [Ds, Mm, Mm2]
            @test conj(Pc) == Pc
            @test transpose(Pc) == Pc
        end

        w = judiWeights(randn(Float32, model0.n))
        w_out = judiWeights(zeros(Float32, model0.n))

        mul!(w_out, Ds, w)
        @test isapprox(w_out, Ds*w; rtol=ftol)
        @test isapprox(w_out.weights[1][end, end]/w.weights[1][end, end], sqrt((model.n[2]-1)*model.d[2]); rtol=ftol)
        mul!(w_out, Ds', w)
        @test isapprox(w_out, Ds'*w; rtol=ftol)
        @test isapprox(w_out.weights[1][end, end]/w.weights[1][end, end], sqrt((model.n[2]-1)*model.d[2]); rtol=ftol)

        mul!(w_out, Mm, w)
        @test isapprox(w_out, Mm*w; rtol=ftol)
        @test isapprox(norm(w_out.weights[1][:, 1:19]), 0f0; rtol=ftol)
        mul!(w_out, Mm', w)
        @test isapprox(w_out, Mm'*w; rtol=ftol)
    
        w1 = deepcopy(w)

        for Op in [Ds, Ds']
            m = Op*w
            # Test that w wasn't modified
            @test isapprox(w, w1,rtol=eps())

            w_expect = deepcopy(w)
            for j = 1:model.n[2]
                w_expect.weights[1][:,j] = w.weights[1][:,j] * Float32(sqrt(model.d[2]*(j-1)))
            end
           @test isapprox(w_expect,m)
        end

        for Op in [Mm , Mm', Mm2, Mm2']
            m = Op*w
            # Test that w wasn't modified
            @test isapprox(w,w1,rtol=eps())

            @test all(isapprox.(m.weights[1][:,1:18], 0))
            @test isapprox(m.weights[1][:,21:end],w.weights[1][:,21:end])
        end

        # Depth scaling
        n = (100,100,100)
        d = (10f0,10f0,10f0)
        o = (0.,0.,0.)
        m = 0.25*ones(Float32,n)
        model3D = Model(n,d,o,m)

        D3 = judiDepthScaling(model3D)
        @test isa(inv(D3), DepthScaling{Float32, 3, -.5f0})

        dmr = randn(Float32, prod(n))
        dm1 = deepcopy(dmr)
        for Op in [D3, D3']
            opt_out = Op*dmr
            # Test that dm wasn't modified
            @test dm1 == dmr

            dm_expect = zeros(Float32, model3D.n)
            for j = 1:model3D.n[3]
                dm_expect[:,:,j] = reshape(dmr, model3D.n)[:,:,j] * Float32(sqrt(model3D.d[3]*(j-1)))
            end
            @test isapprox(vec(dm_expect), opt_out)
            Op*model3D.m
        end

        M3 = judiTopmute(model3D.n, 20, 1)

        for Op in [M3, M3']
            opt_out = Op*dmr
            # Test that dm wasn't modified
            @test dm1 == dmr

            @test all(isapprox.(reshape(opt_out,model3D.n)[:,:,1:18], 0))
            @test isapprox(reshape(opt_out,model3D.n)[:,:,21:end],reshape(dmr,model3D.n)[:,:,21:end])
            Op*model3D.m
        end

        # test find_water_bottom in 3D
        dm3D = ones(Float32,10,10,10)
        dm3D[:,:,4:end] .= 2f0
        @test find_water_bottom(dm3D) == 4*ones(Integer,10,10)

        # Illumination
        I = judiIllumination(FM; mode="v")
        dloc = FM*q
        # forward only, nothing done
        @test I.illums["v"].data == ones(Float32, model.n)

        I = judiIllumination(FM; mode="u", recompute=false)
        dloc = FM[1]*q[1]
        @test norm(I.illums["u"]) != norm(ones(Float32, model.n))
        bck = copy(I.illums["u"].data)
        dloc = FM[2]*q[2]
        # No recompute should not have changed
        @test I.illums["u"].data == bck
        # New mode
        Iv = I("v")
        @test "v" ∈ keys(Iv.illums)
        @test "u" ∉ keys(Iv.illums)
        @test norm(Iv.illums["v"]) == norm(ones(Float32, model.n))
        # Test Product
        @test inv(I)*I*model0.m ≈ model0.m rtol=ftol atol=0

        # Test in place ModelPrecon
        for Pc in [Ds, Mm, Mm2, I]
            mul!(dm, Pc, model.m)
            @test isapprox(dm, Pc*model.m; rtol=ftol)
            mul!(dm, Pc', model.m)
            @test isapprox(dm, Pc'*model.m; rtol=ftol)

            # Check JUDI-JOLI compat
            mul!(dm, Pc*J', dobs)
            dml = Pc*J'*dobs
            @test isapprox(dm, dml; rtol=ftol)
            mul!(dm, Pc*J', vec(dobs))
            @test isapprox(dm, dml; rtol=ftol)
        end
    end

    @timeit TIMEROUTPUT "OOC Data Preconditioners tests" begin
        datapath = joinpath(dirname(pathof(JUDI)))*"/../data/"
        # OOC judiVector
        container = segy_scan(datapath, "unit_test_shot_records_2",
                              ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
        d_cont = judiVector(container; segy_depth_key="RecGroupElevation")
        src_geometry = Geometry(container; key = "source", segy_depth_key = "SourceDepth")
        wavelet = ricker_wavelet(src_geometry.t[1], src_geometry.dt[1], 0.005)
        q_cont = judiVector(src_geometry, wavelet)

        # Make sure we test OOC
        @test typeof(d_cont) == judiVector{Float32, SeisCon}
        @test isequal(d_cont.nsrc, 2)
        @test isequal(typeof(d_cont.data), Array{SegyIO.SeisCon, 1})
        @test isequal(typeof(d_cont.geometry), GeometryOOC{Float32})
        
        # Make OOC preconditioner
        Mdt = judiDataMute(q_cont, d_cont)
        Mdg = judiTimeGain(d_cont, 2f0)

        # Test OOC DataPrecon
        for Pc in [Mdt, Mdg]
            # mul
            m = Pc * d_cont
            @test isa(m, JUDI.LazyMul{Float32})
            @test m.nsrc == d_cont.nsrc
            @test m.P == Pc
            @test m.msv == d_cont

            ma = Pc' * d_cont
            @test isa(ma, JUDI.LazyMul{Float32})
            @test isa(ma[1], JUDI.LazyMul{Float32})
            @test ma.nsrc == d_cont.nsrc
            @test ma.P == Pc'
            @test ma.msv == d_cont

            # getindex
            m1 = m[1]
            @test isa(m1, JUDI.LazyMul{Float32})
            @test m1.nsrc == 1
            @test m1.msv == d_cont[1]
            @test get_data(m1) == get_data(Pc[1] * d_cont[1])

            # data
            @test isa(m.data, JUDI.LazyData{Float32})
            @test_throws MethodError m.data[1] = 1

            @test m.data[1] ≈ (Pc[1] * get_data(d_cont[1])).data
            @test get_data(m.data) ≈ Pc * get_data(d_cont)

            # Propagation
            Fooc = judiModeling(model, src_geometry, d_cont.geometry)

            d_syn = Fooc' * Pc' * d_cont
            d_synic = Fooc' * Pc' * get_data(d_cont)

            @test isapprox(d_syn, d_synic; rtol=1f-5)

            f, g = fwi_objective(model0, Pc*d_cont, q_cont)
            f2, g2 = fwi_objective(model0, get_data(Pc*d_cont), get_data(q_cont))
            @test isapprox(f, f2)
            @test isapprox(g, g2)

        end
    end
end