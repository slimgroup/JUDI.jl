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
        order = .25f0
        Dt = judiTimeDerivative(recGeometry, order)
        It = judiTimeIntegration(recGeometry, order)
        Mm = judiTopmute(model.n, 20, 1)

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
        @test inv(I)*I*model0.m ≈ model0.m.data[:] rtol=ftol atol=0

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
end