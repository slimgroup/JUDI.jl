# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: November 2022

model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
wb = find_water_bottom(model.m .- maximum(model.m))
q, srcGeometry, recGeometry = setup_geom(model; nsrc=2)

ftol = 1f-5

@testset "Preconditioners test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
    @timeit TIMEROUTPUT "Preconditioners tests" begin
        # Propagator if needed
        FM = judiModeling(model, srcGeometry, recGeometry)
        J = judiJacobian(FM, q)
        dobs = FM*q
        dobs_out = 0 .* dobs
        dm = model0.m - model.m

        # JUDI precon make sure it runs
        F = judiFilter(recGeometry, .002, .030)
        Md = judiDataMute(srcGeometry, recGeometry)
        D = judiDepthScaling(model)
        M = judiTopmute(model.n, 20, 1)

        mul!(dobs_out, F, dobs)
        @test isapprox(dobs_out, F*dobs; rtol=ftol)
        mul!(dobs_out, F', dobs)
        @test isapprox(dobs_out, F'*dobs; rtol=ftol)
        mul!(dobs_out, Md, dobs)
        @test isapprox(dobs_out, Md*dobs; rtol=ftol)
        mul!(dobs_out, Md', dobs)
        @test isapprox(dobs_out, Md'*dobs; rtol=ftol)

        # Check JUDI-JOLI compat
        # check JOLI operator w/ judiVector
        mul!(dobs_out, J*D, dm)
        mul!(dobs, J*D, vec(dm.data))
        dlin = J*D*dm
        @test isapprox(dobs_out, dlin; rtol=ftol)
        @test isapprox(dobs_out, dobs; rtol=ftol)
        # check JOLI operator w/ judiVector
        DFT = joDFT(dm.n...; DDT=Float32, RDT=ComplexF32)
        dm1 = adjoint(J*DFT') * dlin
        dm2 = similar(dm1)
        mul!(dm2, adjoint(J*DFT'), dlin)
        @test isapprox(dm1, dm2; rtol=ftol)

        dm1 = M * J' * Md * dobs
        @test length(dm1) == prod(model0.n)
        @test dm1[1] == 0

        w = judiWeights(randn(Float32, model0.n))
        w_out = judiWeights(zeros(Float32, model0.n))

        mul!(w_out, D, w)
        @test isapprox(w_out, D*w; rtol=ftol)
        @test isapprox(w_out.weights[1][end, end]/w.weights[1][end, end], sqrt((model.n[2]-1)*model.d[2]); rtol=ftol)
        mul!(w_out, D', w)
        @test isapprox(w_out, D'*w; rtol=ftol)
        @test isapprox(w_out.weights[1][end, end]/w.weights[1][end, end], sqrt((model.n[2]-1)*model.d[2]); rtol=ftol)

        mul!(w_out, M, w)
        @test isapprox(w_out, M*w; rtol=ftol)
        @test isapprox(norm(w_out.weights[1][:, 1:19]), 0f0; rtol=ftol)
        mul!(w_out, M', w)
        @test isapprox(w_out, M'*w; rtol=ftol)

        # test the output of depth scaling and topmute operators, and test if they are out-of-place
        
        dobs1 = deepcopy(dobs)
        for Op in [F, F', Md , Md']
            m = Op*dobs 
            # Test that dobs wasn't modified
            @test isapprox(dobs,dobs1,rtol=eps())
            # Test that it did compute something 
            @test m != dobs
        end

        w1 = deepcopy(w)

        for Op in [D, D']
            m = Op*w
            # Test that w wasn't modified
            @test isapprox(w,w1,rtol=eps())

            w_expect = deepcopy(w)
            for j = 1:model.n[2]
                w_expect.weights[1][:,j] = w.weights[1][:,j] * Float32(sqrt(model.d[2]*(j-1)))
            end
           @test isapprox(w_expect,m)
        end

        for Op in [M , M']
            m = Op*w
            # Test that w wasn't modified
            @test isapprox(w,w1,rtol=eps())

            @test all(isapprox.(m.weights[1][:,1:18], 0))
            @test isapprox(m.weights[1][:,21:end],w.weights[1][:,21:end])
        end

        
        n = (100,100,100)
        d = (10f0,10f0,10f0)
        o = (0.,0.,0.)
        m = 0.25*ones(Float32,n)
        model3D = Model(n,d,o,m)

        D3 = judiDepthScaling(model3D)

        dm = randn(Float32,prod(n))
        dm1 = deepcopy(dm)
        for Op in [D3, D3']
            opt_out = Op*dm
            # Test that dm wasn't modified
            @test dm1 == dm

            dm_expect = zeros(Float32,model3D.n)
            for j = 1:model3D.n[3]
                dm_expect[:,:,j] = reshape(dm,model3D.n)[:,:,j] * Float32(sqrt(model3D.d[3]*(j-1)))
            end
            @test isapprox(vec(dm_expect),opt_out)
        end

        M3 = judiTopmute(model3D.n, 20, 1)

        for Op in [M3, M3']
            opt_out = Op*dm
            # Test that dm wasn't modified
            @test dm1 == dm

            @test all(isapprox.(reshape(opt_out,model3D.n)[:,:,1:18], 0))
            @test isapprox(reshape(opt_out,model3D.n)[:,:,21:end],reshape(dm,model3D.n)[:,:,21:end])
        end

        # test find_water_bottom in 3D

        dm3D = zeros(Float32,10,10,10)
        dm3D[:,:,4:end] .= 1f0
        @test find_water_bottom(dm3D) == 4*ones(Integer,10,10)


        # Illumination
        I = judiIllumination(FM; mode="v")
        dobs = FM*q
        # forward only, nothing done
        @test I.illums["v"].data == ones(Float32, model.n)

        I = judiIllumination(FM; mode="u", recompute=false)
        dobs = FM[1]*q[1]
        @test norm(I.illums["u"]) != ones(Float32, model.n)
        bck = copy(I.illums["u"].data)
        dobs = FM[2]*q[2]
        # No recompute should not have changed
        @test I.illums["u"].data == bck

        # Test Product
        @test inv(I)*I*model0.m ≈ model0.m.data[:] rtol=ftol atol=0
    end
end