# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: July 2020

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]
isic =  parsed_args["isic"]

@testset "Arithmetic test with $(nlayer) layers and tti $(tti) and freesurface $(fs) and isic $(isic)" begin

        # Test 2D find_water_bottom

        dm2D = zeros(Float32,10,10)
        dm2D[:,6:end] .= 1f0
        @test find_water_bottom(dm2D) == 6*ones(Integer,10)

        ### Model
        model, model0, dm = setup_model(tti, nlayer)
        wb = find_water_bottom(model.m .- maximum(model.m))
        q, srcGeometry, recGeometry, info = setup_geom(model)
        dt = srcGeometry.dt[1]
        nt = length(q.data[1])
        nrec = length(recGeometry.xloc[1])

        opt = Options(free_surface=fs, isic=isic)
        opta = Options(free_surface=fs, isic=isic, return_array=true)
        ftol = 5f-5

        w = judiWeights(randn(Float32, model0.n))

        # Build operators
        Pr = judiProjection(info, recGeometry)
        F = judiModeling(info, model; options=opt)
        Fa = judiModeling(info, model; options=opta)
        Ps = judiProjection(info, srcGeometry)
        Pw = judiLRWF(info, q.data[1])
        J = judiJacobian(Pr*F*adjoint(Ps), q)
        Jw = judiJacobian(Pr*F*adjoint(Pw), w)

        dobs = Pr * F * Ps' * q
        dobsa = Pr * Fa * Ps' * q
        dobs_w = Pr * F * Pw' * w
        dobs_wa = Pr * Fa * Pw' * w
        dobs_out = 0f0 .* dobs
        dobs_outa = 0f0 .* dobsa
        dobs_w_out = 0f0 .* dobs_w
        dobs_w_outa = 0f0 .* dobs_wa
        q_out = 0f0 .* q
        w_out = 0f0 .* w

        # mul!
        mul!(dobs_out, Pr * F * Ps', q)
        mul!(dobs_w_out, Pr * F * Pw', w)
        mul!(dobs_outa, Pr * Fa * Ps', q)
        mul!(dobs_w_outa, Pr * Fa * Pw', w)
        @test isapprox(dobs, dobs_out; rtol=ftol)
        @test isapprox(dobsa, dobs_outa; rtol=ftol)
        @test isapprox(dobs_w, dobs_w_out; rtol=ftol)
        @test isapprox(dobs_wa, dobs_w_outa; rtol=ftol)

        mul!(w_out, adjoint(Pr * F * Pw'), dobs_w_out)
        w_a = adjoint(Pr * Fa * Pw') * dobs_w_out
        w_outa = 0f0 .* w_a
        mul!(w_outa, adjoint(Pr * Fa * Pw'), dobs_w_out)
        @test isapprox(w_out, adjoint(Pr * F * Pw') * dobs_w_out; rtol=ftol)
        @test isapprox(w_outa, w_a; rtol=ftol)

        # jacobian
        dm2 = copy(dm)
        dmd = copy(dm2.data)
        mul!(dm2, J', dobs)
        mul!(dmd, J', dobs)
        @test isapprox(dm2, J'*dobs)
        @test isapprox(dmd, dm2)

        mul!(dobs_out, J, dm)
        mul!(dobs, J, dm.data)
        dlin = J*dm
        @test isapprox(dobs_out, dlin; rtol=ftol)
        @test isapprox(dobs_out, dobs; rtol=ftol)
        # JUDI precon make sure it runs
        F = judiFilter(recGeometry, .002, .030)
        Md = judiMarineTopmute2D(0, recGeometry)
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

end
