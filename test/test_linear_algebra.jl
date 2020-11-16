# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: July 2020

using JUDI.TimeModeling, Test, LinearAlgebra, Printf

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]

@testset "Arithmetic test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin

        ### Model
        model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"])
        wb = find_water_bottom(model.m .- maximum(model.m))
        q, srcGeometry, recGeometry, info = setup_geom(model)
        dt = srcGeometry.dt[1]
        nt = length(q.data[1])
        nrec = length(recGeometry.xloc[1])

        opt = Options(free_surface=parsed_args["fs"])
        opta = Options(free_surface=parsed_args["fs"], return_array=true)
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

        # zerox
        @test isapprox(zerox(J, dobs), 0f0.*dm)
        @test isapprox(zerox(Jw, dobs), 0f0.*dm)

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

        # test in-place
        
        dobs1 = deepcopy(dobs)
        for Op in [F, F', Md , Md']
                m = Op*dobs 
                # Test that dobs wasn't modified
                @test dobs == dobs1
                # Test that it did compute something 
                @test m ~= dobs
        end

        w1 = deepcopy(w)
        for Op in [D, D', M , M']
                m = Op*w
                # Test that w wasn't modified
                @test w == w1
                # Test that it did compute something 
                @test m ~= w
        end
end
