# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: July 2020

@testset "Arithmetic test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
    @timeit TIMEROUTPUT "LA Arithmetic tests" begin
        # Test 2D find_water_bottom

        dm2D = zeros(Float32,10,10)
        dm2D[:,6:end] .= 1f0
        @test find_water_bottom(dm2D) == 6*ones(Integer,10)

        ### Model
        model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
        wb = find_water_bottom(model.m .- maximum(model.m))
        q, srcGeometry, recGeometry = setup_geom(model)
        dt = srcGeometry.dt[1]
        nt = length(q.data[1])
        nrec = length(recGeometry.xloc[1])

        opt = Options(free_surface=fs)
        opta = Options(free_surface=fs, return_array=true)
        ftol = 5f-5

        w = judiWeights(randn(Float32, model0.n))

        # Build operators
        Pr = judiProjection(recGeometry)
        F = judiModeling(model; options=opt)
        Fa = judiModeling(model; options=opta)
        Ps = judiProjection(srcGeometry)
        Pw = judiLRWF(q.geometry.dt[1], q.data[1])
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
    end
end
