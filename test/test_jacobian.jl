# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Update Date: July 2020

### Model
model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
q, srcGeometry, recGeometry, f0 = setup_geom(model)
dt = srcGeometry.dt[1]

m0 = model0.m
######################## WITH DENSITY ############################################

@testset "Jacobian test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) freesurface $(fs)" begin
    # Write shots as segy files to disk
    opt = Options(sum_padding=true, dt_comp=dt, free_surface=fs, f0=f0)

    # Setup operators
    Pr = judiProjection(recGeometry)
    F = judiModeling(model; options=opt)
    F0 = judiModeling(model0; options=opt)
    Ps = judiProjection(srcGeometry)
    J = judiJacobian(Pr*F0*adjoint(Ps), q)

    # Linear modeling
    dobs = Pr*F0*Ps'*q
    dD = J*dm

    @test norm(J*(0f0.*dm)) == 0
    @test isapprox(dD, J*vec(dm.data); rtol=1f-6)

    # Jacobian test
    maxiter = 6
    h = 5f-2
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)

    for j=1:maxiter

        F.model.m = m0 + h*dm
        dpred = Pr*F*Ps'*q

        err1[j] = norm(dpred - dobs)
        err2[j] = norm(dpred - dobs - h*dD)
        j == 1 ? prev = 1 : prev = j - 1
        @printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h, err1[j], err1[prev]/err1[j])
        @printf(", e2 = %2.2e, rate = %2.2e \n", err2[j], err2[prev]/err2[j])

        h = h * .8f0
    end
end

### Extended source

@testset "Extended source Jacobian test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
    @timeit TIMEROUTPUT "Extended source Jacobian" begin
        opt = Options(sum_padding=true, dt_comp=dt, return_array=true, free_surface=fs, f0=f0)

        # Setup operators
        Pr = judiProjection(recGeometry)
        F = judiModeling(model; options=opt)
        F0 = judiModeling(model0; options=opt)
        Pw = judiLRWF(q.geometry.dt[1], q.data[1])

        # Combined operators
        A = Pr*F*adjoint(Pw)
        A0 = Pr*F0*adjoint(Pw)

        # Extended source weights
        w = judiWeights(randn(Float32, model0.n))
        J = judiJacobian(A0, w)

        # Nonlinear modeling
        dobs = A0*w
        wa = adjoint(A0)*dobs
        @test typeof(dobs) == Vector{Float32}
        @test typeof(wa) == Vector{Float32}
        dD = J*dm
        @test typeof(dD) == Vector{Float32}

        # Jacobian test
        maxiter = 6
        h = 5f-2
        err1 = zeros(Float32, maxiter)
        err2 = zeros(Float32, maxiter)

        for j=1:maxiter

            A.model.m = m0 + h*dm
            dpred = A*w
            @test typeof(dpred) == Vector{Float32}

            err1[j] = norm(dpred - dobs)
            err2[j] = norm(dpred - dobs - h*dD)
            j == 1 ? prev = 1 : prev = j - 1
            @printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h, err1[j], err1[prev]/err1[j])
            @printf(", e2 = %2.2e, rate = %2.2e \n", err2[j], err2[prev]/err2[j])

            h = h * .8f0
        end

        rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1)
        rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1)

        @test isapprox(rate_1, 1.25f0; rtol=1f-2)
        @test isapprox(rate_2, 1.5625f0; rtol=1f-2)
    end
end
