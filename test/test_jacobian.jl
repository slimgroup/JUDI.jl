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
    F = judiModeling(model, srcGeometry, recGeometry; options=opt)
    F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
    J = judiJacobian(F0, q)

    # Linear modeling
    dobs = F*q
    dD = J*dm

    @test norm(J*(0f0.*dm)) == 0
    @test isapprox(dD, J*vec(dm.data); rtol=1f-6)

<<<<<<< HEAD
    # Test sim src
    W = judiSimSrcWeights(randn(Float32, 4, dD.nsrc))
    @test isapprox(W * dD, W * J * dm; rtol=1f-4)

    # Jacobian test
    maxiter = 6
    h = 5f-2
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)
=======
    # Gradient test
    grad_test(x-> F(;m=x)*q, m0, dm, dD; data=true)
>>>>>>> master

end

### Extended source
@testset "Extended source Jacobian test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" for ra=[true, false]
    @timeit TIMEROUTPUT "Extended source Jacobian return_array=$(ra)" begin
        opt = Options(sum_padding=true, dt_comp=dt, return_array=ra, free_surface=fs, f0=f0)
        DT = ra ? Vector{Float32} : judiVector{Float32, Matrix{Float32}}
        QT = ra ? Vector{Float32} : judiWeights{Float32}
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
        @test typeof(dobs) == DT
        @test typeof(wa) == QT
        dD = J*dm
        @test typeof(dD) == DT

        # Gradient test
        grad_test(x-> A(;m=x)*w, m0, dm, dD; data=true)
    end
end
