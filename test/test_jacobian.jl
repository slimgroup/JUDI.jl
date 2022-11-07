# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Update Date: July 2020

### Model
model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
q, srcGeometry, recGeometry, f0 = setup_geom(model; nsrc=2)
dt = srcGeometry.dt[1]

m0 = model0.m
######################## WITH DENSITY ############################################

@testset "Jacobian test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) freesurface $(fs)" begin
    @timeit TIMEROUTPUT "Jacobian generic tests" begin
        # Write shots as segy files to disk
        opt = Options(sum_padding=true, dt_comp=dt, free_surface=fs, f0=f0)

        # Setup operators
        F = judiModeling(model, srcGeometry, recGeometry; options=opt)
        F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
        J = judiJacobian(F0, q)

        # Linear modeling
        dobs = F*q
        dD = J*dm
        dlin0 = J*(0f0.*dm)
    
        @test norm(dlin0) == 0
        @test isapprox(dD, J*vec(dm.data); rtol=1f-6)

        # Gradient test
        grad_test(x-> F(;m=x)*q, m0, dm, dD; data=true)

        # Check return_array returns correct size with zero dm and multiple shots
        J.options.return_array = true
        dlin0v = J*(0f0.*dm)
        @test length(dlin0) == length(vec(dlin0))
    end
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
        Pw = judiLRWF(q.geometry.dt, q.data)

        # Combined operators
        A = Pr*F*adjoint(Pw)
        A0 = Pr*F0*adjoint(Pw)

        # Extended source weights
        w = judiWeights(randn(Float32, model0.n); nsrc=2)
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
