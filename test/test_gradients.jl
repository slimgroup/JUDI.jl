# 2D LS-RTM gradient test with 1 source
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020
#
# Ziyi Yin, ziyi.yin@gatech.edu
# Updated July 2021

### Model
model, model0, dm = setup_model(tti, viscoacoustic, 4)
q, srcGeometry, recGeometry, f0 = setup_geom(model)
dt = srcGeometry.dt[1]

opt = Options(sum_padding=true, free_surface=fs, f0=f0)
F = judiModeling(model, srcGeometry, recGeometry; options=opt)
F0 = judiModeling(model0, srcGeometry, recGeometry; options=opt)
J = judiJacobian(F0, q)

# Observed data
dobs = F*q
dobs0 = F0*q
dm1 = 2f0*circshift(dm, 10)

ftol = (tti | fs | viscoacoustic) ? 1f-1 : 1f-2

###################################################################################################

@testset "FWI gradient test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
	# FWI gradient and function value for m0
	Jm0, grad = fwi_objective(model0, q, dobs; options=opt)
	# Check get same misfit as l2 misifit on forward data
	Jm01 = .5f0 * norm(F(model0)*q - dobs)^2
	@test Jm0 ≈ Jm01

	grad_test(x-> .5f0*norm(F(;m=x)*q - dobs)^2, model0.m , dm, grad)

end

###################################################################################################
@testset "FWI preconditionners test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
	Ml = judiDataMute(q.geometry, dobs.geometry; t0=.2)
	Ml2 = judiTimeDerivative(dobs.geometry, 1)


	Jm0, grad = fwi_objective(model0, q, dobs; options=opt, data_precon=Ml)
	ghand = J'*Ml*(F0*q - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = fwi_objective(model0, q, dobs; options=opt, data_precon=[Ml, Ml2])
	ghand = J'*Ml*Ml2*(F0*q - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = fwi_objective(model0, q, dobs; options=opt, data_precon=Ml*Ml2)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)
end


@testset "LSRTM preconditionners test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
	Mr = judiTopmute(model0; taperwidth=10)
	Ml = judiDataMute(q.geometry, dobs.geometry)
	Ml2 = judiTimeDerivative(dobs.geometry, 1)
	Mr2 = judiIllumination(J)

	Jm0, grad = lsrtm_objective(model0, q, dobs, dm; options=opt, data_precon=Ml, model_precon=Mr)
	ghand = J'*Ml*(J*Mr*dm - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = lsrtm_objective(model0, q, dobs, dm; options=opt, data_precon=[Ml, Ml2], model_precon=[Mr, Mr2])
	ghand = J'*Ml*Ml2*(J*Mr2*Mr*dm - dobs)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

	Jm0, grad = lsrtm_objective(model0, q, dobs, dm; options=opt, data_precon=Ml*Ml2, model_precon=Mr*Mr2)
	@test isapprox(norm(grad - ghand)/norm(grad+ghand), 0f0; rtol=0, atol=ftol)

end

###################################################################################################

@testset "LSRTM gradient test with $(nlayer) layers, tti $(tti), viscoacoustic $(viscoacoustic). freesurface $(fs), nlind $(nlind)" for nlind=[true, false]
	@timeit TIMEROUTPUT "LSRTM gradient test, nlind=$(nlind)" begin
		# LS-RTM gradient and function value for m0
		dD = nlind ? (dobs - dobs0) : dobs
		Jm0, grad = lsrtm_objective(model0, q, dD, dm; options=opt, nlind=nlind)

		# Gradient test
		grad_test(x-> lsrtm_objective(model0, q, dD, x;options=opt, nlind=nlind)[1], dm, dm1, grad)

		# test that with zero dm we get the same as fwi_objective for residual
		if nlind
			Jls, gradls = @single_threaded lsrtm_objective(model0, q, dobs, 0f0.*dm; options=opt, nlind=true)
			Jfwi, gradfwi = @single_threaded fwi_objective(model0, q, dobs; options=opt)
			@test isapprox(Jls, Jfwi; rtol=0f0, atol=0f0)
			@test isapprox(gradls, gradfwi; rtol=0f0, atol=0f0)
		end
	end
end

# Test if lsrtm_objective produces the same value/gradient as is done by the correct algebra
@testset "LSRTM gradient linear algebra test with $(nlayer) layers, tti $(tti), viscoacoustic $(viscoacoustic), freesurface $(fs)" begin
	# Draw a random case to avoid long CI.
	ic = rand(["isic", "fwi", "as"])
	printstyled("LSRTM validity with dft, IC=$(ic)\n", color=:blue)
    @timeit TIMEROUTPUT "LSRTM validity with dft, IC=$(ic)" begin
		ftol = fs ? 1f-3 : 5f-4
		freq = [[2.5, 4.5],[3.5, 5.5],[10.0, 15.0], [30.0, 32.0]]
		J.options.free_surface = fs
		J.options.IC = ic
		J.options.frequencies = freq

		d_res = dobs0 + J*dm1 - dobs
		Jm0_1 = 0.5f0 * norm(d_res)^2f0
		grad_1 = @single_threaded J'*d_res

		opt = J.options
		Jm0, grad = @single_threaded lsrtm_objective(model0, q, dobs, dm1; options=opt, nlind=true)
		Jm01, grad1 = @single_threaded lsrtm_objective(model0, q, dobs-dobs0, dm1; options=opt, nlind=false)
	
		@test isapprox(grad, grad_1; rtol=ftol)
		@test isapprox(Jm0, Jm0_1; rtol=ftol)
		@test isapprox(grad, grad1; rtol=ftol)
		@test isapprox(Jm0, Jm01; rtol=ftol)
	end
end
