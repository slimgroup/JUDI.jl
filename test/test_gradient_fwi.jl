# 2D FWI gradient test with 1 source
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

### Model
model, model0, dm = setup_model(tti, viscoacoustic, 4)
q, srcGeometry, recGeometry, f0 = setup_geom(model)
dt = srcGeometry.dt[1]

###################################################################################################

@testset "FWI gradient test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
	# Gradient test
	h = 5f-2
	maxiter = 6
	err1 = zeros(maxiter)
	err2 = zeros(maxiter)
	h_all = zeros(maxiter)
	modelH = deepcopy(model0)

	# Observed data
	opt = Options(sum_padding=true, free_surface=fs, f0=f0)
	F = judiModeling(model, srcGeometry, recGeometry; options=opt)
	d = F*q

	# FWI gradient and function value for m0
	Jm0, grad = fwi_objective(model0, q, d; options=opt)
	# Check get same misfit as l2 misifit on forward data
	Jm01 = .5f0 * norm(F(model0)*q - d)^2
	@test Jm0 ≈ Jm01
	dJ = dot(grad, dm)

	for j=1:maxiter
		# FWI function value for m0 + h*dm
		modelH.m = model0.m + h*dm
		Jm = .5f0 * norm(F(modelH)*q - d)^2

		# Check convergence
		err1[j] = abs(Jm - Jm0)
		err2[j] = abs(Jm - (Jm0 + h*dJ))
		j == 1 ? prev = 1 : prev = j - 1
		@printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h, err1[j], err1[prev]/err1[j])
		@printf(", e2  = %2.2e, rate = %2.2e \n", err2[j], err2[prev]/err2[j])
		h_all[j] = h
		h = h * .8f0
	end

	# CHeck convergence rates
	rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1)
	rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1)

	@test isapprox(rate_1, 1.25f0; rtol=5f-2)
	@test isapprox(rate_2, 1.5625f0; rtol=5f-2)
end
