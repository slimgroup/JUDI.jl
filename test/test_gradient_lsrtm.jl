# 2D FWI gradient test with 4 sources
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]
isic =  parsed_args["isic"]

### Model
model, model0, dm = setup_model(tti, 4)
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

###################################################################################################

@testset "LSRTM gradient test with $(nlayer) layers and tti $(tti) and freesurface $(fs) and isic $(isic)" begin
	# Gradient test
	ftol = (tti && fs) ? 1f-1 : 5f-2
	h = 5f-2
	maxiter = 5
	err1 = zeros(maxiter, 2)
	err2 = zeros(maxiter, 2)
	h_all = zeros(maxiter)

	# Observed data
	opt = Options(sum_padding=true, free_surface=fs, isic=isic)
	F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
	F0 = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
	J = judiJacobian(F0, q)
	d = F*q

	# FWI gradient and function value for m0
	Jm0, grad = lsrtm_objective(model0, q, d, dm; options=opt)
	Jm01, grad1 = lsrtm_objective(model0, q, d, dm; options=opt, nlind=true)

	# Perturbation
	dmp = 2f0*circshift(dm, 10)
	dJ = dot(grad, dmp)
	dJ1 = dot(grad1, dmp)

	for j=1:maxiter
		dmloc = dm + h*dmp
		# FWI gradient and function falue for m0 + h*dm
		Jm, _ = lsrtm_objective(model0, q, d, dmloc; options=opt)
		Jm1, _ = lsrtm_objective(model0, q, d, dmloc; options=opt, nlind=true)
		@printf("h = %2.2e, J0 = %2.2e, Jm = %2.2e \n", h, Jm0, Jm)
		@printf("h = %2.2e, J01 = %2.2e, Jm1 = %2.2e \n", h, Jm01, Jm1)
		# Check convergence
		err1[j, 1] = abs(Jm - Jm0)
		err1[j, 2] = abs(Jm1 - Jm01)
		err2[j, 1] = abs(Jm - Jm0 - h*dJ)
		err2[j, 2] = abs(Jm1 - Jm01 - h*dJ1)

		j == 1 ? prev = 1 : prev = j - 1
		for i=1:2
			@printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h, err1[j, i], err1[prev, i]/err1[j, i])
			@printf(", e2  = %2.2e, rate = %2.2e \n", err2[j, i], err2[prev, i]/err2[j, i])
		end
		h_all[j] = h
		h = h * .5f0
	end

	for i=1:2
		# CHeck convergence rates
		rate_1 = sum(err1[1:end-1, i]./err1[2:end, i])/(maxiter - 1)
		rate_2 = sum(err2[1:end-1, i]./err2[2:end, i])/(maxiter - 1)

		# This is a linearized problem, so the whole expansiaon is O(dm) and
		# "second order error" should be first order
		@test isapprox(rate_1, 2f0; rtol=ftol)
		@test isapprox(rate_2, 4f0; rtol=ftol)
	end

	# test that with zero dm we get the same as fwi_objective for residual
	ENV["OMP_NUM_THREADS"]=1
	Jls, gradls = lsrtm_objective(model0, q, d, 0f0.*dm; options=opt, nlind=true)
	Jfwi, gradfwi = fwi_objective(model0, q, d; options=opt)
	@test isapprox(Jls, Jfwi;rtol=0, atol=0)
	@test isapprox(gradls, gradfwi;rtol=0, atol=0)
end
