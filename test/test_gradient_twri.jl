# 2D FWI gradient test with 4 sources
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

using JUDI.TimeModeling, Test, LinearAlgebra, PyPlot, Printf

parsed_args = parse_commandline()


nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]

### Model
model, model0, dm = setup_model(parsed_args["tti"], 4)
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

###################################################################################################

@testset "TWRI gradient test w.r.t m with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
	optw = TWRIOptions(grad_corr=false, comp_alpha=false, weight_fun=nothing, eps=0, params=:m)
	# Gradient test
	h = 5f-2
	maxiter = 5
	err1 = zeros(maxiter)
	err2 = zeros(maxiter)
	h_all = zeros(maxiter)
	modelH = deepcopy(model0)

	# Observed data
	opt = Options(sum_padding=true, free_surface=parsed_args["fs"])
	F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
	F0 = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
	d = F*q
	d0 = F0*q
	y = 2.5f0*(d0 - d)

	# FWI gradient and function value for m0
	Jm0, gradm = twri_objective(model0, q, d, y; options=opt, optionswri=optw)

	dJ = dot(gradm, dm)
	@printf("Perturbation size is %2.2e and reference objective function is %2.4e \n", dJ, Jm0)

	for j=1:maxiter
		# FWI gradient and function falue for m0 + h*dm
		modelH.m = model0.m + h*dm
		Jm, _ = twri_objective(modelH, q, d, y; options=opt, optionswri=optw)

		# Check convergence
		err1[j] = abs(Jm - Jm0)
		err2[j] = abs(Jm - Jm0 - h*dJ)
		j == 1 ? prev = 1 : prev = j - 1
		@printf("h = %2.2e, phi = %2.4e, e1 = %2.2e, rate = %2.2e", h, Jm, err1[j], err1[prev]/err1[j])
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


@testset "TWRI gradient test w.r.t y with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
	optw = TWRIOptions(grad_corr=false, comp_alpha=false, weight_fun=nothing, eps=0, params=:y)
	# Gradient test
	h = 5f-2
	maxiter = 5
	err1 = zeros(maxiter)
	err2 = zeros(maxiter)
	h_all = zeros(maxiter)
	modelH = deepcopy(model0)

	# Observed data
	opt = Options(sum_padding=true, free_surface=parsed_args["fs"])
	F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
	F0 = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
	d = F*q
	d0 = F0*q
	y = .1f0 * (d0 - d)
	dy = .05f0*(d0 - d)

	# FWI gradient and function value for m0
	Jm0, grady = twri_objective(model0, q, d, y; options=opt, optionswri=optw)

	dJ = dot(grady, dy)
	@printf("Perturbation size is %2.4e and reference objective function is %2.4e \n", dJ, Jm0)

	for j=1:maxiter
		# FWI gradient and function falue for m0 + h*dm
		yloc = y + h * dy
		Jm, _ = twri_objective(model0, q, d, yloc;options=opt, optionswri=optw)

		# Check convergence
		err1[j] = abs(Jm - Jm0)
		err2[j] = abs(Jm - Jm0 - h*dJ)
		j == 1 ? prev = 1 : prev = j - 1
		@printf("h = %2.2e, phi = %2.4e, e1 = %2.2e, rate = %2.2e", h, Jm, err1[j], err1[prev]/err1[j])
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