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
tti = parsed_args["fs"]
fs =  parsed_args["fs"]

### Model
model, model0, dm = setup_model(parsed_args["tti"], 4)
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

###################################################################################################

@testset "FWI gradient test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
	# Gradient test
	h = 5f-2
	maxiter = 6
	err1 = zeros(maxiter)
	err2 = zeros(maxiter)
	h_all = zeros(maxiter)
	modelH = deepcopy(model0)

	# Observed data
	opt = Options(sum_padding=true, free_surface=parsed_args["fs"])
	F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
	d = F*q

	# FWI gradient and function value for m0
	Jm0, grad = fwi_objective(model0, q, d; options=opt)

	dJ = dot(grad, vec(dm))

	for j=1:maxiter
		# FWI gradient and function falue for m0 + h*dm
		modelH.m = model0.m + h*reshape(dm, model.n)
		Jm, gradm = fwi_objective(modelH, q, d;options=opt)

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


	# Plot errors
	if isinteractive()
		loglog(h_all, err1); loglog(h_all, h_all/h_all[1]*err1[1])
		loglog(h_all, err2); loglog(h_all, ( h_all/h_all[1]).^2 * err2[1])
		legend([L"$\Phi(m) - \Phi(m0)$", "1st order", L"$\Phi(m) - \Phi(m0) - \nabla \Phi \delta m$", "2nd order"], loc="lower right")
		xlabel("h")
		ylabel(L"Error $||\cdot||^\infty$")
		title("FWI gradient test")
		
	end
end