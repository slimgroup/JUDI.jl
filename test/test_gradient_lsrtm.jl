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

@testset "LSRTM gradient test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
	# Gradient test
	h = 5f-2
	maxiter = 3
	err1 = zeros(maxiter, 2)
	err2 = zeros(maxiter, 2)
	h_all = zeros(maxiter)

	# Observed data
	opt = Options(sum_padding=true, free_surface=parsed_args["fs"])
	F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
	F0 = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
	J = judiJacobian(F0, q)
	d = F*q
	d0 = F0*q

	# FWI gradient and function value for m0
	Jm0, grad = lsrtm_objective(model0, q, d, dm; options=opt)
	Jm01, grad1 = lsrtm_objective(model0, q, d, dm; options=opt, nlind=true)

	dJ = dot(grad, dm)
	dJ1 = dot(grad1, dm)
	dm_pert = randn(dm.n) .* dm

	for j=1:maxiter
		# FWI gradient and function falue for m0 + h*dm
		dmloc = dm + h*dm_pert
		Jm, _ = lsrtm_objective(model0, q, d, dmloc; options=opt)
		Jm1, _ = lsrtm_objective(model0, q, d, dmloc; options=opt, nlind=true)
		Jm2 = .5*norm(J*dmloc - d)^2
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
		h = h * .8f0
	end

	for i=1:2
		# CHeck convergence rates
		rate_1 = sum(err1[1:end-1, i]./err1[2:end, i])/(maxiter - 1)
		rate_2 = sum(err2[1:end-1, i]./err2[2:end, i])/(maxiter - 1)

		# This is a linearized problem, so the whole expansiaon is O(dm) and
		# "second order error" should be first order
		@test isapprox(rate_1, 1.25f0; rtol=5f-2)
		@test isapprox(rate_2, 1.25f0; rtol=5f-2)
	end

	# Plot errors
	if isinteractive()
		loglog(h_all, err1); loglog(h_all, h_all/h_all[1]*err1[1])
		loglog(h_all, err2); loglog(h_all, ( h_all/h_all[1]).^2 * err2[1])
		legend([L"$\Phi(m) - \Phi(m0)$", "1st order", L"$\Phi(m) - \Phi(m0) - \nabla \Phi \delta m$", "2nd order"], loc="lower right")
		xlabel("h")
		ylabel(L"Error $||\cdot||^\infty$")
		title("FWI gradient test")
		
	end

	# test that with zero dm we get the same as fwi_objective for residual
	ENV["OMP_NUM_THREADS"]=1
	Jls, gradls = lsrtm_objective(model0, q, d, 0f0.*dm; options=opt, nlind=true)
	Jfwi, gradfwi = fwi_objective(model0, q, d; options=opt)
	@test isapprox(Jls, Jfwi;rtol=0, atol=0)
	@test isapprox(gradls, -gradfwi;rtol=0, atol=0)
end
