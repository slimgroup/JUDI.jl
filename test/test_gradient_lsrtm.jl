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

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]

### Model
model, model0, dm = setup_model(tti, 4)
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

###################################################################################################

@testset "LSRTM gradient test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
	# Gradient test
	ftol = (tti && fs) ? 1f-1 : 5f-2
	h = 5f-2
	maxiter = 5
	err1 = zeros(maxiter, 2)
	err2 = zeros(maxiter, 2)
	h_all = zeros(maxiter)

	# Observed data
	opt = Options(sum_padding=true, free_surface=fs)
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

		# This is a linearized problem, so the whole expansion is O(dm) and
		# "second order error" should be first order
		@test isapprox(rate_1, 2f0; rtol=ftol)
		@test isapprox(rate_2, 4f0; rtol=ftol)
	end

	# test that with zero dm we get the same as fwi_objective for residual
	ENV["OMP_NUM_THREADS"]=1
	Jls, gradls = lsrtm_objective(model0, q, d, 0f0.*dm; options=opt, nlind=true)
	Jfwi, gradfwi = fwi_objective(model0, q, d; options=opt)
	@test isapprox(Jls, Jfwi; rtol=0f0, atol=0f0)
	@test isapprox(gradls, gradfwi; rtol=0f0, atol=0f0)
end

# Test if lsrtm_objective produces the same value/gradient as is done by the correct algebra
cases = [(true, false, true), (false, false, true), (true, true, false), (true, false, false), (false, true, false), (false, false, false)]	# DFT and optimal_checkpointing normally don't co-exist
@testset "LSRTM gradient linear algebra test with $(nlayer) layers and tti $(tti) and freesurface $(fs) and isic $(isic) and optimal_checkpointing $(optchk) and DFT $(dft)" for (isic,optchk,dft) = cases

	ftol = (fs||dft) ? 1f-2 : 1f-3
	freq = dft ? [[2.5, 4.5],[3.5, 5.5],[10.0, 15.0], [30.0, 32.0]] : []
	opt = Options(free_surface=fs, isic=isic, optimal_checkpointing=optchk, frequencies=freq)
	F = judiModeling(info, model, srcGeometry, recGeometry; options=opt)
	dobs = F*q
	F0 = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
	dobs0 = F0*q
	J = judiJacobian(F0, q)

	dm1 = 2f0*circshift(dm, 10)
	d_res = dobs0 + J*dm1 - dobs
	Jm0_1 = 0.5f0 * norm(d_res)^2f0
	grad_1 = J'*d_res
	
	Jm0, grad = lsrtm_objective(model0, q, dobs, dm1; options=opt, nlind=true)
	Jm01, grad1 = lsrtm_objective(model0, q, dobs-dobs0, dm1; options=opt, nlind=false)

	@test isapprox(vec(grad), vec(grad_1.data); rtol=ftol)
	@test isapprox(Jm0, Jm0_1; rtol=ftol)
	@test isapprox(vec(grad), vec(grad1); rtol=ftol)
	@test isapprox(Jm0, Jm01; rtol=ftol)
			
end
