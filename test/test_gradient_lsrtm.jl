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

###################################################################################################

@testset "LSRTM gradient test with $(nlayer) layers, tti $(tti), viscoacoustic $(viscoacoustic). freesurface $(fs), nlind $(nlind)" for nlind=[true, false]
	@timeit TIMEROUTPUT "LSRTM gradient test, nlind=$(nlind)" begin
		# Gradient test
		ftol = (tti && fs) ? 1f-1 : 5f-2
		h = 5f-2
		maxiter = 5
		err1 = zeros(maxiter)
		err2 = zeros(maxiter)
		h_all = zeros(maxiter)

		# LS-RTM gradient and function value for m0
		Jm0, grad = lsrtm_objective(model0, q, dobs, dm; options=opt, nlind=nlind)
		dD = nlind ? (dobs - dobs0) : dobs

		# Perturbation
		dmp = 2f0*circshift(dm, 10)
		dJ = dot(grad, dmp)

		for j=1:maxiter
			dmloc = dm + h*dmp
			# LS-RTM gradient and function falue for m0 + h*dm
			Jm = lsrtm_objective(model0, q, dobs, dmloc; options=opt, nlind=nlind)[1]
			@printf("h = %2.2e, J0 = %2.2e, Jm = %2.2e \n", h, Jm0, Jm)
			# Check convergence
			err1[j] = abs(Jm - Jm0)
			err2[j] = abs(Jm - Jm0 - h*dJ)

			j == 1 ? prev = 1 : prev = j - 1
			@printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h, err1[j], err1[prev]/err1[j])
			@printf(", e2  = %2.2e, rate = %2.2e \n", err2[j], err2[prev]/err2[j])
			h_all[j] = h
			h = h * .5f0
		end

		# Check convergence rates
		rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1)
		rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1)

		# This is a linearized problem, so the whole expansion is O(dm) and
		# "second order error" should be first order
		@test isapprox(rate_1, 2f0; rtol=ftol)
		@test isapprox(rate_2, 4f0; rtol=ftol)

		# test that with zero dm we get the same as fwi_objective for residual
		if nlind
			ENV["OMP_NUM_THREADS"]=1
			Jls, gradls = lsrtm_objective(model0, q, dobs, 0f0.*dm; options=opt, nlind=nlind)
			Jfwi, gradfwi = fwi_objective(model0, q, dobs; options=opt)
			@test isapprox(Jls, Jfwi; rtol=0f0, atol=0f0)
			@test isapprox(gradls, gradfwi; rtol=0f0, atol=0f0)
		end
	end
end


# Test if lsrtm_objective produces the same value/gradient as is done by the correct algebra
@testset "LSRTM gradient linear algebra test with $(nlayer) layers, tti $(tti), viscoacoustic $(viscoacoustic), freesurface $(fs)" begin
	# Draw a random case to avoid long CI.
	isic, dft, optchk = rand([true, false], 3)
	optchk = optchk && !dft
    @timeit TIMEROUTPUT "LSRTM validity (isic=$(isic), checkpointing=$(optchk), dft=$(dft))" begin
		ftol = fs ? 1f-3 : 5f-4
		freq = dft ? [[2.5, 4.5],[3.5, 5.5],[10.0, 15.0], [30.0, 32.0]] : []
		J.options.free_surface = fs
		J.options.isic = isic
		J.options.optimal_checkpointing = optchk
		J.options.frequencies = freq

		dm1 = 2f0*circshift(dm, 10)
		d_res = dobs0 + J*dm1 - dobs
		Jm0_1 = 0.5f0 * norm(d_res)^2f0
		grad_1 = J'*d_res

		opt = J.options
		Jm0, grad = lsrtm_objective(model0, q, dobs, dm1; options=opt, nlind=true)
		Jm01, grad1 = lsrtm_objective(model0, q, dobs-dobs0, dm1; options=opt, nlind=false)

		@test isapprox(grad, grad_1; rtol=ftol)
		@test isapprox(Jm0, Jm0_1; rtol=ftol)
		@test isapprox(grad, grad1; rtol=ftol)
		@test isapprox(Jm0, Jm01; rtol=ftol)
	end
end
