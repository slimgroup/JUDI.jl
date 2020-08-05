# 2D gradient test for system 0.5*||Fq-d||_2^2
# Single simultaneous source w/ random weights for both Array and judiVector
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: August 2020

using JUDI.TimeModeling, Test, LinearAlgebra, PyPlot, Printf

include("test_utils.jl")

parsed_args = parse_commandline()

println("Gradient test for system 0.5*||Fq-d||_2^2")

println(parsed_args["nlayer"], " layers and tti: ",
		parsed_args["tti"], " and freesurface: ", parsed_args["fs"],
		" and isic: ", parsed_args["isic"] )
### Model
model, _, _ = setup_model(parsed_args["tti"], parsed_args["nlayer"])
_, _, recGeometry, info = setup_geom(model)
dt = recGeometry.dt[1]
tn = 1500f0
wavelet = ricker_wavelet(tn, dt, 0.015f0)
###################################################################################################

# Gradient test
h1 = 2f-2
h2 = 2f-2
iter = 5
error11 = zeros(iter)
error12 = zeros(iter)
error21 = zeros(iter)
error22 = zeros(iter)
h_all1 = zeros(iter)
h_all2 = zeros(iter)

# Observed data
opt1 = Options(return_array=true,free_surface=parsed_args["fs"],isic=parsed_args["isic"])
opt2 = Options(return_array=false,free_surface=parsed_args["fs"],isic=parsed_args["isic"])

Pr  = judiProjection(info, recGeometry)
F1  = judiModeling(info, model; options=opt1)
F2  = judiModeling(info, model; options=opt2)
Pw = judiLRWF(info, wavelet)
# Combined operators
F1 = Pr*F1*adjoint(Pw)
F2 = Pr*F2*adjoint(Pw)

q = randn(Float32,model.n)
q = q/norm(q)

d1 = F1*vec(q)
d2 = F2*judiWeights(q)

q1 = randn(Float32,info.n)
q1 = q1/norm(q1)
q2 = judiWeights(randn(Float32,model.n))
q2.weights[1] = q2.weights[1]/norm(q2.weights[1])

objective1 = 0.5*norm(F1*q1-d1)^2
objective2 = 0.5*norm(F2*q2-d2)^2

# Take the gradient w.r.t. q = F'(Fq-d)
grad1 = F1'*(F1*q1-d1)
grad2 = F2'*(F2*q2-d2)

# First test return_array = true
dq1 = randn(Float32,info.n)
dq1 = dq1/norm(dq1)
dobj1 = dot(grad1,dq1)

dq2 = judiWeights(randn(Float32,model.n))
dq2.weights[1] = dq2.weights[1]/norm(dq2.weights[1])
dobj2 = dot(grad2,dq2)

rate_0th_order = 2^(iter - 1)   # error decays w/ factor 2
rate_1st_order = 4^(iter - 1)   # error decays w/ factor 4

println("Test for julia Array")
for j=1:iter
	q1_now = q1 + h1*dq1
	objective_now = 0.5*norm(F1*q1_now-d1)^2

	# Check convergence
	error11[j] = abs(objective_now - objective1)
	error12[j] = abs(objective_now - objective1 - h1*dobj1)
	j == 1 ? prev = 1 : prev = j - 1
	@printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h1, error11[j], error11[prev]/error11[j])
	@printf(", e2  = %2.2e, rate = %2.2e \n", error12[j], error12[prev]/error12[j])
	h_all1[j] = h1
	global h1 = h1/2f0
	
end

# Check error decay

@test isapprox(error11[end] / error11[1] * rate_0th_order,1,atol = 20f0)
@test isapprox(error12[end] / error12[1] * rate_1st_order,1,atol = 10f0)

println("Test for judiVector")

for j=1:iter
	q2_now = q2 + h2*dq2
	objective_now = 0.5*norm(F2*q2_now-d2)^2

	# Check convergence
	error21[j] = abs(objective_now - objective2)
	error22[j] = abs(objective_now - objective2 - h2*dobj2)
	j == 1 ? prev = 1 : prev = j - 1
	@printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h2, error21[j], error21[prev]/error21[j])
	@printf(", e2  = %2.2e, rate = %2.2e \n", error22[j], error22[prev]/error22[j])
	h_all2[j] = h2
	global h2 = h2/2f0
	
end

# Check error decay

@test isapprox(error21[end] / error21[1] * rate_0th_order,1,atol = 20f0)
@test isapprox(error22[end] / error22[1] * rate_1st_order,1,atol = 10f0)

