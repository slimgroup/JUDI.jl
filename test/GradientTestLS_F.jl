# 2D gradient test for system 0.5*||Fq-d||_2^2 -- gradient w.r.t. q
# Single simultaneous source w/ random weights for both Array and judiVector
# Author: Ziyi Yin, ziyi.yin@gatech.edu
# Date: August 2020

using JUDI.TimeModeling, Test, LinearAlgebra, PyPlot, Printf

parsed_args = parse_commandline()

println("Gradient test for system 0.5*||Fq-d||_2^2 w.r.t. q")

println(parsed_args["nlayer"], " layers and tti: ",
		parsed_args["tti"], " and freesurface: ", parsed_args["fs"])
### Model
model, _, _ = setup_model(parsed_args["tti"], parsed_args["nlayer"])
_, _, recGeometry, info = setup_geom(model)
dt = recGeometry.dt[1]
tn = 1500f0
wavelet = ricker_wavelet(tn, dt, 0.015f0)
###################################################################################################

# Gradient test
h1 = 2f-1
h2 = 2f-1

alpha = 0.5f0 # decay rate of step length h

iter = 5
error11 = zeros(iter)
error12 = zeros(iter)
error21 = zeros(iter)
error22 = zeros(iter)
h_all1 = zeros(iter)
h_all2 = zeros(iter)

# Observed data
opt1 = Options(return_array=true,free_surface=parsed_args["fs"])
opt2 = Options(return_array=false,free_surface=parsed_args["fs"])

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
	global h1 = h1*alpha
	
end

# Check error decay

rate_11 = sum(error11[1:end-1]./error11[2:end])/(iter - 1)
rate_12 = sum(error12[1:end-1]./error12[2:end])/(iter - 1)
@test isapprox(rate_11, 1/alpha; rtol=5f-1)
@test isapprox(rate_12, 1/alpha^2; rtol=5f-1)

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

rate_21 = sum(error21[1:end-1]./error21[2:end])/(iter - 1)
rate_22 = sum(error22[1:end-1]./error22[2:end])/(iter - 1)
@test isapprox(rate_21, 1/alpha; rtol=5f-1)
@test isapprox(rate_22, 1/alpha^2; rtol=5f-1)

# Check if the value of gradients are the same

println("Test if the gradients from Array or judiVector/judiWeights are the same")
q_array = randn(Float32,info.n)
d_array = F1*randn(Float32,info.n)
grad_array = F1'*(F1*q_array-d_array)

q_judi = judiWeights(reshape(q_array,model.n))
d_judi = judiVector(recGeometry,reshape(d_array,recGeometry.nt[1],length(recGeometry.xloc[1])))
grad_judi = F2'*(F2*q_judi-d_judi)

@test isapprox(norm(grad_array-vec(grad_judi.weights[1]))/norm(grad_array),0,atol=1f-3)
