# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Update Date: July 2020

using JUDI.TimeModeling, SegyIO, LinearAlgebra, Test, Printf

parsed_args = parse_commandline()

println("Jacobian test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )
### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"])
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

m0 = model0.m
######################## WITH DENSITY ############################################

# Write shots as segy files to disk
opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"])

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Ps = judiProjection(info, srcGeometry)
J = judiJacobian(Pr*F0*adjoint(Ps), q)

# Nonlinear modeling
dpred = Pr*F0*Ps'*q
dD = J*dm

# Jacobian test
maxiter = 6
h = 5f-2
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

for j=1:maxiter

    F.model.m = m0 + h*reshape(dm, model0.n)
    dobs = Pr*F*Ps'*q

    err1[j] = norm(dobs - dpred)
    err2[j] = norm(dobs - dpred - h*dD)
    j == 1 ? prev = 1 : prev = j - 1
	@printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h, err1[j], err1[prev]/err1[j])
    @printf(", e2 = %2.2e, rate = %2.2e \n", err2[j], err2[prev]/err2[j])
    
    global h = h * .8f0
end

rate_1 = sum(err1[1:end-1]./err1[2:end])/(maxiter - 1)
rate_2 = sum(err2[1:end-1]./err2[2:end])/(maxiter - 1)

@test isapprox(rate_1, 1.25f0; rtol=1f-2)
@test isapprox(rate_2, 1.5625f0; rtol=1f-2)
