# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, SegyIO, LinearAlgebra, Test

parsed_args = parse_commandline()

println("Extended Jacobian test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )
### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"])
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]
m0 = model0.m

###################################################################################

# Write shots as segy files to disk
opt = Options(sum_padding=true, dt_comp=dt, return_array=true, free_surface=parsed_args["fs"])

# Setup operators
Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model; options=opt)
F0 = judiModeling(info, model0; options=opt)
Pw = judiLRWF(info, q.data[1])

# Combined operators
A = Pr*F*adjoint(Pw)
A0 = Pr*F0*adjoint(Pw)

# Extended source weights
w = judiWeights(randn(Float32, model0.n))
J = judiJacobian(Pr*F0*Pw', w)

# Nonlinear modeling
dpred = A0*w
dD = J*dm

# Jacobian test
maxiter = 6
h = .1f0
err1 = zeros(Float32, maxiter)
err2 = zeros(Float32, maxiter)

for j=1:maxiter

    A.model.m = m0 + h*reshape(dm, model0.n)
    dobs = A*w

    err1[j] = norm(dobs - dpred)
    err2[j] = norm(dobs - dpred - h*dD)
    j == 1 ? prev = 1 : prev = j - 1
	@printf("h = %2.2e, e1 = %2.2e, rate = %2.2e", h, err1[j], err1[prev]/err1[j])
	@printf(", e2 = %2.2e, rate = %2.2e \n", err2[j], err2[prev]/err2[j])

    global h = h/2f0
end

@test isapprox(err1[end] / (err1[1]/2^(maxiter-1)), 1f0; atol=1f1)
@test isapprox(err2[end] / (err2[1]/4^(maxiter-1)), 1f0; atol=1f1)
