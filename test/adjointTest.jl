# Adjoint test for F and J
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#
#

using PyCall, PyPlot, JUDI.TimeModeling, Images, LinearAlgebra, Test, ArgParse, Printf

parsed_args = parse_commandline()

println("Adjoint test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )
### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"])
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

tol = 1f-4
###################################################################################################
# Modeling operators

opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"])
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)

# Nonlinear modeling
y = F*q

# Generate random noise data vector with size of d_hat in the range of F
wave_rand = rand(size(q.data[1])).*q.data[1]
x = judiVector(srcGeometry, wave_rand)

# Forward-adjoint 
y_hat = F*x
x_hat = adjoint(F)*y

# Result F
a = dot(y, y_hat)
b = dot(x, x_hat)
@printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, a/b - 1)
@test isapprox(a, b, rtol=tol)

# Linearized modeling
J = judiJacobian(F, q)
x = vec(dm)

y_hat = J*x
x_hat = adjoint(J)*y

c = dot(y, y_hat)
d = dot(x, x_hat)
@printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
@test isapprox(c, d, rtol=tol)


###################################################################################################
# Extended source modeling

println("Extended source adjoint test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )

opt = Options(return_array=true, sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"])

Pr = judiProjection(info, recGeometry)
F = judiModeling(info, model0; options=opt)
Pw = judiLRWF(info, q.data[1])
F = Pr*F*adjoint(Pw)

# Extended source weights
w = vec(randn(Float32, model0.n))
x = vec(randn(Float32, model0.n))

# Generate random noise data vector with size of d_hat in the range of F
y = F*w

# Forward-Adjoint computation
y_hat = F*x
x_hat = adjoint(F)*y

# Result F
a = dot(y, y_hat)
b = dot(x, x_hat)
@printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, a/b - 1)
@test isapprox(a, b, rtol=tol)

# Linearized modeling
J = judiJacobian(F, w)
x = vec(dm)

y_hat = J*x
x_hat = adjoint(J)*y

c = dot(y, y_hat)
d = dot(x, x_hat)
@printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
@test isapprox(a, b, rtol=tol)
