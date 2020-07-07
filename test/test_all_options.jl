# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: July 2020

using PyCall, PyPlot, JUDI.TimeModeling, Images, LinearAlgebra, Test, ArgParse, Printf

parsed_args = parse_commandline()

println("Adjoint test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )
### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"])
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

##################################ISIC########################################################
println("Testing isic")
opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], isic=true)
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)

# Linearized modeling
J = judiJacobian(F, q)
x = vec(dm)

y0 = F*q
y_hat = J*x
x_hat1 = adjoint(J)*y0

c = dot(y0, y_hat)
d = dot(x, x_hat1)
@printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
@test isapprox(c, d, rtol=5f-2)
@test !isnan(norm(y0))
@test !isnan(norm(y_hat))
@test !isnan(norm(x_hat1))

##################################checkpointing###############################################
println("Testing checkpointing")
opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], optimal_checkpointing=true)
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)

# Linearized modeling
J = judiJacobian(F, q)

y_hat = J*x
x_hat2 = adjoint(J)*y0

c = dot(y0, y_hat)
d = dot(x, x_hat2)
@printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
@test isapprox(c, d, rtol=5f-4)

@test !isnan(norm(y_hat))
@test !isnan(norm(x_hat2))

##################################DFT#########################################################
println("Testing DFT")

opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], frequencies=[[2.5, 4.5]])
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)

# Linearized modeling
J = judiJacobian(F, q)

y_hat = J*x
x_hat3 = adjoint(J)*y0

c = dot(y0, y_hat)
d = dot(x, x_hat3)
@printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
@test !isnan(norm(y_hat))
@test !isnan(norm(x_hat3))

##################################subsampling#################################################
println("Testing subsampling")
opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], subsampling_factor=4)
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)

# Linearized modeling
J = judiJacobian(F, q)

y_hat = J*x
x_hat4 = adjoint(J)*y0

c = dot(y0, y_hat)
d = dot(x, x_hat4)
@printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
@test isapprox(c, d, rtol=1f-2)
@test !isnan(norm(y_hat))
@test !isnan(norm(x_hat4))


##################################ISIC + DFT #########################################################
println("Testing isic+dft")
opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"],
              isic=true, frequencies=[[2.5, 4.5]])
F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)

# Linearized modeling
J = judiJacobian(F, q)

y_hat = J*x
x_hat5 = adjoint(J)*y0

c = dot(y0, y_hat)
d = dot(x, x_hat5)
@printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, c/d - 1)
@test !isnan(norm(y_hat))
@test !isnan(norm(x_hat5))
