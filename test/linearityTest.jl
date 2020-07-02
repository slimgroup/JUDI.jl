# Test linearity of sources
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using JUDI.TimeModeling, Test, LinearAlgebra, Printf

parsed_args = parse_commandline()

println("Adjoint test with ", parsed_args["nlayer"], " layers and tti: ",
        parsed_args["tti"], " and freesurface: ", parsed_args["fs"] )
### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"])
q1, srcGeometry1, recGeometry, info = setup_geom(model)
srcGeometry2 = deepcopy(srcGeometry1)
srcGeometry2.xloc[:] .= .9*srcGeometry2.xloc[:] 
srcGeometry2.zloc[:] .= .9*srcGeometry2.zloc[:]
dt = srcGeometry1.dt[1]

opt = Options(free_surface=parsed_args["fs"])
ftol = 5f-5
###################################################################################################

# Modeling operators
Pr = judiProjection(info,recGeometry)
Ps1 = judiProjection(info,srcGeometry1)
Ps2 = judiProjection(info,srcGeometry2)
F = judiModeling(info,model; options=opt)
q2 = judiVector(srcGeometry2,q1.data[1])

J = judiJacobian(Pr*F*adjoint(Ps1), q1)

d1 = Pr*F*adjoint(Ps1)*q1
d2 = Pr*F*adjoint(Ps2)*q2
d3 = Pr*F*(adjoint(Ps1)*q1 + adjoint(Ps2)*q2)
d4 = Pr*F*(adjoint(Ps1)*q1 - adjoint(Ps2)*q2)
d5 = Pr * F *adjoint(Ps1) * (2f0 * q1)

q3 = Ps1 * adjoint(F) * adjoint(Pr) * d1
q4 = Ps1 * adjoint(F) * adjoint(Pr) * (2f0 * d1)

dm2 = .5f0 .* vec(circshift(reshape(dm, model.n), (0, 20)))
lind =  J * dm
lind2 = J * (2f0 .* dm)
lind3 = J * dm2
lind4 = J * (dm + dm2)
lind5 = J * (dm - dm2)

dma = adjoint(J) * d1
dmb = adjoint(J) * (2f0 * d1)
dmc = adjoint(J) * d2
dmd = adjoint(J) * (d1 + d2)

# Test linearity F * (a + b) == F * a + F * b

println("Test linearity of F: F * (a + b) == F * a + F * b")
nd1 = norm(d3)
nd2 = norm(d1 + d2)
nd3 = norm(d3 - d1 - d2)/norm(d3)
@printf(" F * (a + b): %2.5e, F * a + F * b : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
nd1 = norm(d4)
nd2 = norm(d1 - d2)
nd3 = norm(d4 - d1 + d2)/norm(d4)
@printf(" F * (a - b): %2.5e, F * a - F * b : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)

@test isapprox(d3, d1 + d2, rtol=ftol)
@test isapprox(d4, d1 - d2, rtol=ftol)

# Test linearity F a x == a F x

println("Test linearity of F: F * (a * b) == a * F * b")
nd1 = norm(2f0 * d1)
nd2 = norm(d5)
nd3 = norm(2f0 * d1 - d5)/norm(d5)
nm1 = norm(2f0 * q3)
nm2 = norm(q4)
nm3 = norm(2f0 * q3 - q4)/norm(q4)
@printf(" a F x: %2.5e, F a x : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
@printf(" a F' x: %2.5e, F' a x : %2.5e, relative error : %2.5e \n", nm1, nm2, nm3)
@test isapprox(2f0 * d1, d5, rtol=ftol)
@test isapprox(2f0 * q3, q4, rtol=ftol)

# Test linearity J * (a + b) == J * a + J * b

println("Test linearity of J: J * (a + b) == J * a + J * b")
nd1 = norm(lind4)
nd2 = norm(lind + lind3)
nd3 = norm(lind4 - lind - lind3)/norm(lind4)
@printf(" J * (a + b): %2.5e, J * a + J * b : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
nd1 = norm(lind5)
nd2 = norm(lind - lind3)
nd3 = norm(lind5 - lind + lind3)/norm(lind5)
@printf(" J * (a - b): %2.5e, J * a - J * b : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)

@test isapprox(lind4, lind + lind3, rtol=ftol)
@test isapprox(lind5, lind - lind3, rtol=ftol)

# Test linearity J a x == a J x
println("Test linearity of J: J * (a * b) == a * J * b")
nd1 = norm(2f0 * lind)
nd2 = norm(lind2)
nd3 = norm(2f0 * lind - lind2)/norm(lind2)
nm1 = norm(2f0 * dma)
nm2 = norm(dmb)
nm3 = norm(2f0*dma - dmb)/norm(dmb)

@printf(" a J x: %2.5e, J a x : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
@printf(" a J' x: %2.5e, J' a x : %2.5e, relative error : %2.5e \n", nm1, nm2, nm3)

@test isapprox(2f0 * lind, lind2, rtol=ftol)
@test isapprox(2f0 * dma, dmb, rtol=ftol)
