using JUDI
using ArgParse, Test, Printf
using SegyIO, LinearAlgebra, Distributed, JOLI

include("utils.jl")

model, model0, dm = setup_model(false, 4)
q, srcGeometry, recGeometry, info = setup_geom(model)
dt = srcGeometry.dt[1]

### on the fly Fourier
opt = Options(frequencies=[2.5,4.5])
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

println(isapprox(vec(grad), vec(grad_1.data); rtol=1f-1))   # false
println(isapprox(Jm0, Jm0_1; rtol=1f-1))    # true
println(norm(grad)) # 0

### optimal checkpointing
opt = Options(optimal_checkpointing=true)
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

println(isapprox(vec(grad), vec(grad_1.data); rtol=1f-5))   # false
println(isapprox(Jm0, Jm0_1; rtol=1f-5))    # false

Jm01, grad1 = lsrtm_objective(model0, q, dobs-dobs0, dm1; options=opt, nlind=false) # error