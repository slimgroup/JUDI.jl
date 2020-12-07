# Adjoint test for F and J
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#

using Distributed

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]

# Set parallel if specified
nw = parsed_args["parallel"]
if nw > 1 && nworkers() < nw
    addprocs(nw-nworkers() + 1; exeflags=["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
end

@everywhere using JUDI, LinearAlgebra, Test, Distributed, Printf

### Model
model, model0, dm = setup_model(parsed_args["tti"], parsed_args["nlayer"])
q, srcGeometry, recGeometry, info = setup_geom(model; nsrc=nw)
dt = srcGeometry.dt[1]

tol = 5f-4
(tti && fs) && (tol = 5f-3)
###################################################################################################
# Modeling operators
@testset "Adjoint test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin 

    opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"])
    F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)

    # Nonlinear modeling
    y = F*q

    # Generate random noise data vector with size of d_hat in the range of F
    wave_rand = (.5 .+ rand(size(q.data[1]))).*q.data[1]
    q_rand = judiVector(srcGeometry, wave_rand)

    # Forward-adjoint
    d_hat = F*q_rand
    q_hat = adjoint(F)*y

    # Result F
    a = dot(y, d_hat)
    b = dot(q_rand, q_hat)
    @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, (a - b)/(a + b))
    @test isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)

    # Linearized modeling
    J = judiJacobian(F, q)

    ld_hat = J*dm
    dm_hat = adjoint(J)*y

    c = dot(ld_hat, y)
    d = dot(dm_hat, dm)
    @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, (c - d)/(c + d))
    @test isapprox(c/(c+d), d/(c+d), atol=tol, rtol=0)

end
###################################################################################################
# Extended source modeling
if parsed_args["tti"] &&  parsed_args["fs"]
    # FS + tti leads to slightly worst (still fairly ok) accuracy
    tol = 5f-3
end

@testset "Extended source adjoint test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin

    opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"])
    F = judiModeling(info, model0, srcGeometry, recGeometry; options=opt)
    # Nonlinear modeling
    y = F*q

    Pr = judiProjection(info, recGeometry)
    Fw = judiModeling(info, model0; options=opt)
    Pw = judiLRWF(info, q.data[1])
    Fw = Pr*Fw*adjoint(Pw)

    # Extended source weights
    w = .5f0 .+ rand(model0.n...)
    parsed_args["fs"] ? w[:, 1:2] .= 0f0 : nothing
    w = judiWeights(w; nsrc=nw)

    # Forward-Adjoint computation
    dw_hat = Fw*w
    w_hat = adjoint(Fw)*y

    # Result F
    a = dot(y, dw_hat)
    b = dot(w, w_hat)
    @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, (a - b)/(a + b))
    @test isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)

    # Linearized modeling
    Jw = judiJacobian(Fw, w)

    ddw_hat = Jw*dm
    dmw_hat = adjoint(Jw)*y

    c = dot(y, ddw_hat)
    d = dot(dm, dmw_hat)
    @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, (c - d)/(c + d))
    @test isapprox(c/(c+d), d/(c+d), atol=tol, rtol=0)

end
