# Test linearity of sources
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]
isic = parsed_args["isic"]

### Model
model, model0, dm = setup_model(tti, nlayer)
q1, srcGeometry1, recGeometry, info = setup_geom(model)
srcGeometry2 = deepcopy(srcGeometry1)
srcGeometry2.xloc[:] .= .9*srcGeometry2.xloc[:] 
srcGeometry2.zloc[:] .= .9*srcGeometry2.zloc[:]
dt = srcGeometry1.dt[1]

opt = Options(free_surface=fs, isic=isic)
ftol = 5f-5

####################### Modeling operators ##########################################

@testset "Linearity test with $(nlayer) layers and tti $(tti) and freesurface $(fs) and isic $(isic)" begin
    # Modeling operators
    Pr = judiProjection(info,recGeometry)
    Ps1 = judiProjection(info,srcGeometry1)
    Ps2 = judiProjection(info,srcGeometry2)
    F = judiModeling(info,model; options=opt)
    q2 = judiVector(srcGeometry2,q1.data[1])

    A1 = Pr*F*adjoint(Ps1)
    A2 = Pr*F*adjoint(Ps2)
    J = judiJacobian(A1, q1)

    d1 = A1*q1
    d2 = A2*q2
    d3 = Pr*F*(adjoint(Ps1)*q1 + adjoint(Ps2)*q2)
    d4 = Pr*F*(adjoint(Ps1)*q1 - adjoint(Ps2)*q2)
    d5 = A1 * (2f0 * q1)
    d6 = (2f0 * A1) * q1

    q3 = adjoint(A1) * d1
    q4 = adjoint(A1) * (2f0 * d1)
    q5 = (2f0 * adjoint(A1)) * d1

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
    nd1_b = norm(d6)
    nd2 = norm(d5)
    nd3 = norm(2f0 * d1 - d5)/norm(d5)
    nd4 = norm(d6 - d5)/norm(d5)

    nm1 = norm(2f0 * q3)
    nm1_b = norm(q5)
    nm2 = norm(q4)
    nm3 = norm(2f0 * q3 - q4)/norm(q4)
    nm4 = norm(q5 - q4)/norm(q4)
    @printf(" a (F x): %2.5e, F a x : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
    @printf(" (a F) x: %2.5e, F a x : %2.5e, relative error : %2.5e \n", nd1_b, nd2, nd4)
    @printf(" a (F' x): %2.5e, F' a x : %2.5e, relative error : %2.5e \n", nm1, nm2, nm3)
    @printf(" (a F') x: %2.5e, F' a x : %2.5e, relative error : %2.5e \n", nm1_b, nm2, nm4)
    @test isapprox(2f0 * d1, d5, rtol=ftol)
    @test isapprox(2f0 * d1, d6, rtol=ftol)
    @test isapprox(2f0 * q3, q4, rtol=ftol)
    @test isapprox(2f0 * q3, q5, rtol=ftol)

    # Test linearity J * (a + b) == J * a + J * b
    dm2 = .5f0 * dm
    dm2.data .= circshift(dm2.data, (0, 20))
    lind =  J * dm
    lind2 = J * (2f0 .* dm)
    lind3 = J * dm2
    lind4 = J * (dm + dm2)
    lind5 = J * (dm - dm2)
    lind6 = (2f0 * J) * dm

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
    nd1_b = norm(lind6)
    nd2 = norm(lind2)
    nd3 = norm(2f0 * lind - lind2)/norm(lind2)
    nd4 = norm(lind6 - lind2)/norm(lind2)

    # Adjoint J
    dma = adjoint(J) * d1
    dmb = adjoint(J) * (2f0 * d1)
    dmc = adjoint(J) * d2
    dmd = adjoint(J) * (d1 + d2)
    dme = adjoint(2f0 * J) * d1

    nm1 = norm(2f0 * dma)
    nm1_b = norm(dme)
    nm2 = norm(dmb)
    nm3 = norm(2f0*dma - dmb)/norm(dmb)
    nm4 = norm(dme - dmb)/norm(dmb)

    @printf(" a (J x): %2.5e, J a x : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
    @printf(" (a J) x: %2.5e, J a x : %2.5e, relative error : %2.5e \n", nd1_b, nd2, nd4)

    @printf(" a (J' x): %2.5e, J' a x : %2.5e, relative error : %2.5e \n", nm1, nm2, nm3)
    @printf(" (a J') x: %2.5e, J' a x : %2.5e, relative error : %2.5e \n", nm1_b, nm2, nm4)

    @test isapprox(2f0 * lind, lind2, rtol=ftol)
    @test isapprox(2f0 * lind, lind6, rtol=ftol)
    @test isapprox(2f0 * dma, dmb, rtol=ftol)
    @test isapprox(2f0 * dma, dme, rtol=ftol)
end


####################### Extended source operators ##########################################

if tti
    ftol = 5f-4
end

@testset "Extended source linearity test with $(nlayer) layers and tti $(tti) and freesurface $(fs) and isic $(isic)" begin
    # Modeling operators
    Pr = judiProjection(info, recGeometry)
    F = judiModeling(info, model; options=opt)
    Pw = judiLRWF(info, q1.data[1])

    A = Pr*F*adjoint(Pw)
    Aa = adjoint(A)
    # Extended source weights
    w = randn(Float32, model0.n)
    x = randn(Float32, model0.n)
    w[:, 1] .= 0f0; w = vec(w)
    x[:, 1] .= 0f0; x = vec(x)

    J = judiJacobian(A, w)

    d1 = A*w
    d2 = A*x
    d3 = Pr*F*(adjoint(Pw)*w + adjoint(Pw)*x)
    d4 = Pr*F*(adjoint(Pw)*w - adjoint(Pw)*x)
    d5 = A*(2f0 * w)
    d6 = (2f0 * A) * w

    q3 = Aa * d1
    q4 = Aa * (2f0 * d1)
    q5 = (2f0 * Aa) * d1

    dm2 =  .5f0 * dm
    dm2.data .= circshift(dm2.data, (0, 20))    
    lind =  J * dm
    lind2 = J * (2f0 .* dm)
    lind3 = J * dm2
    lind4 = J * (dm + dm2)
    lind5 = J * (dm - dm2)
    lind6 = (2f0 * J) * dm

    dma = adjoint(J) * d1
    dmb = adjoint(J) * (2f0 * d1)
    dmc = adjoint(J) * d2
    dmd = adjoint(J) * (d1 + d2)
    dme = adjoint(2f0 * J) * d1

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
    nd1_b = norm(d6)
    nd2 = norm(d5)
    nd3 = norm(2f0 * d1 - d5)/norm(d5)
    nd4 = norm(d6 - d5)/norm(d5)

    nm1 = norm(2f0 * q3)
    nm1_b = norm(q5)
    nm2 = norm(q4)
    nm3 = norm(2f0 * q3 - q4)/norm(q4)
    nm4 = norm(q5 - q4)/norm(q4)
    @printf(" a (F x): %2.5e, F a x : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
    @printf(" (a F) x: %2.5e, F a x : %2.5e, relative error : %2.5e \n", nd1_b, nd2, nd4)
    @printf(" a (F' x): %2.5e, F' a x : %2.5e, relative error : %2.5e \n", nm1, nm2, nm3)
    @printf(" (a F') x: %2.5e, F' a x : %2.5e, relative error : %2.5e \n", nm1_b, nm2, nm4)
    @test isapprox(2f0 * d1, d5, rtol=ftol)
    @test isapprox(2f0 * d1, d6, rtol=ftol)
    @test isapprox(2f0 * q3, q4, rtol=ftol)
    @test isapprox(2f0 * q3, q5, rtol=ftol)

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
    nd1_b = norm(lind6)
    nd2 = norm(lind2)
    nd3 = norm(2f0 * lind - lind2)/norm(lind2)
    nd4 = norm(lind6 - lind2)/norm(lind2)

    nm1 = norm(2f0 * dma)
    nm1_b = norm(dme)
    nm2 = norm(dmb)
    nm3 = norm(2f0*dma - dmb)/norm(dmb)
    nm4 = norm(dme - dmb)/norm(dmb)

    @printf(" a (J x): %2.5e, J a x : %2.5e, relative error : %2.5e \n", nd1, nd2, nd3)
    @printf(" (a J) x: %2.5e, J a x : %2.5e, relative error : %2.5e \n", nd1_b, nd2, nd4)

    @printf(" a (J' x): %2.5e, J' a x : %2.5e, relative error : %2.5e \n", nm1, nm2, nm3)
    @printf(" (a J') x: %2.5e, J' a x : %2.5e, relative error : %2.5e \n", nm1_b, nm2, nm4)

    @test isapprox(2f0 * lind, lind2, rtol=ftol)
    @test isapprox(2f0 * lind, lind6, rtol=ftol)
    @test isapprox(2f0 * dma, dmb, rtol=ftol)
    @test isapprox(2f0 * dma, dme, rtol=ftol)
end
