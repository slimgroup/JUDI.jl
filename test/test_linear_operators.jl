# Unit tests for JUDI linear operators (without PDE solves)
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

# Tests
function test_transpose(Op)
    @test isequal(size(Op), size(conj(Op)))
    @test isequal(reverse(size(Op)), size(transpose(Op)))
    @test isequal(reverse(size(Op)), size(adjoint(Op)))
    @test isequal(reverse(size(Op)), size(transpose(Op)))
    return true
end

function test_getindex(Op, nsrc)
    # requires: Op.info.nsrc == 2
    Op_sub = Op[1]
    @test isequal(Op_sub.info.nsrc, 1)
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), convert(Tuple{Int64, Int64}, size(Op) ./ nsrc))

    inds = nsrc > 1 ? (1:nsrc) : 1
    Op_sub = Op[inds]
    @test isequal(Op_sub.info.nsrc, nsrc)
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), size(Op))
    return true
end


########################################################### judiModeling ###############################################

@testset "judiModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]

    info = example_info(nsrc=nsrc)
    model = example_model()
    F_forward = judiModeling(info, model; options=Options())
    F_adjoint = judiModelingAdjoint(info, model; options=Options())

    @test isequal(typeof(F_forward), judiModeling{Float32, Float32})
    @test isequal(typeof(F_adjoint), judiModelingAdjoint{Float32, Float32})

    # conj, transpose, adjoint
    @test test_transpose(F_forward)
    @test test_transpose(F_adjoint)

    # get index
    @test test_getindex(F_forward, nsrc)
    @test test_getindex(F_adjoint, nsrc)

    if VERSION>v"1.2"
        a = randn(Float32, model.n...)
        F2 = F_forward(;m=a)
        @test isapprox(F2.model.m, a)
        F2 = F_forward(Model(model.n, model.d, model.o, a))
        @test isapprox(F2.model.m, a)
        @test F2.model.n == model.n
    end

end

############################################################# judiPDE ##################################################

@testset "judiPDE Unit Test with $(nsrc) sources" for nsrc=[1, 2]

    info = example_info(nsrc=nsrc)
    model = example_model()
    rec_geometry = example_rec_geometry(nsrc=nsrc)

    PDE_forward = judiPDE("PDE", info, model, rec_geometry; options=Options())
    PDE_adjoint = judiPDEadjoint("PDEadjoint", info, model, rec_geometry; options=Options())

    @test isequal(typeof(PDE_forward), judiPDE{Float32, Float32})
    @test isequal(typeof(PDE_adjoint), judiPDEadjoint{Float32, Float32})

    # conj, transpose, adjoint
    @test test_transpose(PDE_forward)
    @test test_transpose(PDE_adjoint)

    # get index
    @test test_getindex(PDE_forward, nsrc)
    @test test_getindex(PDE_adjoint, nsrc)

    # Multiplication w/ judiProjection
    src_geometry = example_src_geometry()
    Ps = judiProjection(info, src_geometry)

    PDE = PDE_forward*transpose(Ps)
    @test isequal(typeof(PDE), judiPDEfull{Float32, Float32})
    @test isequal(PDE.recGeometry, rec_geometry)
    @test isequal(PDE.srcGeometry, src_geometry)

    PDEad = PDE_adjoint*transpose(Ps)
    @test isequal(typeof(PDEad), judiPDEfull{Float32, Float32})
    @test isequal(PDEad.srcGeometry, rec_geometry)
    @test isequal(PDEad.recGeometry, src_geometry)

end

######################################################### judiPDEfull ##################################################

@testset "judiPDEfull Unit Test with $(nsrc) sources" for nsrc=[1, 2]

    info = example_info(nsrc=nsrc)
    model = example_model()
    rec_geometry = example_rec_geometry(nsrc=nsrc)
    src_geometry = example_src_geometry(nsrc=nsrc)
    PDE = judiModeling(info, model, src_geometry, rec_geometry; options=Options())

    @test isequal(typeof(PDE), judiPDEfull{Float32, Float32})
    @test isequal(PDE.recGeometry, rec_geometry)
    @test isequal(PDE.srcGeometry, src_geometry)

    @test test_transpose(PDE)
    @test test_getindex(PDE, nsrc)

end

######################################################## judiJacobian ##################################################

@testset "judiJacobian Unit Test with $(nsrc) sources" for nsrc=[1, 2]

    info = example_info(nsrc=nsrc)
    model = example_model()
    rec_geometry = example_rec_geometry(nsrc=nsrc)
    src_geometry = example_src_geometry(nsrc=nsrc)
    wavelet = randn(Float32, src_geometry.nt[1])
    PDE = judiModeling(info, model, src_geometry, rec_geometry; options=Options())
    q = judiVector(src_geometry, wavelet)

    J = judiJacobian(PDE, q)

    @test isequal(typeof(J), judiJacobian{Float32, Float32})
    @test isequal(J.recGeometry, rec_geometry)
    @test isequal(J.srcGeometry, src_geometry)
    @test isequal(size(J)[2], prod(model.n))
    @test test_transpose(J)

    # get index
    J_sub = J[1]
    @test isequal(J_sub.info.nsrc, 1)
    @test isequal(J_sub.model, J.model)
    @test isequal(size(J_sub)[1], Int(size(J)[1]/nsrc))

    inds = nsrc > 1 ? (1:nsrc) : 1
    J_sub = J[inds]
    @test isequal(J_sub.info.nsrc, nsrc)
    @test isequal(J_sub.model, J.model)
    @test isequal(size(J_sub), size(J))

end


######################################################## judiJacobian ##################################################

@testset "judiJacobianExtended Unit Test with $(nsrc) sources" for nsrc=[1, 2]

    info = example_info(nsrc=nsrc)
    model = example_model()
    rec_geometry = example_rec_geometry(nsrc=nsrc)
    wavelet = randn(Float32, rec_geometry.nt[1])
    # Setup operators
    Pr = judiProjection(info, rec_geometry)
    F = judiModeling(info, model)
    Pw = judiLRWF(info, wavelet)
    w = judiWeights(randn(Float32, model.n); nsrc=nsrc)
    J = judiJacobian(Pr*F*Pw', w)

    @test isequal(typeof(J), judiJacobianExQ{Float32, Float32})
    @test isequal(J.recGeometry, rec_geometry)
    for i=1:nsrc
        @test isapprox(J.wavelet[i], wavelet)
        @test isapprox(J.weights[i], w.weights[i])
    end
    @test isequal(size(J)[2], prod(model.n))
    @test test_transpose(J)

    # get index
    J_sub = J[1]
    @test isequal(J_sub.info.nsrc, 1)
    @test isequal(J_sub.model, J.model)
    @test isapprox(J_sub.weights, J.weights[1:1])
    @test isequal(size(J_sub)[1], Int(size(J)[1]/nsrc))

    inds = nsrc > 1 ? (1:nsrc) : 1
    J_sub = J[inds]
    @test isequal(J_sub.info.nsrc, nsrc)
    @test isequal(J_sub.model, J.model)
    @test isapprox(J_sub.weights, J.weights[1:nsrc])
    @test isequal(size(J_sub), size(J))

    # Test Pw alone
    P1 = subsample(Pw, 1)
    @test isapprox(P1.wavelet, Pw[1].wavelet)
    @test isapprox(conj(Pw).wavelet, Pw.wavelet)
    @test size(conj(Pw)) == size(Pw)
    @test size(transpose(Pw)) == size(Pw)[end:-1:1]
    @test isapprox(transpose(Pw).wavelet, Pw.wavelet)

    P1 = adjoint(Pw) * w
    @test isequal(typeof(P1), judiExtendedSource{Float32})

end


####################################################### judiProjection #################################################

@testset "judiProjection Unit Test with $(nsrc) sources" for nsrc=[1, 2]

    info = example_info(nsrc=nsrc)
    rec_geometry = example_rec_geometry(nsrc=nsrc)
    src_geometry = example_src_geometry(nsrc=nsrc)
    wavelet = randn(Float32, src_geometry.nt[1])
    q = judiVector(src_geometry, wavelet)

    Pr = judiProjection(info, rec_geometry)
    Ps = judiProjection(info, src_geometry)

    @test isequal(typeof(Pr), judiProjection{Float32, Float32})
    @test isequal(typeof(Ps), judiProjection{Float32, Float32})
    @test isequal(Pr.geometry, rec_geometry)
    @test isequal(Ps.geometry, src_geometry)

    @test test_transpose(Pr)
    @test test_transpose(Ps)

    Pr_sub = Pr[1]
    @test isequal(Pr_sub.info.nsrc, 1)
    @test isequal(size(Pr_sub), convert(Tuple{Int64, Int64}, size(Pr) ./ nsrc))

    inds = nsrc > 1 ? (1:nsrc) : 1
    Pr_sub = Pr[inds]
    @test isequal(Pr_sub.info.nsrc, nsrc)
    @test isequal(size(Pr_sub), size(Pr))

    RHS = transpose(Ps)*q
    @test isequal(typeof(RHS), judiRHS{Float32})
    @test isequal(RHS.geometry, q.geometry)

end
