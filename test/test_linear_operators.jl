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
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), size(Op)) # Sizes are purely symbolic so number of sourcs don't matter

    inds = nsrc > 1 ? (1:nsrc) : 1
    Op_sub = Op[inds]
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), size(Op))
    return true
end


########################################################## judiModeling ###############################################

@testset "judiModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiModeling nsrc=$(nsrc)" begin
        info = example_info(nsrc=nsrc)
        model = example_model()
        F_forward = judiModeling(info, model; options=Options())
        F_adjoint = judiModelingAdjoint(info, model; options=Options())

        model = example_model()
        F_forward = judiModeling(model; options=Options())
        F_adjoint = adjoint(F_forward)

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiModeling{Float32, :adjoint})

        # get index
        @test test_getindex(F_forward, nsrc)
        @test test_getindex(F_adjoint, nsrc)

        # get index
        @test test_getindex(F_forward, nsrc)
        @test test_getindex(F_adjoint, nsrc)

    #     if VERSION>v"1.2"
    #         a = randn(Float32, model.n...)
    #         F2 = F_forward(;m=a)
    #         @test isapprox(F2.model.m, a)
    #         F2 = F_forward(Model(model.n, model.d, model.o, a))
    #         @test isapprox(F2.model.m, a)
    #         @test F2.model.n == model.n
    #     end
    end
end

######################################################## judiJacobian ##################################################

@testset "judiJacobian Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiJacobian nsrc=$(nsrc)" begin
        info = example_info(nsrc=nsrc)
        model = example_model()
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        src_geometry = example_src_geometry(nsrc=nsrc)
        wavelet = randn(Float32, src_geometry.nt[1])
        PDE = judiModeling(info, model, src_geometry, rec_geometry; options=Options())
        q = judiVector(src_geometry, wavelet)

        model = example_model()
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        src_geometry = example_src_geometry(nsrc=nsrc)
        wavelet = randn(Float32, src_geometry.nt[1])
        PDE = judiModeling(model, src_geometry, rec_geometry; options=Options())
        q = judiVector(src_geometry, wavelet)

        @test isequal(typeof(J), judiJacobian{Float32, Float32})
        @test isequal(J.recGeometry, rec_geometry)
        @test isequal(J.source.geometry, src_geometry)
        @test all(isequal(J.source.data[i], wavelet) for i=1:nsrc)
        @test isequal(size(J)[2], prod(model.n))
        @test test_transpose(J)

        @test isequal(typeof(J), judiJacobian{Float32, :born, typeof(PDE)})
        @test isequal(J.F.rInterpolation.geometry, rec_geometry)
        @test isequal(J.F.qInjection.geometry, src_geometry)
        @test isequal(J.q.geometry, src_geometry)

        @test all(isequal(J.q.data[i], wavelet) for i=1:nsrc)
        @test isequal(size(J)[2], JUDI.space_size(2))
        @test test_transpose(J)

        # get index
        J_sub = J[1]
        @test isequal(J_sub.model, J.model)
        @test isequal(J_sub.F, J.F)

        inds = nsrc > 1 ? (1:nsrc) : 1
        J_sub = J[inds]
        @test isequal(J_sub.model, J.model)
        @test isequal(J_sub.F, J.F[inds])

        inds = nsrc > 1 ? (1:nsrc) : 1
        J_sub = J[inds]
        @test isequal(J_sub.info.nsrc, nsrc)
        @test isequal(J_sub.model, J.model)
        @test isequal(size(J_sub), size(J))
    end
end

######################################################## judiJacobian ##################################################

@testset "judiJacobianExtended Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiJacobianExtended nsrc=$(nsrc)" begin
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

        model = example_model()
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        wavelet = randn(Float32, rec_geometry.nt[1])
        # Setup operators
        Pr = judiProjection(rec_geometry)
        F = judiModeling(model)
        Pw = judiLRWF(nsrc, wavelet)
        w = judiWeights(randn(Float32, model.n); nsrc=nsrc)
        J = judiJacobian(Pr*F*Pw', w)

        @test isequal(typeof(J), judiJacobian{Float32, :born, typeof(PDE)})
        @test isequal(J.F.rInterpolation.geometry, rec_geometry)
        for i=1:nsrc
            @test isapprox(J.F.qInjection.wavelet[i], wavelet)
            @test isapprox(J.q.data[i], w.data[i])
        end
        @test isequal(size(J)[2], JUDI.space_size(2))
        @test test_transpose(J)

        # get index
        J_sub = J[1]
        @test isequal(J_sub.model, J.model)
        @test isapprox(J_sub.q.weights, J.q.weights[1])

        inds = nsrc > 1 ? (1:nsrc) : 1
        J_sub = J[inds]
        @test isequal(J_sub.model, J.model)
        @test isapprox(J_sub.q.weights, J.q.weights[inds])

        # Test Pw alone
        P1 = subsample(Pw, 1)
        @test isapprox(P1.wavelet, Pw[1].wavelet)
        @test isapprox(conj(Pw).wavelet, Pw.wavelet)
        @test size(conj(Pw)) == size(Pw)
        @test isapprox(transpose(Pw).wavelet, Pw.wavelet)
    end
end


####################################################### judiProjection #################################################

@testset "judiProjection Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiProjection nsrc=$(nsrc)" begin
        info = example_info(nsrc=nsrc)
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        src_geometry = example_src_geometry(nsrc=nsrc)
        wavelet = randn(Float32, src_geometry.nt[1])
        q = judiVector(src_geometry, wavelet)

        rec_geometry = example_rec_geometry(nsrc=nsrc)
        src_geometry = example_src_geometry(nsrc=nsrc)
        wavelet = randn(Float32, src_geometry.nt[1])
        q = judiVector(src_geometry, wavelet)

        Pr = judiProjection(rec_geometry)
        Ps = judiProjection(src_geometry)

        @test isequal(typeof(Pr), judiProjection{Float32})
        @test isequal(typeof(Ps), judiProjection{Float32})
        @test isequal(Pr.geometry, rec_geometry)
        @test isequal(Ps.geometry, src_geometry)

        Pr_sub = Pr[1]
        @test isequal(Pr_sub.info.nsrc, 1)
        @test isequal(size(Pr_sub), convert(Tuple{Int64, Int64}, size(Pr) ./ nsrc))

        Pr_sub = Pr[1]
        @test isequal(Pr_sub.geometry, rec_geometry[1])

        inds = nsrc > 1 ? (1:nsrc) : 1
        Pr_sub = Pr[inds]
        @test isequal(Pr_sub.geometry, rec_geometry[inds])
    end
end
