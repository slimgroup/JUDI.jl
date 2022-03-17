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
    # requires: Op.nsrc == 2
    Op_sub = Op[1]
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), size(Op)) # Sizes are purely symbolic so number of sourcs don't matter

    inds = nsrc > 1 ? (1:nsrc) : 1
    Op_sub = Op[inds]
    @test isequal(Op_sub.model, Op.model)
    @test isequal(size(Op_sub), size(Op))
    return true
end

model = example_model()
########################################################## judiModeling ###############################################
@testset "judiModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiModeling nsrc=$(nsrc)" begin
        F_forward = judiModeling(model; options=Options())
        F_adjoint = adjoint(F_forward)

        @test isequal(size(F_forward)[1], time_space_size(2))
        @test isequal(size(F_forward)[2], time_space_size(2))
        @test isequal(size(F_adjoint)[1], time_space_size(2))
        @test isequal(size(F_adjoint)[2], time_space_size(2))

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiModeling{Float32, :adjoint})

        # get index
        @test test_getindex(F_forward, nsrc)
        @test test_getindex(F_adjoint, nsrc)

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
end

@testset "judiPointSourceModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiPointSourceModeling nsrc=$(nsrc)" begin
        src_geometry = example_src_geometry(nsrc=nsrc)
        F_forward = judiModeling(model; options=Options()) * judiProjection(src_geometry)'
        F_adjoint = adjoint(F_forward)

        @test isequal(size(F_forward)[1], time_space_size(2))
        @test isequal(size(F_forward)[2], JUDI._rec_space)
        @test isequal(size(F_adjoint)[1], JUDI._rec_space)
        @test isequal(size(F_adjoint)[2], time_space_size(2))

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiPointSourceModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiDataModeling{Float32, :adjoint})

        # get index
        @test test_getindex(F_forward, nsrc)
        @test test_getindex(F_adjoint, nsrc)

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
end

@testset "judiDataModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiDataModeling nsrc=$(nsrc)" begin
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        F_forward = judiProjection(rec_geometry)*judiModeling(model; options=Options())
        F_adjoint = adjoint(F_forward)

        @test isequal(size(F_forward)[1], JUDI._rec_space)
        @test isequal(size(F_forward)[2], time_space_size(2))
        @test isequal(size(F_adjoint)[1], time_space_size(2))
        @test isequal(size(F_adjoint)[2], JUDI._rec_space)

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiDataModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiPointSourceModeling{Float32, :adjoint})

        # get index
        @test test_getindex(F_forward, nsrc)
        @test test_getindex(F_adjoint, nsrc)

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
end

@testset "judiDataSourceModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiDataSourceModeling nsrc=$(nsrc)" begin
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        src_geometry = example_src_geometry(nsrc=nsrc)
        F_forward = judiModeling(model, src_geometry, rec_geometry; options=Options())
        F_adjoint = adjoint(F_forward)

        @test isequal(size(F_forward)[1], JUDI._rec_space)
        @test isequal(size(F_forward)[2], JUDI._rec_space)
        @test isequal(size(F_adjoint)[1], JUDI._rec_space)
        @test isequal(size(F_adjoint)[2], JUDI._rec_space)

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiDataSourceModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiDataSourceModeling{Float32, :adjoint})

        # get index
        @test test_getindex(F_forward, nsrc)
        @test test_getindex(F_adjoint, nsrc)

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
end

######################################################## judiJacobian ##################################################
@testset "judiJacobian Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiJacobian nsrc=$(nsrc)" begin
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        src_geometry = example_src_geometry(nsrc=nsrc)
        wavelet = randn(Float32, src_geometry.nt[1])
        PDE = judiModeling(model, src_geometry, rec_geometry; options=Options())
        q = judiVector(src_geometry, wavelet)
        J = judiJacobian(PDE, q)

        @test isequal(typeof(J), judiJacobian{Float32, :born, typeof(PDE)})
        @test isequal(J.F.rInterpolation.geometry, rec_geometry)
        @test isequal(J.F.qInjection.geometry, src_geometry)
        @test isequal(J.q.geometry, src_geometry)
        @test isequal(size(J)[1], JUDI._rec_space)
        @test isequal(size(J)[2], space_size(2))
        @test all(isequal(J.q.data[i], wavelet) for i=1:nsrc)
        @test test_transpose(J)

        # get index
        J_sub = J[1]
        @test isequal(J_sub.model, J.model)
        @test isequal(J_sub.F, J.F[1])
        @test isequal(J_sub.q, q[1])

        inds = nsrc > 1 ? (1:nsrc) : 1
        J_sub = J[inds]
        @test isequal(J_sub.model, J.model)
        @test isequal(J_sub.F, J.F[inds])
        @test isequal(J_sub.q, q[inds])

        inds = nsrc > 1 ? (1:nsrc) : 1
        J_sub = J[inds]
        @test isequal(J_sub.model, J.model)
        @test isequal(size(J_sub), size(J))
    end
end

######################################################## judiJacobian ##################################################
@testset "judiJacobianExtended Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiJacobianExtended nsrc=$(nsrc)" begin
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        wavelet = randn(Float32, rec_geometry.nt[1])
        # Setup operators
        Pr = judiProjection(rec_geometry)
        F = judiModeling(model)
        Pw = judiLRWF(rec_geometry.dt, wavelet)
        w = judiWeights(randn(Float32, model.n); nsrc=nsrc)
        J = judiJacobian(Pr*F*Pw', w)


        rec_geometry = example_rec_geometry(nsrc=nsrc)
        wavelet = randn(Float32, rec_geometry.nt[1])
        # Setup operators
        Pr = judiProjection(rec_geometry)
        F = judiModeling(model)
        Pw = judiLRWF(nsrc, rec_geometry.dt[1], wavelet)
        w = judiWeights(randn(Float32, model.n); nsrc=nsrc)
        PDE = Pr*F*Pw'
        J = judiJacobian(PDE, w)

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
        @test isapprox(J_sub.q.weights[1], J.q.weights[1])

        inds = 1:nsrc
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
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        P = judiProjection(rec_geometry)

        @test isequal(typeof(P), judiProjection{Float32})
        @test isequal(P.geometry, rec_geometry)

        Pr_sub = P[1]
        @test isequal(get_nsrc(Pr_sub.geometry), 1)
        @test get_nsrc(Pr_sub.geometry) == 1

        inds = nsrc > 1 ? (1:nsrc) : 1
        Pr_sub = P[inds]
        @test isequal(Pr_sub.geometry, rec_geometry[inds])
        @test get_nsrc(Pr_sub.geometry) == nsrc
    end
end

@testset "judiLRWF Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiLRWF nsrc=$(nsrc)" begin
        src_geometry = example_src_geometry(nsrc=nsrc)
        wavelet = randn(Float32, src_geometry.nt[1])

        P = judiLRWF(src_geometry.dt, wavelet)
        @test isequal(typeof(P), judiLRWF{Float32})

        for i=1:nsrc
            @test isequal(P.wavelet[i], wavelet)
            @test isequal(P.dt[i], src_geometry.dt[i])
        end
        @test length(P.wavelet) == nsrc

        P_sub = P[1]
        @test isequal(P_sub.wavelet[1], wavelet)
        @test isequal(P_sub.dt[1], src_geometry.dt[1])
        @test length(P_sub.wavelet) == 1

        inds = nsrc > 1 ? (1:nsrc) : 1
        Pr_sub = P[inds]
        for i=1:nsrc
            @test isequal(Pr_sub.wavelet[i], wavelet)
            @test isequal(Pr_sub.dt[i], src_geometry.dt[i])
        end
        @test length(Pr_sub.wavelet) == nsrc
    end
end
