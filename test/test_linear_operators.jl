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
end

sub_dim(v::Vector{<:Integer}, i::Integer) = v[i:i]
sub_dim(v::Vector{<:Integer}, i::AbstractRange) = v[i]
sub_dim(v::Integer, i) = v

check_dims(v::Integer, vo::Integer) = v == vo
check_dims(v::Vector{<:Integer}, vo::Vector{<:Integer}) = v == vo
check_dims(v::Integer, vo::Vector{<:Integer}) = (length(vo) == 1 && v == vo[1])
check_dims(v::Vector{<:Integer}, vo::Integer) = (length(v) == 1 && v[1] == vo)

function test_getindex(Op, nsrc)
    so = size(Op)

    Op_sub = Op[1]
    @test isequal(Op_sub.model, Op.model)
    # Check sizes. Same dimensions but subsampled source
    @test issetequal(keys(size(Op_sub)), keys(size(Op)))
    s = size(Op_sub)
    for i=1:2
        for (k, v) ∈ s[i]
            vo = k == :src ? 1 : so[i][k]
            @test check_dims(v, sub_dim(vo, 1))
        end
    end

    inds = 1:nsrc
    Op_sub = Op[inds]
    @test isequal(Op_sub.model, Op.model)
    @test isequal(keys(size(Op_sub)), keys(size(Op)))
    s = size(Op_sub)
    for i=1:2
        for (k, v) ∈ s[i]
            vo = k == :src ? nsrc : so[i][k]
            @test check_dims(v, sub_dim(vo, inds))
        end
    end
end

model = example_model()
########################################################## judiModeling ###############################################
@testset "judiModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiModeling nsrc=$(nsrc)" begin
        F_forward = judiModeling(model; options=Options())
        F_adjoint = adjoint(F_forward)

        test_transpose(F_forward)
        @test isequal(size(F_forward)[1], time_space(model.n))
        @test isequal(size(F_forward)[2], time_space(model.n))
        @test isequal(size(F_adjoint)[1], time_space(model.n))
        @test isequal(size(F_adjoint)[2], time_space(model.n))
        @test issetequal(keys(size(F_forward)[1]), [:time, :x, :z])
        @test issetequal(keys(size(F_forward)[2]), [:time, :x, :z])
        # Time is uninitialized until first multiplication since there is no info
        # on propagation time
        @test issetequal(values(size(F_forward)[1]), [[0], model.n...])
        @test Int(size(F_forward)[1]) == 0
        # Update size for check
        nt = 10
        size(F_forward)[1][:time] = [nt for i=1:nsrc]
        @test Int(size(F_forward)[1]) == nsrc * nt * prod(model.n)

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiModeling{Float32, :adjoint})

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

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

        test_transpose(F_forward)
        @test isequal(size(F_forward)[1], time_space_src(nsrc, src_geometry.nt, model.n))
        @test isequal(size(F_forward)[2], rec_space(src_geometry))
        @test isequal(size(F_adjoint)[1], rec_space(src_geometry))
        @test isequal(size(F_adjoint)[2], time_space_src(nsrc, src_geometry.nt, model.n))
        @test issetequal(keys(size(F_forward)[1]), (:src, :time, :x, :z))
        @test issetequal(keys(size(F_forward)[2]), (:src, :time, :rec))
        # With the composition, everything should be initialized
        @test issetequal(values(size(F_forward)[1]), (nsrc, src_geometry.nt, model.n...))
        @test issetequal(values(size(F_forward)[2]), (nsrc, src_geometry.nt, src_geometry.nrec))
        @test Int(size(F_forward)[1]) == prod(model.n) * sum(src_geometry.nt)
        @test Int(size(F_forward)[2]) == sum(src_geometry.nrec .* src_geometry.nt)

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiPointSourceModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiDataModeling{Float32, :adjoint})

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

        if VERSION>v"1.2"
            a = randn(Float32, model.n...)
            F2 = F_forward(;m=a)
            @test isapprox(F2.model.m, a)
            F2 = F_forward(Model(model.n, model.d, model.o, a))
            @test isapprox(F2.model.m, a)
            @test F2.model.n == model.n
        end

        # SimSources
        if nsrc == 2
            M = randn(Float32, 1, nsrc)
            Fs = M*F_forward
            @test isequal(size(Fs)[1], time_space_src(1, src_geometry[1].nt, model.n))
            @test isequal(size(Fs)[2], rec_space(example_rec_geometry(;nrec=2, nsrc=1)))
            @test get_nsrc(Fs.qInjection) == 1
            Fs = M*F_adjoint
            @test isequal(size(Fs)[1], rec_space(example_rec_geometry(;nrec=2, nsrc=1)))
            @test isequal(size(Fs)[2], time_space_src(1, src_geometry[1].nt, model.n))
            @test get_nsrc(Fs.rInterpolation) == 1
        end
    end
end

@testset "judiDataModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiDataModeling nsrc=$(nsrc)" begin
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        F_forward = judiProjection(rec_geometry)*judiModeling(model; options=Options())
        F_adjoint = adjoint(F_forward)

        test_transpose(F_forward)
        @test isequal(size(F_forward)[1], rec_space(rec_geometry))
        @test isequal(size(F_forward)[2], time_space_src(nsrc, rec_geometry.nt, model.n))
        @test isequal(size(F_adjoint)[1], time_space_src(nsrc, rec_geometry.nt, model.n))
        # With the composition, everything should be initialized
        @test issetequal(values(size(F_forward)[2]), (nsrc, rec_geometry.nt, model.n...))
        @test issetequal(values(size(F_forward)[1]), (nsrc, rec_geometry.nt, rec_geometry.nrec))
        @test Int(size(F_forward)[2]) == prod(model.n) * sum(rec_geometry.nt)
        @test Int(size(F_forward)[1]) == sum(rec_geometry.nrec .* rec_geometry.nt)

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiDataModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiPointSourceModeling{Float32, :adjoint})

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

        if VERSION>v"1.2"
            a = randn(Float32, model.n...)
            F2 = F_forward(;m=a)
            @test isapprox(F2.model.m, a)
            F2 = F_forward(Model(model.n, model.d, model.o, a))
            @test isapprox(F2.model.m, a)
            @test F2.model.n == model.n
        end

        # SimSources
        if nsrc == 2
            M = randn(Float32, 1, nsrc)
            Fs = M*F_adjoint
            @test isequal(size(Fs)[1], time_space_src(1, rec_geometry[1].nt, model.n))
            @test isequal(size(Fs)[2], rec_space(rec_geometry[1]))
            @test get_nsrc(Fs.qInjection) == 1
            Fs = M*F_forward
            @test isequal(size(Fs)[1], rec_space(rec_geometry[1]))
            @test isequal(size(Fs)[2], time_space_src(1, rec_geometry[1].nt, model.n))
            @test get_nsrc(Fs.rInterpolation) == 1
        end
    end
end

@testset "judiDataSourceModeling Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiDataSourceModeling nsrc=$(nsrc)" begin
        rec_geometry = example_rec_geometry(nsrc=nsrc)
        src_geometry = example_src_geometry(nsrc=nsrc)
        F_forward = judiModeling(model, src_geometry, rec_geometry; options=Options())
        F_adjoint = adjoint(F_forward)

        test_transpose(F_forward)
        @test isequal(size(F_forward)[2], rec_space(src_geometry))
        @test isequal(size(F_forward)[1], rec_space(rec_geometry))
        @test isequal(size(F_adjoint)[2], rec_space(rec_geometry))
        @test isequal(size(F_adjoint)[1], rec_space(src_geometry))
        @test issetequal(keys(size(F_forward)[1]), (:src, :time, :rec))
        @test issetequal(keys(size(F_forward)[2]), (:src, :time, :rec))
        # With the composition, everything should be initialized
        @test issetequal(values(size(F_forward)[2]), (nsrc, src_geometry.nt, src_geometry.nrec))
        @test issetequal(values(size(F_forward)[1]), (nsrc, rec_geometry.nt, rec_geometry.nrec))
        @test Int(size(F_forward)[2]) == sum(src_geometry.nrec .* src_geometry.nt)
        @test Int(size(F_forward)[1]) == sum(rec_geometry.nrec .* rec_geometry.nt)

        @test adjoint(F_adjoint) == F_forward
        @test isequal(typeof(F_forward), judiDataSourceModeling{Float32, :forward})
        @test isequal(typeof(F_adjoint), judiDataSourceModeling{Float32, :adjoint})

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

        # get index
        test_getindex(F_forward, nsrc)
        test_getindex(F_adjoint, nsrc)

        if VERSION>v"1.2"
            a = randn(Float32, model.n...)
            F2 = F_forward(;m=a)
            @test isapprox(F2.model.m, a)
            F2 = F_forward(Model(model.n, model.d, model.o, a))
            @test isapprox(F2.model.m, a)
            @test F2.model.n == model.n
        end

        # SimSources
        if nsrc == 2
            M = randn(Float32, 1, nsrc)
            Fs = M*F_forward
            @test isequal(size(Fs)[1], rec_space(rec_geometry[1]))
            @test isequal(size(Fs)[2], rec_space(example_rec_geometry(;nrec=2, nsrc=1)))
            @test get_nsrc(Fs.qInjection) == 1
            Fs = M*F_adjoint
            @test isequal(size(Fs)[1], rec_space(example_rec_geometry(;nrec=2, nsrc=1)))
            @test isequal(size(Fs)[2], rec_space(rec_geometry[1]))
            @test get_nsrc(Fs.rInterpolation) == 1
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
        # Check sizes
        @test isequal(size(J)[1], rec_space(rec_geometry))
        @test isequal(size(J)[2], space(model.n))
        @test issetequal(keys(size(J)[2]), (:x, :z))
        @test Int(size(J)[2]) == prod(model.n)
        @test Int(size(J)[1]) == sum(rec_geometry.nrec .* rec_geometry.nt)

        @test isequal(typeof(J), judiJacobian{Float32, :born, typeof(PDE)})
        @test isequal(J.F.rInterpolation.geometry, rec_geometry)
        @test isequal(J.F.qInjection.geometry, src_geometry)
        @test isequal(J.q.geometry, src_geometry)
        @test all(isequal(J.q.data[i], wavelet) for i=1:nsrc)
        test_transpose(J)

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

        # SimSources
        if nsrc == 2
            M = randn(Float32, 1, nsrc)
            Fs = M*J
            @test isequal(size(Fs)[1], rec_space(rec_geometry[1]))
            @test isequal(size(Fs)[2], space(model.n))
            @test Fs.q.nsrc == 1
        end
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
        # Check sizes
        @test isequal(size(J)[1], rec_space(rec_geometry))
        @test isequal(size(J)[2], space(model.n))
        @test issetequal(keys(size(J)[2]), (:x, :z))
        @test Int(size(J)[2]) == prod(model.n)
        @test Int(size(J)[1]) == sum(rec_geometry.nrec .* rec_geometry.nt)

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
        @test isequal(size(J)[2], space(model.n))
        test_transpose(J)

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
        # Check sizes and that's it's unitialized since not combined with any propagator
        @test size(P)[2] == time_space_src(get_nsrc(rec_geometry), rec_geometry.nt, 3)
        @test size(P)[1] == rec_space(rec_geometry)
        # Size is zero since un-initialized
        @test Int(size(P)[2]) == 0 
        @test Int(size(P)[1]) ==  sum(rec_geometry.nrec .* rec_geometry.nt)

        @test isequal(typeof(P), judiProjection{Float32})
        @test isequal(P.geometry, rec_geometry)

        Pr_sub = P[1]
        @test isequal(get_nsrc(Pr_sub.geometry), 1)
        @test get_nsrc(Pr_sub.geometry) == 1

        inds = nsrc > 1 ? (1:nsrc) : 1
        Pr_sub = P[inds]
        @test isequal(Pr_sub.geometry, rec_geometry[inds])
        @test get_nsrc(Pr_sub.geometry) == nsrc
        
        # SimSources
        if nsrc == 2
            M = randn(Float32, 1, nsrc)
            Fs = M*P
            @test size(Fs)[2] == time_space_src(get_nsrc(rec_geometry[1]), rec_geometry[1].nt, 3)
            @test size(Fs)[1] == rec_space(rec_geometry[1])
            @test get_nsrc(Fs.geometry) == 1
        end
    end
end

@testset "judiLRWF Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "judiLRWF nsrc=$(nsrc)" begin
        src_geometry = example_src_geometry(nsrc=nsrc)
        wavelet = randn(Float32, src_geometry.nt[1])

        P = judiLRWF(src_geometry.dt, wavelet)
        @test isequal(typeof(P), judiLRWF{Float32})
        # Check sizes and that's it's unitialized since not combined with any propagator
        @test size(P)[1] == space_src(nsrc)
        @test size(P)[2] == time_space_src(nsrc, src_geometry.nt)
        # Size is zero since un-initialized
        @test Int(size(P)[1]) == 0
        @test Int(size(P)[2]) == 0

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
