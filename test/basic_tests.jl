# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#

using JUDI.TimeModeling, PyCall, LinearAlgebra, PyPlot, Test

ftol = 1f-6

function test_model(ndim; tti=false)
    n = Tuple(121 for i=1:ndim)
    o = Tuple(0. for i=1:ndim)
    d = Tuple(10. for i=1:ndim)
    m = rand(Float32, n...)
    if !tti
        return model = Model(n, d, o, m; nb=10)
    else
        epsilon = .1f0 * m
        delta = .1f0 * m
        theta = .1f0 * m
        phi = .1f0 * m
        return model = Model(n, d, o, m; nb=10, epsilon=epsilon, delta=delta, theta=theta, phi=phi)
    end
end

# Test padding and its adjoint
function test_padding(ndim)

    model = test_model(ndim; tti=false)
    modelPy = devito_model(model, Options())

    v0 = sqrt.(1f0 ./model.m)
    cut = remove_padding(deepcopy(modelPy.vp.data), modelPy.padsizes; true_adjoint=true)
    vpdata = deepcopy(modelPy.vp.data)

    @show dot(v0, cut)/dot(vpdata, vpdata)
    @test isapprox(dot(v0, cut), dot(vpdata, vpdata))
end

function test_limit_m(ndim, tti)
    model = test_model(ndim; tti=tti)
    srcGeometry = example_src_geometry()
    recGeometry = example_rec_geometry(cut=true)
    dm = rand(Float32, model.n...)
    new_mod, dm_n = limit_model_to_receiver_area(srcGeometry, recGeometry, deepcopy(model), 100f0; pert=dm)

    # check inds
    inds = ndim == 3 ? [6:116, 1:11, 1:121] : [6:116, 1:121]
    @test new_mod.n[1] == 111
    @test new_mod.n[end] == 121
    if ndim == 3
        @test new_mod.n[2] == 11
    end

    @test isapprox(new_mod.m, model.m[inds...]; rtol=ftol)
    @test isapprox(dm_n, vec(dm[inds...]); rtol=ftol)
    if tti
        @test isapprox(new_mod.epsilon, model.epsilon[inds...]; rtol=ftol)
        @test isapprox(new_mod.delta, model.delta[inds...]; rtol=ftol)
        @test isapprox(new_mod.theta, model.theta[inds...]; rtol=ftol)
        @test isapprox(new_mod.phi, model.phi[inds...]; rtol=ftol)
    end

    ex_dm = extend_gradient(model, new_mod, dm_n)
    @test size(ex_dm) == size(dm)

end

@testset "Test basic utilities" begin
    for ndim=[2, 3]
        test_padding(ndim)
    end
    opt = Options(frequencies=[[2.5, 4.5], [3.5, 5.5]])
    @test subsample(opt, 1).frequencies[1] == [2.5, 4.5]
    @test subsample(opt, 2).frequencies[1] == [3.5, 5.5]

    for ndim=[2, 3]
        for tti=[true, false]
            test_limit_m(ndim, tti)
        end
    end
end