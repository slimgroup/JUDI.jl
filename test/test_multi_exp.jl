# 2D LSRTM/FWI parallelization test with 2 sources and 2 vintages
#
# Ziyi Yin, ziyi.yin@gatech.edu
# August 2021

@testset "test reduction from distributed tasks" begin
    function judiWeightsConst(i::Int)
        return judiWeights(i*ones(Float32,10*i,10*i))
    end
    f = Vector{Future}(undef, 6)
    for i = 1:6
        f[i] = @spawn judiWeightsConst(i)
    end
    using JUDI:reduce!
    w = reduce!(f)
    @info "test if stacking follows the correct order through reduction"
    for i = 1:6
        @test w.data[i] == i*ones(Float32,10*i,10*i)
    end

    # Test objective function
    f1 = @spawn (Ref{Float32}(1f0), randn(10))
    f2 = @spawn (Ref{Float32}(2f0), randn(10))
    JUDI.local_reduce!(f1, f2)
    res = fetch(f1)

    @test res[1][] == 3f0
    @test typeof(res[1]) == Base.RefValue{Float32}

    res = JUDI.as_vec(res, Val(false))
    @test res[1] == 3f0
    @test typeof(res[1]) == Float32
end

parsed_args = parse_commandline()

nlayer = parsed_args["nlayer"]
tti = parsed_args["tti"]
fs =  parsed_args["fs"]

### Model
model, model0, dm = setup_model(tti, 4)
q, srcGeometry, recGeometry, info = setup_geom(model; nsrc=4)
q1 = q[[1,4]]
q2 = q[[2,3]]
srcGeometry1 = subsample(srcGeometry,[1,4])
srcGeometry2 = subsample(srcGeometry,[2,3])
recGeometry1 = subsample(recGeometry,[1,4])
recGeometry2 = subsample(recGeometry,[2,3])
info1 = subsample(info,[1,4])
info2 = subsample(info,[2,3])

opt = Options(sum_padding=true, free_surface=fs)
F1 = judiModeling(info1, model, srcGeometry1, recGeometry1; options=opt)
F2 = judiModeling(info2, model, srcGeometry2, recGeometry2; options=opt)

# Observed data
dobs1 = F1*q1
dobs2 = F2*q2

# Perturbations
dm1 = 2f0*circshift(dm, 10)
dm2 = 2f0*circshift(dm, 30)


@testset "Multi-experiment arg processing" begin
    for (nm, m) in zip([1, 2], [model0, [model0, model0]])
        for (nq, q) in zip([1, 2], [q1, [q1, q2]])
            for (nd, d) in zip([1, 2], [dobs1, [dobs1, dobs2]])
                for dmloc in [dm1, dm1[:]]
                    for (ndm, dm) in zip([1, 2], [dmloc, [dmloc, dmloc]])
                        args = m, q, d, dm
                        @test JUDI.get_nexp(args...) == maximum((nm, nq, nd, ndm))
                        @test all(JUDI.get_exp(1, args...) .== (model0, q1, dobs1, dmloc))
                    end
                end
            end
        end
    end

    @test_throws ArgumentError JUDI.get_nexp([model0, model0], [dobs1, dobs2, dobs1])
    @test_throws ArgumentError JUDI.get_nexp([model0, model0], [dobs1, dobs2], [dm1, dm2, dm1])
    @test_throws ArgumentError JUDI.get_nexp(model0, [dobs1, dobs2], [dm1, dm2, dm1])
end


@testset "FWI/LSRTM objective multi-level parallelization test with $(nlayer) layers and tti $(tti) and freesurface $(fs)" begin
    
    ftol = 1f-5

    Jm0, grad = lsrtm_objective([model0, model0], [q1, q2], [dobs1, dobs2], [dm1, dm2]; options=opt, nlind=true)
    Jm01, grad1 = lsrtm_objective(model0, q1, dobs1, dm1; options=opt, nlind=true)
    Jm02, grad2 = lsrtm_objective(model0, q2, dobs2, dm2; options=opt, nlind=true)
    @test isapprox(Jm01+Jm02, Jm0; rtol=ftol)
    @test isapprox(grad1, grad[1]; rtol=ftol)
    @test isapprox(grad2, grad[2]; rtol=ftol)

    _Jm0, _grad = fwi_objective([model0, model0], [q1, q2], [dobs1, dobs2]; options=opt)
    _Jm01, _grad1 = fwi_objective(model0, q1, dobs1; options=opt)
    _Jm02, _grad2 = fwi_objective(model0, q2, dobs2; options=opt)
    @test isapprox(_Jm01+_Jm02, _Jm0; rtol=ftol)
    @test isapprox(_grad1, _grad[1]; rtol=ftol)
    @test isapprox(_grad2, _grad[2]; rtol=ftol)

end
