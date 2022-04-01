# Test backward compatibility

### Model
nsrc = 2
model, model0, dm = setup_model(tti, nlayer)
q, srcGeometry, recGeometry = setup_geom(model; nsrc=nsrc)
dt = srcGeometry.dt[1]
nt = srcGeometry.nt[1]
nrec = length(recGeometry.xloc[1])

@testset "Backward compatibility" begin
    @timeit TIMEROUTPUT "Backward compatibility" begin
        info =  Info(prod(model.n), nt, nsrc)
        @test_logs (:warn, "Info is deprecated and will be removed in future versions") Info(prod(model.n), nt, nsrc)
        @test_logs (:warn, "judiModeling(info::Info, ar...; kw...) is deprecated, use judiModeling(ar...; kw...)") judiModeling(info, model)
        @test_logs (:warn, "judiProjection(info::Info, ar...; kw...) is deprecated, use judiProjection(ar...; kw...)") judiProjection(info, recGeometry)
        @test_logs (:warn, "judiWavefield(info::Info, ar...; kw...) is deprecated, use judiWavefield(ar...; kw...)") judiWavefield(info, dt, nsrc, randn(nt, model.n...))
    
        @test_throws ArgumentError judiRHS(info, recGeometry, randn(Float32, nt, nrec))
        @test_throws ArgumentError judiLRWF(info, nsrc, randn(nt))

    end
end