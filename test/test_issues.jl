

@testset "Test issue #83" begin
    nxrec = 120
    nyrec = 100
    xrec = range(50f0, stop=1150f0, length=nxrec)
    yrec = range(100f0, stop=900f0, length=nyrec)
    zrec = 50f0

    # Construct 3D grid from basis vectors
    (x3d, y3d, z3d) = setup_3D_grid(xrec, yrec, zrec)

    for k=1:nyrec
        s = (k - 1) * nxrec + 1
        e = s + nxrec - 1
        @test all(x3d[k:nxrec:end] .== xrec[k])
        @test all(y3d[s:e] .== yrec[k])
    end
    @test all(z3d .== zrec)
end