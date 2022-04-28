using JLD2

@testset "Test issue #83" begin
    @timeit TIMEROUTPUT "Issue 83" begin
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
end

@testset "Test backward comp issue" begin
    @timeit TIMEROUTPUT "Backward compatibility" begin
        datapath = joinpath(dirname(pathof(JUDI)))*"/../data/"

        # Load file with old judiVector type and julia <1.7 StepRangeLen
        @load "$(datapath)backward_comp.jld" dat model_true

        @test typeof(dat) == judiVector{Float32, Matrix{Float32}}
        @test typeof(dat.geometry) == GeometryIC{Float32}
        @test typeof(dat.geometry.xloc) == Vector{Vector{Float32}}
        @test typeof(model_true) == Model
        @test typeof(model_true.m) == PhysicalParameter{Float32}
    end
end

@testset "Test issue #113" begin
    @timeit TIMEROUTPUT "Issue 113" begin
        # we are going to check 'limit_model_to_receiver_area' function for 3D with non-zero origin
        model = example_model(n=(100,100,100), d=(10f0,10f0,10f0), o=(100f0,100f0,0f0))

        # Set up 3D receiver geometry by defining one receiver vector in each x and y direction
        nxrec = 5
        nyrec = 3
        xrec = range(300f0, stop=700f0, length=nxrec)
        yrec = range(400f0, stop=600f0, length=nyrec)
        zrec = 50f0

        # Construct 3D grid from basis vectors
        (xrec, yrec, zrec) = setup_3D_grid(xrec, yrec, zrec)

        # Set up receiver structure
        recGeometry = Geometry(xrec, yrec, zrec; dt=2f0, t=100f0, nsrc=1)

        # Set up source geometry (cell array with source locations for each shot)
        xsrc = convertToCell([500f0])
        ysrc = convertToCell([500f0])
        zsrc = convertToCell([0f0])

        # Set up source structure
        srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=2f0, t=100f0)

        buffer = 100f0
        model_limited, _ = limit_model_to_receiver_area(srcGeometry, recGeometry, model, buffer)

        # Check results
        # 'model' and 'model_limited' refer to the same reference (address)
        # one can check it with 'repr(UInt64(pointer_from_objref(model)))'

        # min/max X
        @test isequal(model_limited.o[1], xrec[1]-buffer)
        @test isequal(model_limited.o[1] + model_limited.d[1]*(model_limited.n[1]-1), xrec[end]+buffer)
        # min/max Y
        @test isequal(model_limited.o[2], yrec[1]-buffer)
        @test isequal(model_limited.o[2] + model_limited.d[2]*(model_limited.n[2]-1), yrec[end]+buffer)
    end
end
