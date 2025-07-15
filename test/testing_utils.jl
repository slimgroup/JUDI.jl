mean(x) = sum(x)/length(x)

function grad_test(misfit, x0, dx, g; maxiter=6, h0=5f-2, data=false, stol=1f-1)
    # init
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)
    
    gdx = data ? g : dot(g, dx)
    f0 = misfit(x0)
    h = h0

    @printf("%11.5s, %11.5s, %11.5s, %11.5s, %11.5s, %11.5s \n", "h", "gdx", "e1", "e2", "rate1", "rate2")
    for j=1:maxiter
        f = misfit(x0 + h*dx)
        err1[j] = norm(f - f0, 1)
        err2[j] = norm(f - f0 - h*gdx, 1)
        j == 1 ? prev = 1 : prev = j - 1
        @printf("%5.5e, %5.5e, %5.5e, %5.5e, %5.5e, %5.5e \n", h, h*norm(gdx, 1), err1[j], err2[j], err1[prev]/err1[j], err2[prev]/err2[j])
        h = h * .8f0
    end

    rate1 = err1[1:end-1]./err1[2:end]
    rate2 = err2[1:end-1]./err2[2:end]
    @test isapprox(mean(rate1), 1.25f0; atol=stol)
    @test isapprox(mean(rate2), 1.5625f0; atol=stol)
end

@testset "Testing devito config" begin
    lg = get(ENV, "DEVITO_LANGUAGE", nothing)
    arch = get(ENV, "DEVITO_ARCH", nothing)
    plt = get(ENV, "DEVITO_PLATFORM", nothing)
    pop!(ENV, "DEVITO_LANGUAGE", nothing)
    pop!(ENV, "DEVITO_ARCH", nothing)
    pop!(ENV, "DEVITO_PLATFORM", nothing)
    default_devito_config()
    if Sys.isapple()
        @test string(JUDI.devito.configuration["language"]) == "openmp"
        @test string(JUDI.devito.configuration["compiler"].name) == "clang"
        if Sys.ARCH == :aarch64
            @test string(JUDI.devito.configuration["platform"].name) == "m1"
        else
            @test Bool(JUDI.PythonCall.pybuiltins.isinstance(JUDI.devito.configuration["platform"], JUDI.devito.Cpu64))
        end
    else
        @test string(JUDI.devito.configuration["language"]) == "openmp"
        @test Bool(JUDI.PythonCall.pybuiltins.isinstance(JUDI.devito.configuration["platform"], JUDI.devito.Cpu64))
        @test string(JUDI.devito.configuration["compiler"].name) == "gcc"
    end
    !isnothing(lg) && (ENV["DEVITO_LANGUAGE"] = lg)
    !isnothing(arch) && (ENV["DEVITO_ARCH"] = arch)
    !isnothing(plt) && (ENV["DEVITO_PLATFORM"] = plt)
end