function grad_test(misfit, x0, dx, g; maxiter=6, h0=1f-1)
    # init
    err1 = zeros(Float32, maxiter)
    err2 = zeros(Float32, maxiter)
    
    gdx = dot(g, dx)
    f0 = misfit(x0)
    h = h0

    @printf("%11.5s, %11.5s, %11.5s, %11.5s, %11.5s, %11.5s \n", "h", "gdx", "e1", "e2", "rate1", "rate2")
    for j=1:maxiter
        f = misfit(x0 + h*dx)
        err1[j] = abs(f - f0)
        err2[j] = abs(f - f0 - h*gdx)
        j == 1 ? prev = 1 : prev = j - 1
        @printf("%5.5e, %5.5e, %5.5e, %5.5e, %5.5e, %5.5e \n", h, h*abs(gdx), err1[j], err2[j], err1[prev]/err1[j], err2[prev]/err2[j])
        h = h * .8f0
    end

    rate1 = err1[1:end-1]./err1[2:end]
    rate2 = err2[1:end-1]./err2[2:end]
    @show mean(rate1), mean(rate2)
    # @test isapprox(mean(rate1), 2f0; atol=stol)
    # @test isapprox(mean(rate2), 4f0; atol=stol)
end