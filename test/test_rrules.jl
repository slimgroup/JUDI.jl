# Gradient tests for adjoint extended modeling
# Author: Mathias Louboutin, mathias.louboutin@gmail.com
# April 2022
using Flux

### Model
nsrc = 1
model, model0, dm = setup_model(tti, viscoacoustic, 4)
m, m0 = model.m.data, model0.m.data
q, srcGeometry, recGeometry, f0 = setup_geom(model; nsrc=nsrc)
dt = srcGeometry.dt[1]

# Common op
Pr = judiProjection(recGeometry)
Ps = judiProjection(srcGeometry)
Pw = judiLRWF(dt, q.data[1])

function GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, n)
	weights = zeros(Float32, n[1], n[2], 1, nsrc)
	for j=1:nsrc
        weights[xsrc_index[j], zsrc_index[j], 1, j] = 1f0
    end
    return weights
end

perturb(x::Vector{T}) where T = circshift(x, rand(1:20))
perturb(x::Array{T, N}) where {T, N} = circshift(x, (rand(1:20), zeros(N-1)...))
perturb(x::judiVector) = judiVector(x.geometry, perturb(.5f0.*x.data))
mean(x) = sum(x)/length(x)

misfit_objective(d_obs, q0, m0, F) = .5f0*norm(F(m0, q0) - d_obs)^2

function loss(d_obs, q0, m0, F)
    local ϕ
    g = gradient(Flux.params(q0, m0)) do
        ϕ = misfit_objective(d_obs, q0, m0, F)
        dt = F.options.return_array ? get_dt(F.model) : 1f0
        return dt*ϕ
    end
    return ϕ, g[q0], g[m0]
end

xsrc_index, zsrc_index = rand(1:model.n[1], nsrc), rand(1:model.n[2], nsrc)
w = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);

sinput = zip(["Point", "Extended"], [Ps, Pw], (q, w))
#####################################################################################
ftol = sqrt(eps(1f0))

@testset "AD correctness check return_array=$(ra)" for ra in []#[true, false]
    opt = Options(return_array=ra, sum_padding=true, f0=f0)
    A_inv = judiModeling(model; options=opt)
    A_inv0 = judiModeling(model0; options=opt)
    @testset "AD correctness check source type: $(stype)" for (stype, Pq, q) in sinput
        @timeit TIMEROUTPUT "$(stype) source AD" begin
            # Linear operators
            q0 = perturb(q)
            # Operators
            F = Pr*A_inv*adjoint(Pq)
            F0 = Pr*A_inv0*adjoint(Pq)

            d_obs = F(m, q)
            # PDE accept model as input but AD expect the actual model param (model.m)
            d_obs2 = F(model, q)
            @test d_obs ≈ d_obs2 rtol=ftol

            J = judiJacobian(F0, q0)
            gradient_m = adjoint(J)*(F(m0, q0) - d_obs)
            gradient_m2 = adjoint(J)*(F(model0.m, q0) - d_obs)
            @test gradient_m ≈ gradient_m2 rtol=ftol

            # Gradient with m array
            gs_inv = gradient(x -> misfit_objective(d_obs, q0, x, F), m0)
            # Gradient with m PhysicalParameter
            gs_inv2 = gradient(x -> misfit_objective(d_obs, q0, x, F), model0.m)
            @test gs_inv[1][:] ≈ gs_inv2[1][:] rtol=ftol

            g1 = vec(gradient_m)
            g2 = vec(gs_inv[1])

            @test isapprox(norm(g1 - g2) / norm(g1 + g2), 0f0; atol=ftol)
            @test isapprox(dot(g1, g2)/norm(g1)^2,1f0;rtol=ftol)
            @test isapprox(dot(g1, g2)/norm(g2)^2,1f0;rtol=ftol)
        end
    end
end

stol = 1f-1

@testset "AD Gradient test" for ra in [false, true]
    opt = Options(return_array=ra, sum_padding=true, f0=f0)
    F = judiModeling(model; options=opt)
    ginput = zip(["Point", "Extended", "AdjExtended"], [Pr*F*Ps', Pr*F*Pw', (Ps*F*Pw')'], (q, w, q))
    @testset "Gradient test: $(stype) source" for (stype, F, q) in ginput
        @timeit TIMEROUTPUT "$(stype) source gradient" begin
            # return linearized data as Julia array
            @printf("%5.5s, %5.5s, %5.5s, %5.5s, %5.5s \n", "h", "e1", "e2", "rate1", "rate2")
            q0 = perturb(q)
            # Linear operators
            d, dq = F*q, q-q0

            #####################################################################################
        
            # Gradient test for extended modeling: weights
            f0, gq, gm = loss(d, q0, m0, F)
            h = 5f-2
            maxiter = 6

            err1 = zeros(Float32, maxiter)
            err2 = zeros(Float32, maxiter)

            print("\nGradient test source $(stype) source, $(ra)\n")
            for j=1:maxiter
                f = misfit_objective(d, q0 + h*dq, m0, F)
                err1[j] = abs(f - f0)
                err2[j] = abs(f - f0 - h*dot(dq, gq))
                j == 1 ? prev = 1 : prev = j - 1
                @printf("%5.5e, %5.5e, %5.5e, %5.5e \n", h, err1[j], err2[j], err1[prev]/err1[j], err2[prev]/err2[j])
                h = h/2f0
            end

            rate1 = err1[1:end-1]./err1[2:end]
            rate2 = err2[1:end-1]./err2[2:end]
            @show rate1, rate2
            @test isapprox(mean(rate1), 2f0; atol=stol)
            @test isapprox(mean(rate2), 4f0; atol=stol)

            # Gradient test for extended modeling: model
            h = 5f-2
            maxiter = 6

            err3 = zeros(Float32, maxiter)
            err4 = zeros(Float32, maxiter)

            print("\nGradient test model $(stype) source, $(ra)\n")
            for j=1:maxiter
                f = misfit_objective(d, q0, m0 + h*dm, F)
                err3[j] = abs(f - f0)
                err4[j] = abs(f - f0 - h*dot(dm, gm))
                j == 1 ? prev = 1 : prev = j - 1
                @printf("%5.5e, %5.5e, %5.5e, %5.5e \n", h, err3[j], err4[j], err3[prev]/err3[j], err4[prev]/err4[j])
                h = h/2f0
            end

            rate1 = err3[1:end-1]./err3[2:end]
            rate2 = err4[1:end-1]./err4[2:end]
            @show rate1, rate2
            @test isapprox(mean(rate1), 2f0; atol=stol)
            @test isapprox(mean(rate2), 4f0; atol=stol)
        end
    end
end
