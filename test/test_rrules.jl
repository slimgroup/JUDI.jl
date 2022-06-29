# Gradient tests for adjoint extended modeling
# Author: Mathias Louboutin, mathias.louboutin@gmail.com
# April 2022
using Flux
Flux.Random.seed!(2022)

### Model
nsrc = 1
dt = 1f0

model, model0, dm = setup_model(tti, viscoacoustic, 4)
m, m0 = model.m.data, model0.m.data
q, srcGeometry, recGeometry, f0 = setup_geom(model; nsrc=nsrc, dt=dt)

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

randx(x::Array{Float32}) = x .* (1 .+ randn(Float32, size(x)))
perturb(x::Vector{T}) where T = circshift(x, rand(1:20))
perturb(x::Array{T, N}) where {T, N} = circshift(x, (rand(1:20), zeros(N-1)...))
perturb(x::judiVector) = judiVector(x.geometry, [randx(x.data[i]) for i=1:x.nsrc])
reverse(x::judiVector) = judiVector(x.geometry, [x.data[i][end:-1:1, :] for i=1:x.nsrc])

misfit_objective(d_obs, q0, m0, F) = .5f0*norm(F(m0, q0) - d_obs)^2
    
function loss(d_obs, q0, m0, F)
    local ϕ
    # Reshape as ML size if returns array
    d_obs = F.options.return_array ? reshape(d_obs, F.rInterpolation, F.model; with_batch=true) : d_obs
    # Misfit and gradient
    g = gradient(Flux.params(q0, m0)) do
        ϕ = misfit_objective(d_obs, q0, m0, F)
        return ϕ
    end
    return ϕ, g[q0], g[m0]
end

xsrc_index, zsrc_index = rand(1:model.n[1], nsrc), rand(1:model.n[2], nsrc)
w = GenSimSourceMulti(xsrc_index, zsrc_index, nsrc, model.n);

sinput = zip(["Point", "Extended"], [Ps, Pw], (q, w))
#####################################################################################
ftol = sqrt(eps(1f0))

@testset "AD correctness check return_array=$(ra)" for ra in [true, false]
    opt = Options(return_array=ra, sum_padding=true, f0=f0)
    A_inv = judiModeling(model; options=opt)
    A_inv0 = judiModeling(model0; options=opt)
    @testset "AD correctness check source type: $(stype)" for (stype, Pq, q) in sinput
        @timeit TIMEROUTPUT "$(stype) source AD, array=$(ra)" begin
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

            # Reshape d_obs into ML size (nt, nrec, 1, nsrc)
            d_obs = ra ? reshape(d_obs, Pr, model; with_batch=true) : d_obs

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


@testset "AD Gradient test return_array=$(ra)" for ra in [true, false]
    opt = Options(return_array=ra, sum_padding=true, f0=f0, dt_comp=dt)
    F = judiModeling(model; options=opt)
    ginput = zip(["Point", "Extended", "Adjoint Extended"], [Pr*F*Ps', Pr*F*Pw', Pw*F'*Ps'], (q, w, reverse(q)))
    @testset "Gradient test: $(stype) source" for (stype, F, q) in ginput
        @timeit TIMEROUTPUT "$(stype) source gradient, array=$(ra)" begin
            # Initialize source for source perturbation
            q0 = perturb(q)
            # Data and source perturbation
            d, dq = F*q, q-q0

            #####################################################################################
            f0, gq, gm = loss(d, q0, m0, F)
            # Gradient test for extended modeling: source
            print("\nGradient test source $(stype) source, array=$(ra)\n")
            grad_test(x-> misfit_objective(d, x, m0, F), q0, dq, gq)
  
            # Gradient test for extended modeling: model
            print("\nGradient test model $(stype) source, array=$(ra)\n")
            grad_test(x-> misfit_objective(d, q0, x, F), m0, dm, gm)
        end
    end
end
