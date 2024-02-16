# Adjoint test for F and J
# Author: Mathias Louboutin, mlouboutin3@gatech.edu
# Date: May 2020
#

using Distributed

# # Set parallel if specified
if nw > 1 && nworkers() < nw
    addprocs(nw-nworkers() + 1; exeflags=["--code-coverage=user", "--inline=no", "--check-bounds=yes"])
end

@everywhere using JUDI, LinearAlgebra, Test, Distributed

### Model
model, model0, dm = setup_model(tti, viscoacoustic, nlayer)
q, srcGeometry, recGeometry, f0 = setup_geom(model; nsrc=nw)
dt = srcGeometry.dt[1]

# testing parameters and utils
tol = 5f-4
(tti && fs) && (tol = 5f-3)
maxtry = viscoacoustic ? 5 : 3

#################################################################################################
# adjoint test utility function so that can retry if fails

function run_adjoint(F, q, y, dm; test_F=true, test_J=true)
    adj_F, adj_J = !test_F, !test_J
    if test_F
        # Forward-adjoint
        d_hat = F*q
        q_hat = F'*y

        # Result F
        a = dot(y, d_hat)
        b = dot(q, q_hat)
        @printf(" <F x, y> : %2.5e, <x, F' y> : %2.5e, relative error : %2.5e \n", a, b, (a - b)/(a + b))
        adj_F = isapprox(a/(a+b), b/(a+b), atol=tol, rtol=0)
    end

    if test_J
        # Linearized modeling
        J = judiJacobian(F, q)
        ld_hat = J*dm
        dm_hat = J'*y

        c = dot(ld_hat, y)
        d = dot(dm_hat, dm)
        @printf(" <J x, y> : %2.5e, <x, J' y> : %2.5e, relative error : %2.5e \n", c, d, (c - d)/(c + d))
        adj_J = isapprox(c/(c+d), d/(c+d), atol=tol, rtol=0)
    end
    return adj_F, adj_J
end

test_adjoint(f::Bool, j::Bool, last::Bool) = (test_adjoint(f, last), test_adjoint(j, last))
test_adjoint(adj::Bool, last::Bool) = (adj || last) ? (@test adj) : (@test_skip adj)

###################################################################################################
# Modeling operators
@testset "Adjoint test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
    @timeit TIMEROUTPUT "Adjoint" begin
        opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], f0=f0)
        F = judiModeling(model0, srcGeometry, recGeometry; options=opt)

        # Nonlinear modeling
        y = F*q

        # Run test until succeeds in case of bad case
        adj_F, adj_J = false, false
        ntry = 0
        while (!adj_F || !adj_J) && ntry < maxtry
            wave_rand = (.5f0 .+ rand(Float32, size(q.data[1]))).*q.data[1]
            q_rand = judiVector(srcGeometry, wave_rand)

            adj_F, adj_J = run_adjoint(F, q_rand, y, dm; test_F=!adj_F, test_J=!adj_J)
            ntry +=1
            test_adjoint(adj_F, adj_J, ntry==maxtry)
        end
        println("Adjoint test after $(ntry) tries, F: $(success_log[adj_F]), J: $(success_log[adj_J])")
    end
end
###################################################################################################
# Extended source modeling
@testset "Extended source adjoint test with $(nlayer) layers and tti $(tti) and viscoacoustic $(viscoacoustic) and freesurface $(fs)" begin
    @timeit TIMEROUTPUT "Extended source adjoint" begin
        opt = Options(sum_padding=true, dt_comp=dt, free_surface=parsed_args["fs"], f0=f0)
        F = judiModeling(model0, srcGeometry, recGeometry; options=opt)
        Pr = judiProjection(recGeometry)
        Fw = judiModeling(model0; options=opt)
        Pw = judiLRWF(srcGeometry.dt[1], circshift(q.data[1], 51))
        Fw = Pr*Fw*adjoint(Pw)

        # Extended source weights
        w = zeros(Float32, model0.n...)
        w[141:160, 65:84] .= randn(Float32, 20, 20)
        w = judiWeights(w; nsrc=nw)

        # Nonlinear modeling
        y = F*q

        # Run test until succeeds in case of bad case
        adj_F, adj_J = false, false
        ntry = 0
        while (!adj_F || !adj_J) && ntry < maxtry
            wave_rand = (.5f0 .+ rand(Float32, size(q.data[1]))).*q.data[1]
            q_rand = judiVector(srcGeometry, wave_rand)

            adj_F, adj_J = run_adjoint(F, q_rand, y, dm; test_F=!adj_F, test_J=!adj_J)
            ntry +=1
            test_adjoint(adj_F, adj_J, ntry==maxtry)
        end
        println("Adjoint test after $(ntry) tries, F: $(success_log[adj_F]), J: $(success_log[adj_J])")
    end
end
