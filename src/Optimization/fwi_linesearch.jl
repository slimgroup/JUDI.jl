
export backtracking_linesearch

function backtracking_linesearch(model_orig, q, dobs, f_prev, g, proj; alpha=1f0, tau=.1f0, c1=1f-4, maxiter=10, verbose=false)

    # evaluate FWI objective as function of step size alpha
    function objective(alpha,p)
        model.m = proj(model_orig.m + alpha*reshape(p,model.n))

        # Set up linear operator and calculate data residual
        info = JUDI.TimeModeling.Info(prod(model.n), dobs.nsrc, JUDI.TimeModeling.get_computational_nt(q.geometry,dobs.geometry,model))
        F = JUDI.TimeModeling.judiModeling(info,model,q.geometry,dobs.geometry)
        dpred = F*q
        return .5f0*norm(dpred - dobs)^2
    end
    
    model = deepcopy(model_orig)    # don't modify original model
    p = -g/norm(g,Inf)  # normalized descent direction
    f_new = objective(alpha,p)
    iter = 1
    verbose == true && println("    Iter LS: ", iter, "; ", f_new, " <= ", f_prev + c1*alpha*dot(g,p), "; alpha: ", alpha)

    # sufficient decrease (Armijo) condition
    while f_new > f_prev + c1*alpha*dot(g,p) && iter < maxiter
        alpha *= tau
        f_new = objective(alpha,p)
        iter += 1
        verbose == true && println("    Iter LS: ", iter, "; ", f_new, " <= ", f_prev + c1*alpha*dot(g,p), "; alpha: ", alpha)
    end
    return alpha*p
end

function backtracking_linesearch(model_orig, q, dobs, f_prev, g, proj, objective; alpha=1f0, tau=.1f0, c1=1f-4, maxiter=10, verbose=false)

    model = deepcopy(model_orig)    # don't modify original model
    p = -g/norm(g,Inf)  # normalized descent direction
    model.m = proj(model_orig.m + alpha*reshape(p,model_orig.n))
    f_new = objective(model, q, dobs; compute_gradient=false)
    println("    LS: ",f_new)
    iter = 1
    verbose == true && println("    Iter LS: ", iter, "; ", f_new, " <= ", f_prev + c1*alpha*dot(g,p), "; alpha: ", alpha)

    # sufficient decrease (Armijo) condition
    while f_new > f_prev + c1*alpha*dot(g,p) && iter < maxiter
        alpha *= tau
        model.m = proj(model_orig.m + alpha*reshape(p,model_orig.n))
        f_new = objective(model, q, dobs; compute_gradient=false)
        println("    LS: ",f_new)
        iter += 1
        verbose == true && println("    Iter LS: ", iter, "; ", f_new, " <= ", f_prev + c1*alpha*dot(g,p), "; alpha: ", alpha)
    end
    return alpha*p
end



