export minConf_PQN, pqn_options


mutable struct PQN_params
    verbose
    optTol
    progTol
    maxIter
    suffDec
    corrections
    adjustStep
    bbInit
    store_trace
    SPGoptTol
    SPGprogTol
    SPGiters
    SPGtestOpt
    maxLinesearchIter
end

"""
    pqn_options(;verbose=2, optTol=1f-5, progTol=1f-7,
                maxIter=20, suffDec=1f-4, corrections=10, adjustStep=false,
                bbInit=false, store_trace=false, SPGoptTol=1f-6, SPGprogTol=1f-7,
                SPGiters=10, SPGtestOpt=false, maxLinesearchIter=20)

Options structure for Spectral Project Gradient algorithm.

    * verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3: debug)
    * optTol: tolerance used to check for optimality (default: 1e-5)
    * progTol: tolerance used to check for progress (default: 1e-9)
    * maxIter: maximum number of calls to objective (default: 15)
    * suffDec: sufficient decrease parameter in Armijo condition (default: 1e-4)
    * corrections: number of lbfgs corrections to store (default: 10)
    * adjustStep: use quadratic initialization of line search (default: 0)
    * bbInit: initialize sub-problem with Barzilai-Borwein step (default: 1)
    * store_trace: Whether to store the trace/history of x (default: false)
    * SPGoptTol: optimality tolerance for SPG direction finding (default: 1e-6)
    * SPGprogTol: SPG tolerance used to check for progress (default: 1e-7)
    * SPGiters: maximum number of iterations for SPG direction finding (default:10)
    * SPGtestOpt: Whether to check for optimality in SPG (default: false)
    * maxLinesearchIter: Maximum number of line search iteration (default: 20)
"""
function pqn_options(;verbose=2, optTol=1f-5, progTol=1f-7,
                     maxIter=20, suffDec=1f-4, corrections=10, adjustStep=false,
                     bbInit=false, store_trace=false, SPGoptTol=1f-6, SPGprogTol=1f-7,
                     SPGiters=10, SPGtestOpt=false, maxLinesearchIter=20)
    return PQN_params(verbose, optTol ,progTol, maxIter, suffDec, corrections,
                      adjustStep, bbInit, store_trace, SPGoptTol,
                      SPGprogTol, SPGiters, SPGtestOpt, maxLinesearchIter)
end

"""
    minConf_PQN(objective, projection, x,options)

Function for using a limited-memory projected quasi-Newton to solve problems of the form
  min objective(x) s.t. x in C

The projected quasi-Newton sub-problems are solved the spectral projected
gradient algorithm

  * objective(x): function to minimize (returns gradient as second argument)
  * projection(x): function that returns projection of x onto C
  * x: Initial guess
  * options: pqn_options structure
"""
function  minConf_PQN(funObj, x::Array{vDt}, funProj, options) where {vDt}
    nVars = length(x)
    spg_opt = spg_options(optTol=options.SPGoptTol,progTol=options.SPGprogTol, maxIter=options.SPGiters,
                          testOpt=options.SPGtestOpt, feasibleInit=~options.bbInit, verbose=0)
    # Output Parameter Settings
    if options.verbose >= 3
       @printf("Running PQN...\n");
       @printf("Number of L-BFGS Corrections to store: %d\n",options.corrections)
       @printf("Spectral initialization of SPG: %d\n",options.bbInit)
       @printf("Maximum number of SPG iterations: %d\n",options.SPGiters)
       @printf("SPG optimality tolerance: %.2e\n",options.SPGoptTol)
       @printf("SPG progress tolerance: %.2e\n",options.SPGprogTol)
       @printf("PQN optimality tolerance: %.2e\n",options.optTol)
       @printf("PQN progress tolerance: %.2e\n",options.progTol)
       @printf("Quadratic initialization of line search: %d\n",options.adjustStep)
       @printf("Maximum number of iterations: %d\n",options.maxIter)
    end

    # Result structure
    sol = result(x)

    # Setup Function to track number of evaluations
    projection(x) = (sol.n_project +=1; return funProj(x))
    objective(x) = (sol.n_feval +=1 ; return funObj(x))

    # Project initial parameter vector
    x = projection(x)
    # Evaluate initial parameters
    f, g = objective(x)
    update!(sol; iter=1, misfit=f, sol=x, gradient=g, store_trace=options.store_trace)

    # Output Log
    if options.verbose >= 2
        @printf("%10s %10s %10s %15s %15s %15s\n","Iteration","FunEvals","Projections","Step Length","Function Val","Opt Cond")
        @printf("%10d %10d %10d %15.5e %15.5e %15.5e\n",0, 0, 0, 0, f, norm(projection(x-g)-x, Inf))
    end
    
    # Check Optimality of Initial Point
    if maximum(abs.(projection(x-g)-x)) < options.optTol
        options.verbose >= 1 && @printf("First-Order Optimality Conditions Below optTol at Initial Point\n");
        update!(sol; misfit=f, gradient=g, store_trace=options.store_trace)
        return sol
    end

    # Initialize variables
    S = zeros(vDt, nVars,0)
    Y = zeros(vDt, nVars,0)
    Hdiag = 1

    for i=1:options.maxIter
        # Compute Step Direction
        if i == 1
            p = projection(x-g)
        else
            y = g-sol.gradient
            s = x-sol.sol
            S, Y, Hdiag = lbfgsUpdate(y,s,options.corrections,options.verbose,S,Y,Hdiag)

            # Make Compact Representation
            k = size(Y,2)
            L = zeros(Float32, k, k)
            for j = 1:k
                L[j+1:k,j] = transpose(S[:,j+1:k])*Y[:,j]
            end
            N = [S/Hdiag Y];
            M = [transpose(S)*S/Hdiag L;transpose(L) -diagm(diag(transpose(S)*Y))]
            HvFunc(v) = lbfgsHvFunc2(v,Hdiag,N,M)

            if options.bbInit
                # Use Barzilai-Borwein step to initialize sub-problem
                alpha = dot(s,s)/dot(s,y);
                if alpha <= 1e-10 || alpha > 1e10
                    alpha = min(1,1/sum(abs.(g)))
                end
                # Solve Sub-problem
                xSubInit = x-alpha*g
            else
                xSubInit = x
            end
            # Solve Sub-problem
            p = solveSubProblem(x,g,HvFunc,projection,spg_opt,xSubInit)
        end
        d = p-x

        # Check that Progress can be made along the direction
        gtd = dot(g,d)
        if gtd > -options.progTol && i>(options.corrections/2)
            options.verbose >= 1 && @printf("Directional Derivative below progTol\n")
            break
        end
        # Select Initial Guess to step length
        if i == 1 || ~options.adjustStep
            i==1 ? t = vDt(.1*norm(x, Inf)/norm(d, Inf)) : t = 1
        else
            t = vDt(min(1, 2*(f-sol.misfit)/gtd))
        end
        # Save history
        i>1 && update!(sol; iter=i, misfit=f, sol=x, gradient=g, store_trace=options.store_trace)
        # Evaluate the Objective and Gradient at the Initial Step
        x = sol.sol + t*d
        f, g = objective(x)

        # Backtracking Line Searchf
        lineSearchIters = 1
        ls_failed = false
        while (f > sol.misfit + options.suffDec*dot(d,(x-sol.sol)) || ~isLegal(f) || ~isLegal(g)) && lineSearchIters < options.maxLinesearchIter
            temp = t
            # Backtrack to next trial value
            if ~isLegal(f) || ~isLegal(g)
                options.verbose == 3 && @printf("Halving Step Size\n")
                t = t/2
            else
                options.verbose == 3 && @printf("Cubic Backtracking\n")
                t = vDt(polyinterp([0 f gtd; t f dot(g,d)]))
                isnan(t) && (t=  vDt(temp/2))
            end

            # Adjust if change is too small/large
            if t < temp*1e-3
                options.verbose == 3 && @printf("Interpolated value too small, Adjusting\n")
                t = vDt(temp*1e-3);
            elseif t > temp*0.6
                options.verbose == 3 && @printf("Interpolated value too large, Adjusting\n")
                t = vDt(temp*0.6)
            end

            # Check whether step has become too small
            if norm(t*d, 1) < options.progTol || t == 0
                ls_failed = true
                break
            end

            # Evaluate New Point
            x = sol.sol+ t*d
            f, g = objective(x)
            lineSearchIters = lineSearchIters+1
            ls_failed = ls_failed || (lineSearchIters == options.maxLinesearchIter)
        end
        ls_failed && (@printf("Too mane Line Search iterations\n"); break)
	    optCond = norm(projection(x-g) - x, Inf)
        i>1 && (terminate(options, optCond, t, d, f, sol.misfit) && break)
        # Output Log
        if options.verbose >= 2
            @printf("%10d %10d %10d %15.5e %15.5e %15.5e\n",i,sol.n_feval, sol.n_project, t, f, optCond)
        end

    end
    update!(sol; iter=options.maxIter+1, misfit=f, sol=x, gradient=g, store_trace=options.store_trace)
    return return sol
end
