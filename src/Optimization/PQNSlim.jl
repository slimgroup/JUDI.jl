export minConf_PQN, pqn_options


mutable struct PQN_params
    verbose
    optTol
    progTol
    maxIter
    maxProject
    suffDec
    corrections
    adjustStep
    bbInit
    SPGoptTol
    SPGprogTol
    SPGiters
    SPGtestOpt
end

function pqn_options(;verbose=3,optTol=1f-5,progTol=1f-7,
                     maxIter=20,maxProject=100000, suffDec=1f-4,
                     corrections=10, adjustStep=false,bbInit=false,
                     SPGoptTol=1f-6,SPGprogTol=1f-7,
                     SPGiters=10,SPGtestOpt=false)
    return PQN_params(verbose,optTol,progTol,maxIter,
                      maxProject,suffDec,corrections,
                      adjustStep,bbInit,SPGoptTol,
                      SPGprogTol,SPGiters,SPGtestOpt)
end


function  minConf_PQN(funObj,x,funProj,options)
# function [x,f] = minConf_PQN(funObj,funProj,x,options)
#
# Function for using a limited-memory projected quasi-Newton to solve problems of the form
#   min funObj(x) s.t. x in C
#
# The projected quasi-Newton sub-problems are solved the spectral projected
# gradient algorithm
#
#   @funObj(x): function to minimize (returns gradient as second argument)
#   @funProj(x): function that returns projection of x onto C
#
#   options:
#       verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3:
#       debug)
#       optTol: tolerance used to check for optimality (default: 1e-5)
#       progTol: tolerance used to check for progress (default: 1e-9)
#       maxIter: maximum number of calls to funObj (default: 15)
#       maxProject: maximum number of calls to funProj (default: 100000)
#       numDiff: compute derivatives numerically (0: use user-supplied
#       derivatives (default), 1: use finite differences, 2: use complex
#       differentials)
#       suffDec: sufficient decrease parameter in Armijo condition (default: 1e-4)
#       corrections: number of lbfgs corrections to store (default: 10)
#       adjustStep: use quadratic initialization of line search (default: 0)
#       bbInit: initialize sub-problem with Barzilai-Borwein step (default: 1)
#       SPGoptTol: optimality tolerance for SPG direction finding (default: 1e-6)
#       SPGiters: maximum number of iterations for SPG direction finding (default:10)

    fsave = 0;
    nVars = length(x);
    #iter_save = zeros(nVars,1);

    # Output Parameter Settings
    if options.verbose >= 3
       @printf("Running PQN...\n");
       @printf("Number of L-BFGS Corrections to store: %d\n",options.corrections);
       @printf("Spectral initialization of SPG: %d\n",options.bbInit);
       @printf("Maximum number of SPG iterations: %d\n",options.SPGiters);
       @printf("SPG optimality tolerance: %.2e\n",options.SPGoptTol);
       @printf("SPG progress tolerance: %.2e\n",options.SPGprogTol);
       @printf("PQN optimality tolerance: %.2e\n",options.optTol);
       @printf("PQN progress tolerance: %.2e\n",options.progTol);
       @printf("Quadratic initialization of line search: %d\n",options.adjustStep);
       @printf("Maximum number of function evaluations: %d\n",options.maxIter);
       @printf("Maximum number of projections: %d\n",options.maxProject);
    end

    # Output Log
    if options.verbose >= 2
            @printf("%10s %10s %10s %15s %15s %15s\n","Iteration","FunEvals","Projections","Step Length","Function Val","Opt Cond");
    end

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1;

    # Project initial parameter vector
    x = funProj(x);
    projects = 1;

    # Evaluate initial parameters
    f, g = funObj(x);
    funEvals = 1;

    # Check Optimality of Initial Point
    projects = projects+1;
    if maximum(abs.(funProj(x-g)-x)) < options.optTol
        if options.verbose >= 1
            @printf("First-Order Optimality Conditions Below optTol at Initial Point\n");
        end
        fsave = f;
        return;
    end

    i = 1;
    while funEvals <= options.maxIter

        # Compute Step Direction
        if i == 1

            p = funProj(x-g);

            projects = projects+1;
            S = zeros(Float32, nVars,0);
            Y = zeros(Float32, nVars,0);
            Hdiag = 1;
        else
            y = g-g_old;
            s = x-x_old;
            S, Y, Hdiag = lbfgsUpdate(y,s,options.corrections,options.verbose,S,Y,Hdiag);

            # Make Compact Representation
            k = size(Y,2);
            L = zeros(Float32, k, k);
            for j = 1:k
                L[j+1:k,j] = transpose(S[:,j+1:k])*Y[:,j];
            end
            N = [S/Hdiag Y];
            M = [transpose(S)*S/Hdiag L;transpose(L) -diagm(diag(transpose(S)*Y))];
            HvFunc(v) = lbfgsHvFunc2(v,Hdiag,N,M);

            if options.bbInit
                # Use Barzilai-Borwein step to initialize sub-problem
                alpha = dot(s,s)/dot(s,y);
                if alpha <= 1e-10 || alpha > 1e10
                    alpha = min(1,1/sum(abs.(g)));
                end

                # Solve Sub-problem
                xSubInit = x-alpha*g;
                feasibleInit = false;
            else
                xSubInit = x;
                feasibleInit = true;
            end
            # Solve Sub-problem
            p, subProjects = solveSubProblem(x,g,HvFunc,funProj,options.SPGoptTol,options.SPGprogTol,options.SPGiters,options.SPGtestOpt,feasibleInit,xSubInit);
            projects = projects+subProjects;
        end
        global d = p-x;
        global g_old = g;
        global x_old = x;

        # Check that Progress can be made along the direction
        gtd = dot(g,d);
        if gtd > -options.progTol && i>(options.corrections/2)
            if options.verbose >= 1
                @printf("Directional Derivative below progTol\n");
            end
            break;
        end

        # Select Initial Guess to step length
        if i == 1 || options.adjustStep == 0
           t = 1;
        else
            t = min(1,2*(f-f_old)/gtd);
        end

        # Bound Step length on first iteration
        if i == 1
            t = min(1,1/sum(abs.(d)));
        end

        # Evaluate the Objective and Gradient at the Initial Step
        if t == 1
            x_new = p;
        else
            x_new = x + Float32(t)*d;
        end
        f_new, g_new = funObj(x_new);
        funEvals = funEvals+1;

        # Backtracking Line Search
        f_old = f;
        while f_new > f + options.suffDec*dot(g,(x_new-x)) || ~isLegal(f_new) || ~isLegal(g_new)
            temp = t;

            # Backtrack to next trial value
            if ~isLegal(f_new) || ~isLegal(g_new)
                if options.verbose == 3
                    @printf("Halving Step Size\n");
                end
                t = t/2;
            else
                if options.verbose == 3
                    @printf("Cubic Backtracking\n");
                end
                t = polyinterp([0 f gtd; t f_new transpose(g_new)*d]);
            end

            # Adjust if change is too small/large
            if t < temp*1e-3
                if options.verbose == 3
                    @printf("Interpolated value too small, Adjusting\n");
                end
                t = temp*1e-3;
            elseif t > temp*0.6
                if options.verbose == 3
                    @printf("Interpolated value too large, Adjusting\n");
                end
                t = temp*0.6;
            end

            # Check whether step has become too small
            if sum(abs.(t*d)) < options.progTol || t == 0
                if options.verbose == 3
                    @printf("Line Search failed\n");
                end
                t = 0;
                f_new = f;
                g_new = g;
                break;
            end

            # Evaluate New Point
            f_prev = f_new;
            t_prev = temp;
            x_new = x + Float32(t)*d;
            f_new, g_new = funObj(x_new);
            funEvals = funEvals+1;

            if funEvals > options.maxIter
                break
            end

        end

        # Take Step
        x = x_new;
        f = f_new;
        fsave = [fsave ; f];
        # iter_save = [iter_save x];
        g = g_new;

        optCond = maximum(abs.(funProj(x-g)-x));
        projects = projects+1;

        # Output Log
        if options.verbose >= 2
                @printf("%10d %10d %10d %15.5e %15.5e %15.5e\n",i,funEvals*funEvalMultiplier,projects,t,f,optCond);
        end

        # Check optimality
            if optCond < options.optTol && (options.verbose >= 1)
                @printf("First-Order Optimality Conditions Below optTol\n");
                break;
            end

        if (maximum(abs.(t*d)) < options.progTol) && (funEvals*funEvalMultiplier > options.maxIter/2)
            if options.verbose >= 1
                @printf("Step size below progTol\n");
            end
            break;
        end

        if abs.(f-f_old) < options.progTol
            if options.verbose >= 1
                @printf("Function value changing by less than progTol\n");
            end
            break;
        end

        if funEvals*funEvalMultiplier > options.maxIter
            if options.verbose >= 1
                @printf("Function Evaluations exceeds maxIter\n");
            end
            break;
        end

        if projects > options.maxProject
            if options.verbose >= 1
                @printf("Number of projections exceeds maxProject\n");
            end
            break;
        end

        i = i + 1;
    #    pause
    end

    return x, fsave, funEvals
end
