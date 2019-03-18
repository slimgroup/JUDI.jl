export boundproject, isLegal, minConf_SPG, polyinterp, polyval, spg_options

mutable struct SPG_params
    verbose
    optTol
    progTol
    maxIter
    suffDec
    interp
    memory
    useSpectral
    curvilinear
    feasibleInit
    testOpt
    bbType
    testInit
    optNorm
    iniStep
end

function spg_options(;verbose=3,optTol=1f-5,progTol=1f-7,
                     maxIter=20,suffDec=1f-4,interp=0,memory=2,
                     useSpectral=true,curvilinear=false,
                     feasibleInit=false,testOpt=true,
                     bbType=true,testInit=false,
                     optNorm=Inf,iniStep=1f0)
    return SPG_params(verbose,optTol,progTol,
                        maxIter,suffDec,interp,memory,
                        useSpectral,curvilinear,
                        feasibleInit,testOpt,
                        bbType,testInit,optNorm,iniStep)
end

function minConf_SPG(funObj, x, funProj, options)
    # function [x,f] = minConF_SPG(funObj,x,funProj,options)
    #
    # Function for using Spectral Projected Gradient to solve problems of the form
    #   min funObj(x) s.t. x in C
    #
    #   @funObj(x): function to minimize (returns gradient as second argument)
    #   @funProj(x): function that returns projection of x onto C
    #
    #   options:
    #       verbose: level of verbosity (0: no output, 1: final, 2: iter (default), 3:
    #       debug)
    #       optTol: tolerance used to check for optimality (default: 1e-5)
    #       progTol: tolerance used to check for lack of progress (default: 1e-9)
    #       maxIter: maximum number of calls to funObj (default: 500)
    #       numDiff: compute derivatives numerically (0: use user-supplied
    #       derivatives (default), 1: use finite differences, 2: use complex
    #       differentials)
    #       suffDec: sufficient decrease parameter in Armijo condition (default
    #       : 1e-4)
    #       interp: type of interpolation (0: step-size halving, 1: quadratic,
    #       2: cubic)
    #       memory: number of steps to look back in non-monotone Armijo
    #       condition
    #       useSpectral: use spectral scaling of gradient direction (default:
    #       1)
    #       curvilinear: backtrack along projection Arc (default: 0)
    #       testOpt: test optimality condition (default: 1)
    #       feasibleInit: if 1, then the initial point is assumed to be
    #       feasible
    #       bbType: type of Barzilai Borwein step (default: 1)
    #
    #   Notes:
    #       - if the projection is expensive to compute, you can reduce the
    #           number of projections by setting testOpt to 0

    if options.verbose >= 3
       @printf("Running SPG...\n");
       @printf("Number of objective function to store: %d\n",options.memory);
       @printf("Using  spectral projection : %s\n",options.useSpectral);
       @printf("Maximum number of function evaluations: %d\n",options.maxIter);
       @printf("SPG optimality tolerance: %.2e\n",options.optTol);
       @printf("SPG progress tolerance: %.2e\n",options.progTol);
    end


    nVars = length(x)

    # Output Log
    if options.verbose >= 2
        if options.testOpt
            @printf("%10s %10s %10s %15s %15s %15s\n","Iteration","FunEvals","Projections","Step Length","Function Val","Opt Cond")
        else
            @printf("%10s %10s %10s %15s %15s\n","Iteration","FunEvals","Projections","Step Length","Function Val")
        end
    end

    # Make objective function (if using numerical derivatives)
    funEvalMultiplier = 1

    # Evaluate Initial Point
    if ~options.feasibleInit
        x = funProj(x)
    end
    f, g = funObj(x)
    hist = f
    projects = 1
    funEvals = 1

    x_best = x
    f_best = f

    # Optionally check optimality
    if options.testOpt && options.testInit
        projects = projects+1
        if norm(funProj(x-g)-x,options.optNorm) < optTol
            if verbose >= 1
                @printf("First-Order Optimality Conditions Below optTol at Initial Point, norm g is %5.4f \n", norm(g))
            end
            return
        end
    end

    i = 1
    while funEvals <= options.maxIter

        # Compute Step Direction
        if i == 1 || ~options.useSpectral
            alpha = 1
        else
            y = g-g_old
            s = x-x_old
            if options.bbType == 1
                alpha = dot(s,s)/dot(s,y)
            else
                alpha = dot(s,y)/dot(y,y)
            end
            if alpha <= 1e-10 || alpha > 1e10
                alpha = 1
            end
        end
        global d = -Float32(alpha)*g
        global f_old = f
        global x_old = x
        global g_old = g

        # Compute Projected Step
        if ~options.curvilinear
            if i==1
                d = funProj(x+d)-x
            else
                d = funProj(x+d)-x
            end
            projects = projects+1
        end
        # Check that Progress can be made along the direction
        gtd = dot(g,d)
        if gtd > -options.progTol
            if options.verbose >= 1
                @printf("Directional Derivative below progTol\n")
            end
            break
        end

        # Select Initial Guess to step length
        t = options.iniStep

        # Compute reference function for non-monotone condition

        if options.memory == 1
            funRef = f
        else
            if i == 1
                global old_fvals = -Inf*ones(Float32,options.memory,1)
            end

            if i <= options.memory
                global old_fvals[i] = f
            else
                global old_fvals = [old_fvals[2:end];f]
            end
            funRef = maximum(old_fvals)
        end
        # Evaluate the Objective and Gradient at the Initial Step
        if options.curvilinear
            x_new = funProj(x + Float32(t)*d)
            projects = projects+1
        else
            x_new = x + Float32(t)*d
        end
        f_new, g_new = funObj(x_new)
        funEvals = funEvals+1
        # Backtracking Line Search
        lineSearchIters = 1
        while f_new > funRef + options.suffDec*dot(g,(x_new-x)) || ~isLegal(f_new) || ~isLegal(g_new)
            temp = t
            if lineSearchIters == 1
                @printf("Unit step length not feasible, starting line search\n")
            end
            # @printf("%10d %15.5e %15.5e %15.5e %15.5e\n",lineSearchIters,t, f_new,funRef,funRef + options.suffDec*dot(g,(x_new-x)))
            if options.interp == 0 || ~isLegal(f_new)
                if options.verbose == 3
                    @printf("Halving Step Size\n");
                end
                t = t/2;
            elseif options.interp == 2 && isLegal(g_new)
                if options.verbose == 3
                    @printf("Cubic Backtracking\n");
                end
                t = polyinterp([0 f gtd; t f_new dot(g_new,d)]);
            elseif lineSearchIters < 2 || ~isLegal(f_prev)
                if options.verbose == 3
                    @printf("Quadratic Backtracking\n");
                end
                t = polyinterp([0 f gtd; t f_new sqrt(complex(-1))]);
            else
                if options.verbose == 3
                    @printf("Cubic Backtracking on Function Values\n");
                end
                t = polyinterp([0 f gtd; t f_new sqrt(complex(-1));t_prev f_prev sqrt(complex(-1))]);
            end

            # Check whether step has become too small
            if maximum(abs.(t*d)) < options.progTol || t == 0
                if options.verbose == 3
                    @printf("Line Search failed\n")
                end
                t = 0
                f_new = f
                g_new = g
                break
            end

            # Evaluate New Point
            f_prev = f_new
            t_prev = temp
            if options.curvilinear
                x_new = funProj(x + Float32(t)*d)
                projects = projects+1
            else
                x_new = x + Float32(t)*d
            end
            f_new, g_new = funObj(x_new)
            funEvals = funEvals+1
            lineSearchIters = lineSearchIters+1
            if funEvals*funEvalMultiplier > options.maxIter
                if options.verbose >= 1
                    @printf("Function Evaluations exceeds maxIter\n")
                end
                f_new = f
                g_new = g
                x_new = x
                break
            end

            if lineSearchIters > 20
                if options.verbose >= 1
                    @printf("Linesearch Iterations exceeds maxLinesearchIter\n")
                end
                f_new = f
                g_new = g
                x_new = x
                break
            end

        end

        # Take Step
        x = x_new
        f = f_new
        g = g_new
        hist = [hist;f]
        if f < f_best
            x_best = x
            f_best = f
        end

        if options.testOpt
            optCond = norm(funProj(x-g)-x,options.optNorm)
            projects = projects+1
        end
        # Output Log
        if options.verbose >= 2
            if options.testOpt
                @printf("%10d %10d %10d %15.5e %15.5e %15.5e\n",i,funEvals*funEvalMultiplier,projects,t,f,optCond)
            else
                @printf("%10d %10d %10d %15.5e %15.5e\n",i,funEvals*funEvalMultiplier,projects,t,f)
            end
        end

        # Check optimality
        if options.testOpt
            if optCond < options.optTol && i>1
                if options.verbose >= 1
                    @printf("First-Order Optimality Conditions Below optTol\n")
                end
                break
            end
        end

        if maximum(abs.(t*d)) < options.progTol && i>1
            if options.verbose >= 1
                @printf("Step size below progTol\n")
            end
            break
        end

        if abs.(f-f_old) < options.progTol && i>1
            if options.verbose >= 1
                @printf("Function value changing by less than progTol\n")
            end
            break
        end

        if funEvals*funEvalMultiplier > options.maxIter
            if options.verbose >= 1
                @printf("Function Evaluations exceeds maxIter\n")
            end
            break
        end

        i = i + 1
    end

    # Restore best iteration
    x = x_best
    f = f_best
    return x, f, funEvals, projects, hist
end
