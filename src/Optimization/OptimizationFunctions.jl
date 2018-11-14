
export isLegal, lbfgsUpdate, lbfgsHvFunc2, ssbin, solveSubProblem, subHv, polyval, polyinterp

function isLegal(v)
    return sum(isnan.(v))==0 & sum(isinf.(v))==0
end

function lbfgsUpdate(y,s,corrections,debug,old_dirs,old_stps,Hdiag)
    ys = dot(y,s);
    if ys > 1e-10 || size(old_dirs,2)==0
        numCorrections = size(old_dirs,2);
        if numCorrections < corrections
            # Full Update
            old_dirs = [old_dirs s];
            old_stps = [old_stps y];
        else
            # Limited-Memory Update
            old_dirs = [old_dirs[:,2:corrections] s];
            old_stps = [old_stps[:,2:corrections] y];
        end

        # Update scale of initial Hessian approximation
        Hdiag = ys/dot(y,y);
    else
        if debug==3
            @printf("Skipping Update\n");
        end
    end
    return old_dirs, old_stps, Hdiag
end

function lbfgsHvFunc2(v,Hdiag,N,M)
    if cond(M)>(1/(eps(Float32)))
        pr =  Array{Float32}(ssbin(M,500))
        L = spdiagm((pr,),0)
        Hv = v/Hdiag - N*L*((L*M*L)\(L*(transpose(N)*v)))
    
    else
        Hv = v/Hdiag - N*(M\(transpose(N)*v))
    end
    
    return Hv
end

function ssbin(A,nmv)
    # Stochastic matrix-free binormalization for symmetric real A.
    # x = ssbin(A,nmv,n)
    #   A is a symmetric real matrix or function handle. If it is a
    #     function handle, then v = A(x) returns A*x.
    #   nmv is the number of matrix-vector products to perform.
    #   [n] is the size of the matrix. It is necessary to specify n
    #     only if A is a function handle.
    #   diag(x) A diag(x) is approximately binormalized.

    # Jan 2010. Algorithm and code by Andrew M. Bradley (ambrad@stanford.edu).
    # Aug 2010. Modified to record and use dp. New omega schedule after running
    #   some tests.
    # Jul 2011. New strategy to deal with reducible matrices: Use the old
    #   iteration in the early iterations; then switch to snbin-like behavior,
    #   which deals properly with oscillation.
    n = size(A,1)
    d = ones(Float32,n,1)
    dp = d
    for k = 1:nmv
      # Approximate matrix-vector product
      u = randn(Float32,n,1)
      s = u./sqrt.(dp)
      y = A*s
      # omega^k
      alpha = (k - 1)/nmv
      omega = (1 - alpha)*1/2 + alpha*1/nmv
      # Iteration
      d = (1-omega)*d/sum(d) + omega*y.^2/sum(y.^2)
      if (k < min(32,floor(nmv/2)))
        # First focus on making d a decent approximation
        dp = d
      else
        # This block makes ssbin behave like snbin except for omega
        tmp = dp
        dp = d
        d = tmp
      end
    end
    return 1f0./(d.*dp).^(1/4)
end

function solveSubProblem(x,g,H,funProj,optTol,progTol,maxIter,testOpt,feasibleInit,x_init)
# Uses SPG to solve for projected quasi-Newton direction
    options = spg_options(verbose=0,
                          optTol=optTol,
                          progTol=progTol,
                          maxIter = maxIter,
                          testOpt = testOpt,
                          feasibleInit = feasibleInit)
    funObj(p) = subHv(p,x,g,H);
    p, f, funEvals, subProjects, hist = minConf_SPG(funObj,x_init,funProj,options);
    return p, subProjects 
end

function subHv(p,x,g,HvFunc)
    d = p-x;
    Hd = HvFunc(d);
    f = dot(g,d) + (1f0/2f0)*dot(d,Hd);
    g = g + Hd;
    return f, g
end


function polyval(p,x)
    value = 0
    order = length(p)
    for i=1:length(p)
        value = value + p[i]*x^(order-i)
    end
    return value
end

function polyinterp(points;xminBound=-Inf,xmaxBound=Inf)
# function [minPos] = polyinterp(points,doPlot,xminBound,xmaxBound)
#
#   Minimum of interpolating polynomial based on function and derivative
#   values
#
#   In can also be used for extrapolation if {xmin,xmax} are outside
#   the domain of the points.
#
#   Input:
#       points(pointNum,[x f g])
#       doPlot: set to 1 to plot, default: 0
#       xmin: min value that brackets minimum (default: min of points)
#       xmax: max value that brackets maximum (default: max of points)
#
#   set f or g to sqrt(-1) if they are not known
#   the order of the polynomial is the number of known f and g values minus 1


    nPoints = size(points,1);
    order = length(findin(imag(points[:,2:3]),0))-1;
    # Code for most common case:
    #   - cubic interpolation of 2 points
    #       w/ function and derivative values for both
    #   - no xminBound/xmaxBound

    if nPoints == 2 && order ==3 && isinf.(xmaxBound)
        # Solution in this case (where x2 is the farthest point):
        #    d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
        #    d2 = sqrt(d1^2 - g1*g2);
        #    minPos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
        #    t_new = min(max(minPos,x1),x2);
        minVal, minPos = findmin(points[:,1]);
        notMinPos = -minPos+3;
        d1 = points[minPos,3] + points[notMinPos,3] - 3*(points[minPos,2]-points[notMinPos,2])/(points[minPos,1]-points[notMinPos,1]);
        d2 = sqrt.(d1^2 - points[minPos,3]*points[notMinPos,3]);
        if isreal(d2)
            t = points[notMinPos,1] - (points[notMinPos,1] - points[minPos,1])*((points[notMinPos,3] + d2 - d1)/(points[notMinPos,3] - points[minPos,3] + 2*d2));
            minPos = min(max(t,points[minPos,1]),points[notMinPos,1]);
        else
            minPos = mean(points[:,1]);
        end
        return minPos
    end

    xmin = minimum(points[:,1]);
    xmax = maximum(points[:,1]);

    # Compute Bounds of Interpolation Area
    if isinf.(xminBound)
        xminBound = xmin;
    end
    if isinf.(xmaxBound)
        xmaxBound = xmax;
    end

    # Constraints Based on available Function Values
    A = zeros(Float32,0,order+1);
    b = zeros(Float32,0);
    for i = 1:nPoints
        if imag(points[i,2])==0
            constraint = zeros(Float32,1,order+1);
            for j = order:-1:0
                constraint[order-j+1] = points[i,1]^j;
            end
            A = [A;constraint];
            b = [b;points[i,2]];
        end
    end

    # Constraints based on available Derivatives
    for i = 1:nPoints
        if isreal(points[i,3])
            constraint = zeros(Float32,1,order+1);
            for j = 1:order
                constraint[j] = (order-j+1)*points[i,1]^(order-j);
            end
            A = [A;constraint];
            b = [b;points[i,3]];
        end
    end

    # Find interpolating polynomial
    params = A\b;

    # Compute Critical Points
    dParams = zeros(Float32,order);
    for i = 1:length(params)-1
        dParams[i] = params[i]*(order-i+1);
    end

    if sum(isinf.(dParams)) >0
        cp = copy(transpose([xminBound;xmaxBound;points[:,1]]));
    else
        cp = copy(transpose([xminBound;xmaxBound;points[:,1];-roots(dParams)]));
    end

    # Test Critical Points
    fmin = inf;
    minPos = (xminBound+xmaxBound)/2; # Default to Bisection if no critical points valid
    for xCP = cp
        if imag(xCP)==0 && xCP >= xminBound && xCP <= xmaxBound
            fCP = polyval(params,xCP);
            if imag(fCP)==0 && fCP < fmin
                minPos = real(xCP);
                fmin = real(fCP);
            end
        end
    end
    
    return minPos, fmin
end
