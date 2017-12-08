# 2D FWI on Overthrust model using minConf library 
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: May 2017
#

using PyCall, HDF5, opesciSLIM.TimeModeling, opesciSLIM.SLIM_optim, SeisIO, PyPlot

# Load starting model
n,d,o,m0 = read(h5open("/scratch/slim/pwitte/models/overthrust_mini.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1./model0.m)
vmin = ones(Float32,model0.n) * 1.3f0
vmax = ones(Float32,model0.n) * 6.5f0
vmin[:,1:21] = v0[:,1:21]	# keep water column fixed
vmax[:,1:21] = v0[:,1:21]

# Slowness squared [s^2/km^2]
mmin = vec((1f0./vmax).^2)
mmax = vec((1f0./vmin).^2)

# Load data
block = segy_read("/scratch/slim/pwitte/overthrust2D/overthrust_mini.segy")
dobs = joData(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")#, segy_depth_key="SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)	# 8 Hz wavelet
q = joData(src_geometry,wavelet)

############################### FWI ###########################################

# Optimization parameters
srand(10)
niterations = 10
batchsize = 10
fhistory_SGD = zeros(Float32,niterations)

# Projection operator for bound constraints
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)],2),model0.n)

function backtracking_linesearch(model_orig, q, dobs, f_prev, g, proj; alpha=1f0, tau=.1f0, c1=1f-4, maxiter=10)

    # evaluate FWI objective as function of step size alpha
    function objective(alpha,p)
        model.m = proj(model_orig.m + alpha*reshape(p,model.n))

        # Set up linear operator and calculate data residual
        info = Info(prod(model.n), dobs.nsrc, get_computational_nt(q.geometry,dobs.geometry,model))
        F = joModeling(info,model,q.geometry,dobs.geometry)
        dpred = F*q
        return .5f0*norm(dpred - dobs)^2
    end
    
    model = deepcopy(model_orig)    # don't modify original model
    p = -g/norm(g,Inf)  # normalized descent direction
    f_new = objective(alpha,p)
    iter = 1
    println("	Iter LS: ", iter, "; ", f_new, " <= ", f_prev + c1*alpha*dot(g,p), "; alpha: ", alpha)

    # sufficient decrease (Armijo) condition
    while f_new > f_prev + c1*alpha*dot(g,p) && iter < maxiter
        alpha *= tau
        f_new = objective(alpha,p)
        iter += 1
        println("	Iter LS: ", iter, "; ", f_new, " <= ", f_prev + c1*alpha*dot(g,p), "; alpha: ", alpha)
    end
    return alpha*p
end

# Main loop
for j=1:niterations

	# get fwi objective function value and gradient
	i = randperm(dobs.nsrc)[1:batchsize]
	fval, gradient = fwi_objective(model0,q[i],dobs[i])
	println("FWI iteration no: ",j,"; function value: ",fval)
    fhistory_SGD[j] = fval

	# linesearch
	step = backtracking_linesearch(model0,q[i],dobs[i],fval,gradient,proj;alpha=1f0)
	
	# Update model and bound projection
	model0.m = proj(model0.m + reshape(step,model0.n))
    figure(); imshow(sqrt.(1f0./model0.m)'); title(string(j))
end


