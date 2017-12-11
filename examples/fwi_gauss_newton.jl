# FWI on Overthrust model using minConf library 
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: May 2017
#

using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot, IterativeSolvers

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
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")#, segy_depth_key="SourceDepth")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)	# 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

############################### FWI ###########################################

# Set up operators
ntComp = get_computational_nt(q.geometry,d_obs.geometry,model0)	# no. of computational time steps
info = Info(prod(model0.n),d_obs.nsrc,ntComp)
Pr = judiProjection(info,d_obs.geometry)
Ps = judiProjection(info,q.geometry)
F = judiModeling(info,model0)
J = judiJacobian(Pr*F*Ps',q)

# Optimization parameters
maxiter = 10
maxiter_GN = 8
batchsize = 8
fhistory_GN = zeros(Float32,maxiter)
proj(x) = reshape(median([vec(mmin) vec(x) vec(mmax)],2),model0.n)

# Gauss-Newton method
for j=1:maxiter
    println("Iteration: ",j)

    # Model predicted data for subset of sources
    i = randperm(d_obs.nsrc)[1:batchsize]
    d_pred = Pr[i]*F[i]*Ps[i]'*q[i]
    fhistory_GN[j] = .5f0*norm(d_pred - d_obs[i])^2
                        
    # GN update direction
    p = lsqr(J[i], d_pred - d_obs[i]; maxiter=maxiter_GN, verbose=true)
                                                                                
    # update model and bound constraints
    model0.m = proj(model0.m - reshape(p, model0.n))    # alpha=1
    figure(); imshow(sqrt.(1f0./model0.m)'); title(string(j))
end

figure(); imshow(sqrt(1f0./model0.m)'); title("FWI with SGD")


