# 2D FWI on Overthrust model using minConf library
# Author: Philipp Witte, pwitte@eoas.ubc.ca
# Date: December 2017
#

using Statistics, Random, Pkg
using LinearAlgebra
using JUDI.TimeModeling, JUDI.SLIM_optim, HDF5, SeisIO, PyPlot, FFTW
using SetIntersectionProjection

# Load starting model
n,d,o,m0 = read(h5open("../../data/overthrust_model.h5","r"), "n", "d", "o", "m0")
model0 = Model((n[1],n[2]), (d[1],d[2]), (o[1],o[2]), m0)

# Bound constraints
v0 = sqrt.(1f0 ./ model0.m)

# Load data
block = segy_read("../../data/overthrust_shot_records.segy")
d_obs = judiVector(block)

# Set up wavelet
src_geometry = Geometry(block; key="source")
wavelet = ricker_wavelet(src_geometry.t[1],src_geometry.dt[1],0.008f0)  # 8 Hz wavelet
q = judiVector(src_geometry,wavelet)

############################### FWI ###########################################

# Optimization parameters
niterations = 10
batchsize = 10
fhistory_SGD = zeros(Float32,niterations)


########## Setup constraints
# with constraints:
options=PARSDMM_options()
options.FL=Float32
options=default_PARSDMM_options(options,options.FL)
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 1000
options.feas_tol= 0.001
options.obj_tol=0.001
options.evol_rel_tol = 0.00001

options.rho_ini=[1.0f0]

set_zero_subnormals(true)
BLAS.set_num_threads(2)
FFTW.set_num_threads(2)
options.parallel=false
options.feasibility_only = false
options.zero_ini_guess=true

constraint = Vector{SetIntersectionProjection.set_definitions}()
#bounds:
vmin = ones(Float32,model0.n) .* 1.3f0
vmax = ones(Float32,model0.n) .* 6.5f0
vmin[:,1:21] .= v0[:,1:21]   # keep water column fixed
vmax[:,1:21] .= v0[:,1:21]
# Slowness squared [s^2/km^2]
m_min = vec((1f0 ./ vmax).^2)
m_max = vec((1f0 ./ vmin).^2)
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#TV
(TV,dummy1,dummy2,dummy3) = get_TD_operator(model0,"TV",options.FL)
m_min     = 0.0
m_max     = norm(TV*vec(v0),1) *2.0f0
set_type  = "l1"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#set up constraints, precompute some things and define projector
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,model0,options.FL)
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,model0,options)
options.rho_ini        = ones(length(TD_OP))*10.0

proj_intersection = x-> PARSDMM(x, AtA, TD_OP, set_Prop, P_sub, model0, options)  

function prj(input)
  (x,dummy1,dummy2,dymmy3) = proj_intersection(vec(input))
  return reshape(x, model0.n)
end

########## Run

# Main loop
for j=1:niterations

    # get fwi objective function value and gradient
    i = randperm(d_obs.nsrc)[1:batchsize]
    fval, gradient = fwi_objective(model0,q[i],d_obs[i])
    println("FWI iteration no: ",j,"; function value: ",fval)
    fhistory_SGD[j] = fval

	# linesearch
	step = backtracking_linesearch(model0, q[i], d_obs[i], fval, gradient, prj; alpha=1f0)

	# Update model and bound projection
	model0.m = prj(model0.m + reshape(step,model0.n))
end

figure(); imshow(sqrt.(1f0./adjoint(model0.m))); title("FWI with SGD")
