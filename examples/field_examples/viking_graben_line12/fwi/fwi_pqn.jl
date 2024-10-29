using DrWatson
@quickactivate "Viking"

using Viking
using Statistics, Random, LinearAlgebra, Interpolations, DelimitedFiles, Distributed
using JUDI, SlimOptim, NLopt, HDF5, SegyIO, Plots, ImageFiltering
using SetIntersectionProjection

############# INITIAL DATA #############
modeling_type = "bulk"    # slowness, bulk
frq = 0.005 # kHz

prestk_dir = "$(@__DIR__)/trim_segy/"
prestk_file = "shot"
dir_out = "$(@__DIR__)/fwi_pqn_$(modeling_type)/$(frq)Hz/"

model_file = "$(@__DIR__)/initial_model/model.h5"
# after each iteration you should set an appropriate model file computed with previous step
# model_file = "$(@__DIR__)/fwi_pqn_$(modeling_type)/0.005Hz/model 10.h5"
model_file_out = "model"

# use original wavelet file 
wavelet_file = "$(@__DIR__)/../FarField.dat" # dt=1, skip=25
# or use deghosted wavelet
# wavelet_file = "$(@__DIR__)/../proc/FarField_deghosted.dat" # dt=4, skip=0
wavelet_skip_start = 25 # 25 [lines] for raw source and 0 for deghosted source
wavelet_dt = 1          # 1 [ms] for raw source and 4 [ms] for deghosted source

segy_depth_key_src = "SourceSurfaceElevation"
segy_depth_key_rec = "RecGroupElevation"

seabed = 355  # [m]

############# INITIAL PARAMS #############
# water velocity, km/s
global vwater = 1.5

# water density, g/cm^3
global rhowater = 1.02

# JUDI options
buffer_size = 0f0    # limit model (meters) even if 0 buffer makes reflections from borders that does't hurt much the FWI result

# prepare folder for output data
mkpath(dir_out)

# Load data and create data vector
container = segy_scan(prestk_dir, prestk_file, ["SourceX", "SourceY", "GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = segy_depth_key_rec)

srcx = Float32.(get_header(container, "SourceX")[:,1])
grpx = Float32.(get_header(container, "GroupX")[:,1])
min_src_x = minimum(srcx)./1000f0
max_src_x = maximum(srcx)./1000f0
min_grp_x = minimum(grpx)./1000f0
max_grp_x = maximum(grpx)./1000f0
min_x = minimum([min_src_x, min_grp_x])
max_x = maximum([max_src_x, max_grp_x])

# Load starting model (mlog - slowness built with Vs from logs; mvsp - built from VSP)
fid = h5open(model_file, "r")
n, d, o = read(fid, "n", "d", "o")
if haskey(fid, "mvsp")
    m0 = Float32.(read(fid, "mvsp"))
else
    m0 = Float32.(read(fid, "m"))
end
close(fid)

n = Tuple(Int64(i) for i in n)
d = Tuple(Float32(i) for i in d)
o = Tuple(Float32(i) for i in o)
if modeling_type == "slowness"
    model0 = Model(n, d, o, m0)
elseif modeling_type == "bulk"
    rho0 = rho_from_slowness(m0)
    model0 = Model(n, d, o, m0, rho=rho0)
end

@info "modeling_type: $modeling_type"
@info "n: $n"
@info "d: $d"
@info "o: $o"

x = (o[1]:d[1]:o[1]+(n[1]-1)*d[1])./1000f0
z = (o[2]:d[2]:o[2]+(n[2]-1)*d[2])./1000f0

global seabed_ind = Int.(round.(seabed./d[2]))
if modeling_type == "slowness"
    model0.m[:,1:seabed_ind] .= (1/vwater)^2
elseif modeling_type == "bulk"
    model0.m[:,1:seabed_ind] .= (1/vwater)^2
    model0.rho[:,1:seabed_ind] .= rhowater
end

# Set up wavelet and source vector
src_geometry = Geometry(container; key = "source", segy_depth_key = segy_depth_key_src)

# setup wavelet
wavelet_raw = readdlm(wavelet_file, skipstart=wavelet_skip_start)
itp = LinearInterpolation(0:wavelet_dt:wavelet_dt*(length(wavelet_raw)-1), wavelet_raw[:,1], extrapolation_bc=0f0)
wavelet = Matrix{Float32}(undef,src_geometry.nt[1],1)
wavelet[:,1] = itp(0:src_geometry.dt[1]:src_geometry.t[1])

q = judiVector(src_geometry, wavelet)

# Data muters (t0 somehow can't be 0)
Ml_ref = judiDataMute(q.geometry, d_obs.geometry, vp=1100f0, t0=0.001f0, mode=:reflection) # keep reflections
Ml_tur = judiDataMute(q.geometry, d_obs.geometry, vp=1100f0, t0=0.001f0, mode=:turning)    # keep turning waves

# Bandpass filter
Ml_freq = judiFilter(d_obs.geometry, 0.001, frq*1000f0)
Mr_freq = judiFilter(q.geometry, 0.001, frq*1000f0)

############# FWI #############
# JUDI options
global jopt = JUDI.Options(
    IC = "fwi",
    limit_m = true,
    buffer_size = buffer_size,
    optimal_checkpointing=false,
    free_surface=true,  # free_surface is ON to model multiples as well
    space_order=16)     # increase space order for > 12 Hz source wavelet

# optimization parameters
niterations = 10
shot_from = 1
shot_to = length(d_obs)
shot_step = 3   # we may want to calculate only each Nth shot to economy time
count = 0
fhistory = Vector{Float32}(undef, 0)
mute_reflections = false
mute_turning = false

# SETUP CONSTARAINTS
options=PARSDMM_options()
options.FL=Float32
options=default_PARSDMM_options(options,options.FL)
constraint = Vector{SetIntersectionProjection.set_definitions}()

# Bound constraints
vmin = 1.2
vmax = 5.2
vBoundCoef = 0.5
kr = 50

# Slowness squared [s^2/km^2]
mmin = (1f0 ./ vmax).^2
mmax = (1f0 ./ vmin).^2
mminArr = ones(Float32, size(model0)) .* mmin
mmaxArr = ones(Float32, size(model0)) .* mmax
# mminArr = imfilter(model0.m.data * (1f0-vBoundCoef), Kernel.gaussian(kr))
# mmaxArr = imfilter(model0.m.data * (1f0+vBoundCoef), Kernel.gaussian(kr))
# ind = model0.m .< mminArr
# mminArr[ind] .= model0.m[ind]
# ind = model0.m .> mmaxArr
# mmaxArr[ind] .= model0.m[ind]

# bounds:
set_type = "bounds"
TD_OP = "identity"
app_mode = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,vec(mminArr),vec(mmaxArr),app_mode,custom_TD_OP));

#TV
(TV,dummy1,dummy2,dummy3) = get_TD_operator(model0.m,"TV",options.FL)
mvar_min = 0.0
mvar_max = norm(TV*vec(m0),1) * 2.5
set_type = "l1"
TD_OP = "TV"
app_mode = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,mvar_min,mvar_max,app_mode,custom_TD_OP));

# set up constraints with bounds and TV
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint, model0,options.FL)
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,set_Prop,model0,options)
options.rho_ini = ones(Float32, length(TD_OP))*10.0f0

proj_intersection = x-> PARSDMM(x, AtA, TD_OP, set_Prop, P_sub, model0, options)

# Projection function
function prj(input)
  @info "prj minimum(input): $(minimum(input))"
  input = Float32.(input)
  (x,dummy1,dummy2,dymmy3) = proj_intersection(vec(input))
  return x
end

# Set water bottom mask
wb_mask = ones(Float32,size(m0))
wb_mask[:, 1:seabed_ind] .= 0f0
# wb_mask[:, 1:seabed_ind+20] .= 0f0

# Build small cluster
nthread = Sys.CPU_THREADS
nw = 2

ENV["OMP_DISPLAY_ENV"] = "true"
ENV["OMP_PROC_BIND"] = "close"
ENV["OMP_NUM_THREADS"] = "$(div(nthread, nw))" 
addprocs(nw)
@show workers()
for k in 1:nworkers()
    place1 = (k - 1) * div(nthread,nworkers())
    place2 = (k + 0) * div(nthread,nworkers()) - 1
    @show place1, place2, div(nthread, nw)
    @spawnat workers()[k] ENV["GOMP_CPU_AFFINITY"] = "$(place1)-$(place2)";
end

@everywhere using Distributed, JUDI.TimeModeling, SlimOptim, LinearAlgebra, PythonPlot, SetIntersectionProjection, SegyIO

function objective(m, d_obs, wb_mask)

  global x, z, count, jopt, seabed_ind, mminArr, mmaxArr;
  count += 1

  @info "objective minimum(m): $(minimum(m))"
  m = reshape(m, size(model0))

  # pqn may cross the mmin/mmax limits
  ind = m .< mminArr
  m[ind] .= mminArr[ind]
  ind = m .> mmaxArr
  m[ind] .= mmaxArr[ind]

  model0.m .= Float32.(reshape(m, size(model0)))
  if modeling_type == "bulk"
    model0.rho .= Float32.(reshape(rho_from_slowness(model0.m), size(model0)))
    model0.rho[:,1:seabed_ind] .= rhowater
  end
  t = @elapsed begin
    indsrc = shot_from:shot_step:shot_to
    if mute_reflections
        fval, gradient = fwi_objective(model0, Mr_freq[indsrc]*q[indsrc], Ml_tur[indsrc]*Ml_freq[indsrc]*get_data(d_obs[indsrc]), options=jopt)
    elseif mute_turning
        fval, gradient = fwi_objective(model0, Mr_freq[indsrc]*q[indsrc], Ml_ref[indsrc]*Ml_freq[indsrc]*get_data(d_obs[indsrc]), options=jopt)
    else
        fval, gradient = fwi_objective(model0, Mr_freq[indsrc]*q[indsrc], Ml_freq[indsrc]*get_data(d_obs[indsrc]), options=jopt)
    end
    gradient = reshape(gradient, size(model0))
  end
  @info "elapsed fwi_objective time: $t"
  gradient .*= wb_mask
  if gscale == 0f0
      # compute scalar from first gradient, apply to future gradients
      global gscale = .25f0 ./ maximum(gradient) 
      @show gscale
  end
  gradient .*= gscale

  push!(fhistory, fval)

  println("iteration: ", count, "\tfval: ", fval, "\tnorm: ", norm(gradient))
  save_data(x,z,adjoint(reshape(model0.m.data,size(model0))); 
          pltfile=dir_out * "FWI slowness $count.png",
          title="FWI slowness^2 with PQN $modeling_type: $(frq*1000)Hz, iter $count",
          colormap=:rainbow,
          h5file=dir_out * model_file_out * " " * string(count) * ".h5",
          h5openflag="w",
          h5varname="m")
  save_data(x,z,sqrt.(1f0 ./ adjoint(reshape(model0.m.data,size(model0)))); 
          pltfile=dir_out * "FWI $count.png",
          title="FWI velocity with PQN $modeling_type: $(frq*1000)Hz, iter $count",
          colormap=:rainbow,
          h5file=dir_out * model_file_out * " " * string(count) * ".h5",
          h5openflag="r+",
          h5varname="v")
  save_data(x,z,adjoint(reshape(gradient.data,size(model0))); 
          pltfile=dir_out * "Gradient $count.png",
          title="FWI gradient with PQN $modeling_type: $(frq*1000)Hz, iter $count",
          clim=(-maximum(gradient.data)/5f0, maximum(gradient.data)/5f0),
          colormap=:bluesreds,
          h5file=dir_out * model_file_out * " " * string(count) * ".h5",
          h5openflag="r+",
          h5varname="grad")
  save_fhistory(fhistory; 
          h5file=dir_out * model_file_out * " " * string(count) * ".h5",
          h5openflag="r+",
          h5varname="fhistory")

  return fval, vec(gradient.data)
end

# struct to save the first gradient scalar
gscale = 0f0
g(x) = objective(x, d_obs, wb_mask)

# FWI with PQN
gscale = 0f0
options_pqn = pqn_options(progTol=0, store_trace=true, verbose=3, maxIter=niterations)
sol = pqn(g, vec(m0), prj, options_pqn);

rmprocs(workers());