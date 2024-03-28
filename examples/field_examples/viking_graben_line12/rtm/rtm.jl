using DrWatson
@quickactivate "Viking"

using Viking
using Statistics, Random, LinearAlgebra, Interpolations, DelimitedFiles, Distributed
using JUDI, NLopt, HDF5, SegyIO, Plots

############# INITIAL DATA #############
modeling_type = "bulk"  # slowness, bulk

prestk_dir = "$(@__DIR__)/../proc/"
prestk_file = "s_deghost_gain_mute_dip_radon.sgy"

# dir_out = "$(@__DIR__)/rtm_pqn/"
# dir_out = "$(@__DIR__)/rtm_spg/"
dir_out = "$(@__DIR__)/rtm_lbfgs/"

# choose the most accurate model
# model_file = "$(@__DIR__)/../fwi/fwi_pqn_$(modeling_type)/0.005Hz/model 10.h5"
# model_file = "$(@__DIR__)/../fwi/fwi_spg_$(modeling_type)/0.005Hz/model 10.h5"
model_file = "$(@__DIR__)/../fwi/fwi_lbfgs_$(modeling_type)/0.005Hz/model 10.h5"

# use original wavelet file 
# wavelet_file = "$(@__DIR__)/../FarField.dat" # dt=1, skip=25
# or use deghosted wavelet
wavelet_file = "$(@__DIR__)/../proc/FarField_deghosted.dat" # dt=4, skip=0
wavelet_skip_start = 0 # 25 [lines] for raw source and 0 for deghosted source
wavelet_dt = 4          # 1 [ms] for raw source and 4 [ms] for deghosted source

segy_depth_key_src = "SourceSurfaceElevation"
segy_depth_key_rec = "RecGroupElevation"

dense_factor = 2 # make model n-times denser to achieve better stability
seabed = 355  # [m]

# water velocity, km/s
global vwater = 1.5

# water density, g/cm^3
global rhowater = 1.02

# JUDI options
buffer_size = 0f0    # limit model (meters)

# prepare folder for output data
mkpath(dir_out)

# Load data and create data vector
# block = segy_read(prestk_dir * prestk_file)
container = segy_scan(prestk_dir, prestk_file, ["SourceX", "SourceY", "GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
d_obs = judiVector(container; segy_depth_key = segy_depth_key_rec)

srcx = Float32.(get_header(container, "SourceX")[:,1])
grpx = Float32.(get_header(container, "GroupX")[:,1])
min_src_x = minimum(srcx)./1000f0
max_src_x = maximum(srcx)./1000f0
min_grp_x = minimum(grpx)./1000f0
max_grp_x = maximum(grpx)./1000f0
min_cdp_x = (min_src_x+min_grp_x)/2f0
max_cdp_x = (max_src_x+max_grp_x)/2f0
min_x = minimum([min_src_x, min_grp_x])
max_x = maximum([max_src_x, max_grp_x])

# Load starting model (mlog - slowness built with Vs from logs; mvsp - built from VSP)
n, d, o, m0 = read(h5open(model_file, "r"), "n", "d", "o", "m")
n = Tuple(Int64(i) for i in n)
d = Tuple(Float32(i) for i in d)
o = Tuple(Float32(i) for i in o)

i_dense = 1:1/Float32(dense_factor):size(m0)[1]
j_dense = 1:1/Float32(dense_factor):size(m0)[2]

m0_itp = interpolate(m0, BSpline(Linear()))
m0 = m0_itp(i_dense, j_dense)
n = size(m0)
d = Tuple(Float32(i/dense_factor) for i in d)

if modeling_type == "slowness"
    model0 = Model(n, d, o, m0_dense)
elseif modeling_type == "bulk"
    rho0 = rho_from_slowness(m0)
    model0 = Model(n, d, o, m0, rho=rho0)
end

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

############################################## RTM #################################################

# JUDI options
jopt = JUDI.Options(
    space_order=32,
    limit_m = true,
    buffer_size = buffer_size,
    optimal_checkpointing=false,
    IC = "isic")

# Right-hand preconditioners (model topmute)
idx_wb = find_water_bottom(reshape(model0.m, size(model0)))
Tm = judiTopmute(size(model0), idx_wb, 10)  # Mute water column
S = judiDepthScaling(model0)
Mr = S*Tm

# Left-hand side preconditioners
Ml = judiDataMute(q.geometry, d_obs.geometry, vp=1100f0, t0=0.001f0, mode=:reflection) # keep reflections

# Setup operators
Pr = judiProjection(d_obs.geometry)
F = judiModeling(model0; options=jopt)
Ps = judiProjection(q.geometry)
J = judiJacobian(Pr*F*adjoint(Ps), q)

shot_from = 1
shot_to = length(d_obs)
shot_step = 3   # only compute each 3rd shot (that is enough)

indsrc = rand(shot_from:shot_from+shot_step-1):shot_step:shot_to

# Topmute
d_obs = Ml[indsrc]*d_obs[indsrc]

# RTM
rtm = adjoint(Mr)*adjoint(J[indsrc])*d_obs

data = rtm isa Vector ? rtm : rtm.data

zmax = 3.2f0  # km
nz = sum(z .<= zmax)

# save RTM as HDF5 and plots
save_data(x,z[1:nz],adjoint(reshape(data, size(model0))[:,1:nz]); 
    pltfile=dir_out * "RTM_$(modeling_type).png",
    title="RTM",
    clim=(-mean(abs.(data))*15f0, mean(abs.(data))*15f0),
    colormap=:seismic,
    h5file=dir_out * "rtm_$(modeling_type).h5",
    h5openflag="w",
    h5varname="rtm")

# save RTM as SEGY
block_out = SeisBlock(collect(reshape(data, size(model0))'))
set_header!(block_out, "dt", Int16(round(z[2]*1e6)))
set_header!(block_out, "CDP", collect(1:size(data)[1]))
set_header!(block_out, "CDPX", collect(Int32.(round.(x*100f0))))
segy_write(dir_out * "rtm_$(modeling_type).sgy", block_out)