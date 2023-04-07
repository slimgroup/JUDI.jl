# using Pkg
# Pkg.add("Statistics")
# Pkg.add("Random")
# Pkg.add("LinearAlgebra")
# Pkg.add("Interpolations")
# Pkg.add("DelimitedFiles")
# Pkg.add("Distributed")
# Pkg.add("DSP")
# Pkg.add("SlimOptim")
# Pkg.add("NLopt")
# Pkg.add("HDF5")
# Pkg.add("SegyIO")
# Pkg.add("Plots")
# Pkg.add("SetIntersectionProjection")

using Statistics, Random, LinearAlgebra, Interpolations, DelimitedFiles, Distributed, DSP
using JUDI, SlimOptim, NLopt, HDF5, SegyIO, Plots
using SetIntersectionProjection


############# INITIAL DATA #############
modeling_type = "acoustic"    # scalar, acoustic
frq = 0.005 # kHz
trc_len = 4000  # ms

prestk_dir = "$(@__DIR__)/filt/$(frq)hz_$(trc_len)ms/"
dir_out = "$(@__DIR__)/fwi_$(modeling_type)/$(frq)Hz/"

prestk_file = "shot"
model_file = "$(@__DIR__)/initial_model/model.h5"
# after each iteration you should set an appropriate model file computed with previous step
# model_file = "$(@__DIR__)/fwi_$(modeling_type)/0.005Hz/model.h5"
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
min_cdp_x = (min_src_x+min_grp_x)/2f0
max_cdp_x = (max_src_x+max_grp_x)/2f0
min_x = minimum([min_src_x, min_grp_x])
max_x = maximum([max_src_x, max_grp_x])

# helpful functions
rho_from_slowness(m) = 0.23.*(sqrt.(1f0 ./ m).*1000f0).^0.25

# Load starting model (mlog - slowness built with Vs from logs; mvsp - built from VSP)
fid = h5open(model_file, "r")
n, d, o = read(fid, "n", "d", "o")
if haskey(fid, "mvsp")
    m0 = read(fid, "mvsp")
else
    m0 = read(fid, "m")
end
close(fid)

n = Tuple(Int64(i) for i in n)
d = Tuple(Float32(i) for i in d)
o = Tuple(Float32(i) for i in o)
if modeling_type == "scalar"
    model0 = Model(n, d, o, m0)
elseif modeling_type == "acoustic"
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
if modeling_type == "scalar"
    model0.m[:,1:seabed_ind] .= (1/vwater)^2
elseif modeling_type == "acoustic"
    model0.m[:,1:seabed_ind] .= (1/vwater)^2
    model0.b[:,1:seabed_ind] .= 1f0/rhowater
end

# Set up wavelet and source vector
src_geometry = Geometry(container; key = "source", segy_depth_key = segy_depth_key_src)

# sampling frequency
fs = 1000f0/src_geometry.dt[1]
# Nyquist frequency
f_nyq = fs/2f0

# setup wavelet
wavelet_raw = readdlm(wavelet_file, skipstart=wavelet_skip_start)
itp = LinearInterpolation(0:wavelet_dt:wavelet_dt*(length(wavelet_raw)-1), wavelet_raw[:,1], extrapolation_bc=0f0)
wavelet = Matrix{Float32}(undef,src_geometry.nt[1],1)
responsetype = Lowpass(frq*1000f0; fs=fs)
designmethod = Butterworth(8)
wavelet[:,1] = filt(digitalfilter(responsetype, designmethod), itp(0:src_geometry.dt[1]:src_geometry.t[1]))

q = judiVector(src_geometry, wavelet)

# t0 somehow can't be 0
Ml_ref = judiDataMute(q.geometry, d_obs.geometry, vp=1100f0, t0=0.001f0, mode=:reflection) # keep reflections
Ml_tur = judiDataMute(q.geometry, d_obs.geometry, vp=1100f0, t0=0.001f0, mode=:turning)    # keep turning waves

# Bound constraints
vmin = 1.2
vmax = 5.2
vBoundCoef = 0.3
# vminArr = ones(Float32, model0.n) .* vmin
# vmaxArr = ones(Float32, model0.n) .* vmax
vminArr = sqrt.(1f0 ./ model0.m.data) * (1f0-vBoundCoef)
vmaxArr = sqrt.(1f0 ./ model0.m.data) * (1f0+vBoundCoef)

# Slowness squared [s^2/km^2]
mmin = (1f0 ./ vmax).^2
mmax = (1f0 ./ vmin).^2
# mminArr = ones(Float32, model0.n) .* mmin
# mmaxArr = ones(Float32, model0.n) .* mmax
mminArr = vec(model0.m.data * (1f0-vBoundCoef))
mmaxArr = vec(model0.m.data * (1f0+vBoundCoef))

grad_mem = 40 # Based on n and CFL condition

mem = Sys.free_memory()/(1024^3)
t_sub = max(1, ceil(Int, nworkers()*grad_mem/mem))
@info "subsampling factor: $t_sub"

############# FWI #############
# JUDI options
global jopt = JUDI.Options(
    IC = "fwi",
    limit_m = true,
    buffer_size = buffer_size,
    optimal_checkpointing=false,
    # subsampling_factor=t_sub,     # subsampling_factor with space_order set leads to exception (probably bug)
    free_surface=true,  # free_surface is on to model multiples as well
    space_order=16)     # increase space order for > 12 Hz source wavelet

global modelCell = Vector{Array{Float32}}(undef, 0)
global gradientCell = Vector{Array{Float32}}(undef, 0)

# optimization parameters
niterations = 10
shot_from = 1
shot_to = length(d_obs)
shot_step = 3   # we may want to calculate only each Nth shot to economy time
count = 0
fhistory = Vector{Float32}(undef, 0)
mute_reflections = false
mute_turning = false

function save_results(x,z,m,grad,fhistory,v_plot_title,grad_plot_title,v_file,grad_file,h5_file)
    @info "save_results: $h5_file"
    n = (length(x),length(z))
    o = (x[1],z[1])
    d = (x[2]-x[1],z[2]-z[1])
    fwi_plt = Plots.heatmap(x, z, sqrt.(1f0 ./ adjoint(reshape(m,n))), c=:rainbow, 
        xlims=(x[1],x[end]), 
        ylims=(z[1],z[end]), yflip=true,
        title=v_plot_title,
        xlabel="Lateral position [km]",
        ylabel="Depth [km]",
        dpi=600)
    Plots.savefig(fwi_plt, v_file)

    grad_plt = Plots.heatmap(x, z, adjoint(reshape(grad,n)), c=:bluesreds, 
        xlims=(x[1],x[end]), 
        ylims=(z[1],z[end]), yflip=true,
        clim=(-maximum(grad)/5f0, maximum(grad)/5f0),
        title=grad_plot_title,
        xlabel="Lateral position [km]",
        ylabel="Depth [km]",
        dpi=600)
    Plots.savefig(grad_plt,grad_file)

    h5open(h5_file, "w") do file
        write(file, 
            "v", sqrt.(1f0 ./ reshape(m, n)), 
            "m", reshape(m, n), 
            "o", collect(o.*1000f0), 
            "n", collect(n), 
            "d", collect(d.*1000f0),
            "fhistory", fhistory,
            "grad", reshape(grad, n))
    end
end

# NLopt objective function
function nlopt_obj_fun!(m_update, grad)

    global x, z, count, jopt, gradientCell, seabed_ind;
    count += 1

    # Update model
    model0.m .= Float32.(reshape(m_update, model0.n))
    if modeling_type == "acoustic"
        model0.b .= Float32.(reshape(1f0./rho_from_slowness(model0.m), model0.n))
        model0.b[:,1:seabed_ind] .= 1f0/rhowater
    end

    # Select batch and calculate gradient
    # Subsampling the number of sources should in practice never be used for second order methods such as L-BFGS.
    indsrc = shot_from:shot_step:shot_to
    if mute_reflections
        fval, gradient = fwi_objective(model0, q[indsrc], Ml_tur[indsrc]*d_obs[indsrc], options=jopt)
    elseif mute_turning
        fval, gradient = fwi_objective(model0, q[indsrc], Ml_ref[indsrc]*d_obs[indsrc], options=jopt)
    else
        fval, gradient = fwi_objective(model0, q[indsrc], d_obs[indsrc], options=jopt)
    end
    gradient = reshape(gradient, model0.n)
    gradient[:, 1:seabed_ind] .= 0f0

    grad[1:end] .= gradient[1:end]

    push!(modelCell, copy(m_update))
    push!(gradientCell, copy(gradient))
    push!(fhistory, fval)

    println("iteration: ", count, "\tfval: ", fval, "\tnorm: ", norm(gradient))
    save_results(x,z,model0.m.data,gradient.data,fhistory,
                "FWI with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
                "Gradient with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
                dir_out * "FWI $count.png",
                dir_out * "Gradient $count.png",
                dir_out * model_file_out * " " * string(count) * ".h5")

    return convert(Float64, fval)
end

println("No.  ", "fval         ", "norm(gradient)")

opt = Opt(:LD_LBFGS, prod(model0.n))
min_objective!(opt, nlopt_obj_fun!)
lower_bounds!(opt, mmin); upper_bounds!(opt, mmax)
maxeval!(opt, niterations)
(minf, minx, ret) = optimize(opt, copy(model0.m))

h5open(dir_out * model_file_out * ".h5", "w") do file
    write(file, 
        "v", sqrt.(1f0 ./ reshape(modelCell[end], model0.n)), 
        "m", reshape(modelCell[end], model0.n), 
        "o", collect(o), 
        "n", collect(n), 
        "d", collect(d), 
        "fhistory", fhistory)
end