using DrWatson
@quickactivate "Viking"

using Viking
using Statistics, Random, LinearAlgebra, Interpolations, DelimitedFiles, Distributed
using JUDI, SlimOptim, NLopt, HDF5, SegyIO, Plots, ImageFiltering
using SetIntersectionProjection
@everywhere using JUDI.FFTW, Zygote, Flux

############# INITIAL DATA #############
modeling_type = "bulk"    # slowness, bulk
frq = 0.005 # kHz

prestk_dir = "$(@__DIR__)/trim_segy/"
# prestk_dir = "$(@__DIR__)/trim_segy_200_500/"
prestk_file = "shot"
dir_out = "$(@__DIR__)/fwi_spg_$(modeling_type)/$(frq)Hz/"
# dir_out = "$(@__DIR__)/fwi_spg_$(modeling_type)_buffer_3km/$(frq)Hz/"

model_file = "$(@__DIR__)/initial_model/model.h5"
# after each iteration you should set an appropriate model file computed with previous step
# model_file = "$(@__DIR__)/fwi_spg_$(modeling_type)/0.008Hz/model 10.h5"
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
niterations = 20
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

@everywhere function H(x)
    n = size(x, 1)
    σ = ifftshift(sign.(-n/2+1:n/2))
    y = imag(ifft(σ.*fft(x, 1), 1))
    return y
end

@everywhere envelope(x, y) = sum(abs2.((x - y) .+ 1im .* H(x - y)))
@everywhere denvelope(x, y) = Zygote.gradient(xs->envelope(xs, y), x)[1]
@everywhere myloss(x, y) = (envelope(x, y), denvelope(x, y))

@everywhere myloss(randn(Float32, 10, 10), randn(Float32, 10, 10))

# Optimization parameters
batchsize = 300

# Objective function for minConf library
count = 0
function objective_function(m_update)
    global x, z, count, jopt, seabed_ind;
    count += 1

    # Update model
    model0.m .= Float32.(reshape(m_update, size(model0)))
    if modeling_type == "bulk"
        model0.rho .= Float32.(reshape(rho_from_slowness(model0.m), size(model0)))
        model0.rho[:,1:seabed_ind] .= rhowater
    end

    # fwi function value and gradient
    indsrc = randperm(d_obs.nsrc)[1:batchsize]
    if mute_reflections
        fval, gradient = fwi_objective(model0, Mr_freq[indsrc]*q[indsrc], Ml_tur[indsrc]*Ml_freq[indsrc]*get_data(d_obs[indsrc]), options=jopt, misfit=myloss)
    elseif mute_turning
        fval, gradient = fwi_objective(model0, Mr_freq[indsrc]*q[indsrc], Ml_ref[indsrc]*Ml_freq[indsrc]*get_data(d_obs[indsrc]), options=jopt, misfit=myloss)
    else
        fval, gradient = fwi_objective(model0, Mr_freq[indsrc]*q[indsrc], Ml_freq[indsrc]*get_data(d_obs[indsrc]), options=jopt, misfit=myloss)
    end
    gradient = reshape(gradient, size(model0))
    gradient[:, 1:seabed_ind] .= 0f0
    gradient = .125f0*gradient/maximum(abs.(gradient))  # scale for line search

    push!(fhistory, fval)

    println("iteration: ", count, "\tfval: ", fval, "\tnorm: ", norm(gradient))
    save_data(x,z,adjoint(reshape(model0.m.data,size(model0))); 
            pltfile=dir_out * "FWI slowness $count.png",
            title="FWI slowness^2 with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
            colormap=:rainbow,
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="w",
            h5varname="m")
    save_data(x,z,sqrt.(1f0 ./ adjoint(reshape(model0.m.data,size(model0)))); 
            pltfile=dir_out * "FWI $count.png",
            title="FWI velocity with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
            colormap=:rainbow,
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="r+",
            h5varname="v")
    save_data(x,z,adjoint(reshape(gradient.data,size(model0))); 
            pltfile=dir_out * "Gradient $count.png",
            title="FWI gradient with L-BFGS $modeling_type: $(frq*1000)Hz, iter $count",
            clim=(-maximum(gradient.data)/5f0, maximum(gradient.data)/5f0),
            colormap=:bluesreds,
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="r+",
            h5varname="grad")
    save_fhistory(fhistory; 
            h5file=dir_out * model_file_out * " " * string(count) * ".h5",
            h5openflag="r+",
            h5varname="fhistory")

    return fval, gradient
end

# Bound projection
proj(x) = reshape(median([vec(mminArr) vec(x) vec(mmaxArr)]; dims=2), size(model0))

# FWI with SPG
options = spg_options(verbose=3, maxIter=niterations, memory=3)
sol = spg(objective_function, model0.m.data, proj, options)