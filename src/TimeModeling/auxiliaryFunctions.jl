# Auxiliary functions for TimeModeling module
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: September 2016
#

export ricker_wavelet, get_computational_nt, smooth10, damp_boundary, calculate_dt, setup_grid, setup_3D_grid
export convertToCell, limit_model_to_receiver_area, extend_gradient, remove_out_of_bounds_receivers
export time_resample, remove_padding, backtracking_linesearch, subsample
export generate_distribution, select_frequencies, process_physical_parameter
export load_pymodel, load_acoustic_codegen, load_numpy

function limit_model_to_receiver_area(srcGeometry::Geometry, recGeometry::Geometry, model::Model, buffer; pert=[])
    # Restrict full velocity model to area that contains either sources and receivers
    ndim = length(model.n)
    println("N orig: ", model.n)

    # scan for minimum and maximum x and y source/receiver coordinates
    min_x = minimum([vec(recGeometry.xloc[1]); vec(srcGeometry.xloc[1])])
    max_x = maximum([vec(recGeometry.xloc[1]); vec(srcGeometry.xloc[1])])
    if ndim == 3
        min_y = minimum([vec(recGeometry.yloc[1]); vec(srcGeometry.yloc[1])])
        max_y = maximum([vec(recGeometry.yloc[1]); vec(srcGeometry.yloc[1])])
    end

    # add buffer zone if possible
    min_x = max(model.o[1], min_x-buffer)
    max_x = min(model.o[1] + model.d[1]*(model.n[1]-1), max_x+buffer)
    if ndim == 3
        min_y = max(model.o[2], min_y-buffer)
        max_y = min(model.o[2] + model.d[2]*(model.n[2]-1), max_y+buffer)
    end

    # extract part of the model that contains sources/receivers
    nx_min = Int(round(min_x/model.d[1])) + 1
    nx_max = Int(round(max_x/model.d[1])) + 1
    if ndim == 2
        ox = (nx_min - 1)*model.d[1]
        oz = model.o[2]
    else
        ny_min = Int(round(min_y/model.d[2])) + 1
        ny_max = Int(round(max_y/model.d[2])) + 1
        ox = (nx_min - 1)*model.d[1]
        oy = (ny_min - 1)*model.d[2]
        oz = model.o[3]
    end

    # Extract relevant model part from full domain
    n_orig = model.n
    if ndim == 2
        model.m = model.m[nx_min: nx_max, :]
        typeof(model.rho) <: Array && (model.rho = model.rho[nx_min: nx_max, :])
        model.o = (ox, oz)
    else
        model.m = model.m[nx_min:nx_max,ny_min:ny_max,:]
        typeof(model.rho) <: Array && (model.rho = model.rho[nx_min:nx_max,ny_min:ny_max,:])
        model.o = (ox,oy,oz)
    end
    model.n = size(model.m)
    println("N new: ", model.n)
    if isempty(pert)
        return model
    else
        if ndim==2
            pert = reshape(pert,n_orig)[nx_min: nx_max, :]
        else
            pert = reshape(pert,n_orig)[nx_min: nx_max,ny_min: ny_max, :]
        end
        return model,vec(pert)
    end
end

function extend_gradient(model_full::Model,model::Model,gradient::Array)
    # Extend gradient back to full model size
    ndim = length(model.n)
    full_gradient = zeros(Float32,model_full.n)
    nx_start = Int((model.o[1] - model_full.o[1])/model.d[1] + 1)
    nx_end = nx_start + model.n[1] - 1
    if ndim == 2
        full_gradient[nx_start:nx_end,:] = gradient
    else
        ny_start = Int((model.o[2] - model_full.o[2])/model.d[2] + 1)
        ny_end = ny_start + model.n[2] - 1
        full_gradient[nx_start:nx_end,ny_start:ny_end,:] = gradient
    end
    return full_gradient
end

function remove_out_of_bounds_receivers(recGeometry::Geometry, model::Model)

    # Only keep receivers within the model
    xmin = model.o[1]
    if typeof(recGeometry.xloc[1]) <: Array
        idx_xrec = findall(x -> x >= xmin, recGeometry.xloc[1])
        recGeometry.xloc[1] = recGeometry.xloc[1][idx_xrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_xrec]
    end

    # For 3D shot records, scan also y-receivers
    if length(model.n) == 3 && typeof(recGeometry.yloc[1]) <: Array
        ymin = model.o[2]
        idx_yrec = findall(x -> x >= ymin, recGeometry.yloc[1])
        recGeometry.yloc[1] = recGeometry.yloc[1][idx_yrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_yrec]
    end
    return recGeometry
end

function remove_out_of_bounds_receivers(recGeometry::Geometry, recData::Array, model::Model)

    # Only keep receivers within the model
    xmin = model.o[1]
    if typeof(recGeometry.xloc[1]) <: Array
        idx_xrec = findall(x -> x > xmin, recGeometry.xloc[1])
        recGeometry.xloc[1] = recGeometry.xloc[1][idx_xrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_xrec]
        recData[1] = recData[1][:, idx_xrec]
    end

    # For 3D shot records, scan also y-receivers
    if length(model.n) == 3 && typeof(recGeometry.yloc[1]) <: Array
        ymin = model.o[2]
        idx_yrec = findall(x -> x > ymin, recGeometry.yloc[1])
        recGeometry.yloc[1] = recGeometry.yloc[1][idx_yrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_yrec]
        recData[1] = recData[1][:, idx_yrec]
    end
    return recGeometry, recData
end

"""
    convertToCell(x)

Convert an array `x` to a cell array (`Array{Any,1}`) with `length(x)` entries,\\
where the i-th cell contains the i-th entry of `x`.

"""
function convertToCell(x)
    n = length(x)
    y = Array{Any}(undef, n)
    for j=1:n
        y[j] = x[j]
    end
    return y
end

# 1D source time function
"""
    source(tmax, dt, f0)

Create seismic Ricker wavelet of length `tmax` (in milliseconds) with sampling interval `dt` (in milliseonds)\\
and central frequency `f0` (in kHz).

"""
function ricker_wavelet(tmax, dt, f0)
    t0 = 0.
    nt = Int(trunc((tmax - t0)/dt + 1))
    t = range(t0,stop=tmax,length=nt)
    r = (pi * f0 * (t .- 1 / f0))
    q = zeros(Float32,nt,1)
    q[:,1] = (1f0 .- 2f0 .* r.^2f0) .* exp.(-r.^2f0)
    return q
end

function load_pymodel()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "Python"))
    return pyimport("PyModel")
end

function load_numpy()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "Python"))
    return pyimport("numpy")
end

function load_acoustic_codegen()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "Python"))
    return pyimport("JAcoustic_codegen")
end

function calculate_dt(n,d,o,v,rho; epsilon=0)
    pm = load_pymodel()
    length(n) == 2 ? pyDim = [n[2], n[1]] : pyDim = [n[3],n[2],n[1]]
    modelPy = pm."Model"(o, d, pyDim, PyReverseDims(v))
    dtComp = modelPy.critical_dt
end

"""
    get_computational_nt(srcGeometry, recGeoemtry, model)

Estimate the number of computational time steps. Required for calculating the dimensions\\
of the matrix-free linear modeling operators. `srcGeometry` and `recGeometry` are source\\
and receiver geometries of type `Geometry` and `model` is the model structure of type \\
`Model`.

"""
function get_computational_nt(srcGeometry, recGeometry, model::Model)
    # Determine number of computational time steps
    if typeof(srcGeometry) == GeometryOOC
        nsrc = length(srcGeometry.container)
    else
        nsrc = length(srcGeometry.xloc)
    end
    nt = Array{Any}(undef, nsrc)
    dtComp = calculate_dt(model.n, model.d, model.o, sqrt.(1f0 ./ model.m), model.rho)
    for j=1:nsrc
        ntRec = recGeometry.dt[j]*(recGeometry.nt[j]-1) / dtComp
        ntSrc = srcGeometry.dt[j]*(srcGeometry.nt[j]-1) / dtComp
        nt[j] = max(Int(ceil(ntRec)), Int(ceil(ntSrc)))
    end
    return nt
end

function setup_grid(geometry,n, origin)
    # 3D grid
    if length(n)==3
        if length(geometry.xloc[1]) > 1
            source_coords = Array{Float32,2}([vec(geometry.xloc[1]) vec(geometry.yloc[1]) vec(geometry.zloc[1])])
        else
            source_coords = Array{Float32,2}([geometry.xloc[1] geometry.yloc[1] geometry.zloc[1]])
        end
        orig = Array{Float32}([origin[1] origin[2] origin[3]])
    else
    # 2D grid
        if length(geometry.xloc[1]) > 1
            source_coords = Array{Float32,2}([vec(geometry.xloc[1]) vec(geometry.zloc[1])])
        else
            source_coords = Array{Float32,2}([geometry.xloc[1] geometry.zloc[1]])
        end
        orig = Array{Float32}([origin[1] origin[2]])
    end
    return source_coords .- orig
end

function setup_3D_grid(xrec::Array{Any,1},yrec::Array{Any,1},zrec::Array{Any,1})
    # Take input 1d x and y coordinate vectors and generate 3d grid. Input are cell arrays
    nsrc = length(xrec)
    xloc = Array{Any}(undef, nsrc)
    yloc = Array{Any}(unfef, nsrc)
    zloc = Array{Any}(undef, nsrc)
    for i=1:nsrc
        nxrec = length(xrec[i])
        nyrec = length(yrec[i])

        xloc[i] = zeros(nxrec*nyrec)
        yloc[i] = zeros(nxrec*nyrec)
        zloc[i] = zeros(nxrec*nyrec)

        idx = 1

        for k=1:nyrec
            for j=1:nxrec
                xloc[i][idx] = xrec[i][j]
                yloc[i][idx] = yrec[i][k]
                zloc[i][idx] = zrec[i]
                idx += 1
            end
        end
    end
    return xloc, yloc, zloc
end

function setup_3D_grid(xrec,yrec,zrec)
# Take input 1d x and y coordinate vectors and generate 3d grid. Input are arrays/ranges
    nxrec = length(xrec)
    nyrec = length(yrec)

    xloc = zeros(nxrec*nyrec)
    yloc = zeros(nxrec*nyrec)
    zloc = zeros(nxrec*nyrec)
    idx = 1
    for k=1:nyrec
        for j=1:nxrec
            xloc[idx] = xrec[j]
            yloc[idx] = yrec[k]
            zloc[idx] = zrec
            idx += 1
        end
    end
    return xloc, yloc, zloc
end

function smooth10(velocity,shape)
    # 10 point smoothing function
    out = ones(Float32,shape)
    nz = shape[end]
    if length(shape)==3
        out[:,:,:] = velocity[:,:,:]
        for a=5:nz-6
            out[:,:,a] = sum(velocity[:,:,a-4:a+5], dims=3) / 10
        end
    else
        out[:,:] = velocity[:,:]
        for a=5:nz-6
            out[:,a] = sum(velocity[:,a-4:a+5], dims=2) / 10
        end
    end
    return out
end

function remove_padding(gradient::Array, nb::Integer; true_adjoint::Bool=false)
    if ndims(gradient) == 2
        if true_adjoint
            gradient[nb+1,:] = sum(gradient[1:nb,:], dims=1)
            gradient[end-nb,:] = sum(gradient[end-nb+1:end,:], dims=1)
            gradient[:,nb+1] = sum(gradient[:,1:nb], dims=2)
            gradient[:,end-nb] = sum(gradient[:,end-nb+1:end], dims=2)
        end
        return gradient[nb+1:end-nb,nb+1:end-nb]
    elseif ndims(gradient)==3
        if true_adjoint
            gradient[nb+1,:,:] = sum(gradient[1:nb,:,:], dims=1)
            gradient[end-nb,:,:] = sum(gradient[end-nb+1:end,:,:], dims=1)
            gradient[:,nb+1,:] = sum(gradient[:,1:nb,:], dims=2)
            gradient[:,end-nb,:] = sum(gradient[:,end-nb+1:end,:], dims=2)
            gradient[:,:,nb+1] = sum(gradient[:,:,1:nb], dims=3)
            gradient[:,:,end-nb] = sum(gradient[:,:,end-nb+1:end], dims=3)
        end
        return gradient[nb+1:end-nb,nb+1:end-nb,nb+1:end-nb]
    else
        error("Gradient must have 2 or 3 dimensions")
    end
end

# Vectorization of single variable (not defined in Julia)
vec(x::Float64) = x;
vec(x::Float32) = x;
vec(x::Int64) = x;
vec(x::Int32) = x;


function time_resample(data::Array,geometry_in::Geometry,dt_new;order=2)

    if dt_new==geometry_in.dt[1]
        return data, geometry_in
    else
        geometry = deepcopy(geometry_in)
        numTraces = size(data,2)
        timeAxis = 0:geometry.dt[1]:geometry.t[1]
        timeInterp = 0:dt_new:geometry.t[1]
        dataInterp = zeros(Float32,length(timeInterp),numTraces)
        for k=1:numTraces
            spl = Spline1D(timeAxis,data[:,k];k=order)
            dataInterp[:,k] = spl(timeInterp)
        end
        geometry.dt[1] = dt_new
        geometry.nt[1] = length(timeInterp)
        geometry.t[1] = (geometry.nt[1] - 1)*geometry.dt[1]
        return dataInterp, geometry
    end
end

function time_resample(data::Array,dt_in, geometry_out::Geometry;order=2)

    if dt_in==geometry_out.dt[1]
        return data
    else
        geometry = deepcopy(geometry_out)
        numTraces = size(data,2)
        timeAxis = 0:dt_in:geometry_out.t[1]
        timeInterp = 0:geometry_out.dt[1]:geometry_out.t[1]
        dataInterp = zeros(Float32,length(timeInterp),numTraces)
        for k=1:numTraces
            spl = Spline1D(timeAxis,data[:,k];k=order)
            dataInterp[:,k] = spl(timeInterp)
        end
        return dataInterp
    end
end

#subsample(x::Nothing) = x

function generate_distribution(x; src_no=1)
	# Generate interpolator to sample from probability distribution given
	# from spectrum of the input data

	# sampling information
	nt = x.geometry.nt[src_no]
	dt = x.geometry.dt[src_no]
	t = x.geometry.t[src_no]

	# frequencies
	fs = 1/dt	# sampling rate
	fnyq = fs/2	# nyquist frequency
	df = fnyq/nt	# frequency interval
	f = 0:2*df:fnyq	# frequencies

	# amplitude spectrum of data (serves as probability density function)
	ns = convert(Integer,ceil(nt/2))
	amp = abs.(fft(x.data[src_no]))[1:ns]	# get first half of spectrum

	# convert to cumulative probability distribution (integrate)
	pd = zeros(ns)
	pd[1] = dt*amp[1]
	for j=2:ns
		pd[j] = pd[j-1] + amp[j]*df
	end
	pd /= pd[end]	# normalize

	return Spline1D(pd, f)
end

function select_frequencies(q_dist; fmin=0f0, fmax=Inf, nf=1)
	freq = zeros(Float32, nf)
	for j=1:nf
		while (freq[j] <= fmin) || (freq[j] > fmax)
			freq[j] = q_dist(rand(1)[1])[1]
		end
	end
	return freq
end

function process_physical_parameter(param, dims)
    if length(param) ==1
        return param
    else
        return PyReverseDims(permutedims(param, dims))
    end
end
