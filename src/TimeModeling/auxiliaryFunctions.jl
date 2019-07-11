# Auxiliary functions for TimeModeling module
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: September 2016
#

export ricker_wavelet, get_computational_nt, smooth10, damp_boundary, calculate_dt, setup_grid, setup_3D_grid
export convertToCell, limit_model_to_receiver_area, extend_gradient, remove_out_of_bounds_receivers
export time_resample, remove_padding, backtracking_linesearch, subsample
export generate_distribution, select_frequencies, process_physical_parameter
export load_pymodel, load_devito_jit, load_numpy, devito_model

function devito_model(model::Model, op, mode, options, dm)
    pm = load_pymodel()
    length(model.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)
    # Set up Python model structure
    if op=='J' && mode == 1
        modelPy = pm."Model"(origin=model.o, spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
            rho=process_physical_parameter(model.rho, dims), dm=process_physical_parameter(reshape(dm,model.n), dims), space_order=options.space_order)
    else
        modelPy = pm."Model"(origin=model.o, spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims), nbpml=model.nb,
            rho=process_physical_parameter(model.rho, dims), space_order=options.space_order)
    end
    return modelPy
end

function devito_model(model::Model_TTI, op, mode, options, dm)
    pm = load_pymodel()
    length(model.n) == 3 ? dims = [3,2,1] : dims = [2,1]   # model dimensions for Python are (z,y,x) and (z,x)
    # Set up Python model structure
    if op=='J' && mode == 1
        # Set up Python model structure (force origin to be zero due to current devito bug)
        modelPy = pm."Model"(origin=model.o, spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims),
                           rho=process_physical_parameter(model.rho, dims),
                           epsilon=process_physical_parameter(model.epsilon, dims),
                           delta=process_physical_parameter(model.delta, dims),
                           theta=process_physical_parameter(model.theta, dims),
                           phi=process_physical_parameter(model.phi, dims), nbpml=model.nb,
                           dm=process_physical_parameter(reshape(dm,model.n), dims),
                           space_order=options.space_order)
    else
        # Set up Python model structure (force origin to be zero due to current devito bug)
        modelPy = pm."Model"(origin=model.o, spacing=model.d, shape=model.n, vp=process_physical_parameter(sqrt.(1f0./model.m), dims),
                           rho=process_physical_parameter(model.rho, dims),
                           epsilon=process_physical_parameter(model.epsilon, dims),
                           delta=process_physical_parameter(model.delta, dims),
                           theta=process_physical_parameter(model.theta, dims),
                           phi=process_physical_parameter(model.phi, dims), nbpml=model.nb,
                           space_order=options.space_order)
    end
    return modelPy
end

function limit_model_to_receiver_area(srcGeometry::Geometry, recGeometry::Geometry, model::Model, buffer; pert=[])
    # Restrict full velocity model to area that contains either sources and receivers
    ndim = length(model.n)
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
        ox = Float32((nx_min - 1)*model.d[1])
        oz = model.o[2]
    else
        ny_min = Int(round(min_y/model.d[2])) + 1
        ny_max = Int(round(max_y/model.d[2])) + 1
        ox = Float32((nx_min - 1)*model.d[1])
        oy = Float32((ny_min - 1)*model.d[2])
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
    println("N old: ", model.n)
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

function limit_model_to_receiver_area(srcGeometry::Geometry,recGeometry::Geometry,model::Model_TTI,buffer;pert=[])
    # Restrict full velocity model to area that contains either sources and receivers
    ndim = length(model.n)
    # println("N orig: ", model.n)

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
    nx_min = Int(round((min_x - model.o[1])/model.d[1])) + 1
    nx_max = Int(round((max_x - model.o[1])/model.d[1])) + 1
    if ndim == 2
        ox = Float32((nx_min - 1)*model.d[1])
        oz = model.o[2]
    else
        ny_min = Int(round(min_y/model.d[2])) + 1
        ny_max = Int(round(max_y/model.d[2])) + 1
        ox = Float32((nx_min - 1)*model.d[1])
        oy = Float32((ny_min - 1)*model.d[2])
        oz = model.o[3]
    end

    # Extract relevant model part from full domain
    n_orig = model.n
    if ndim == 2
        model.m = model.m[nx_min: nx_max, :]
        typeof(model.epsilon) <: Array && (model.epsilon = model.epsilon[nx_min: nx_max, :])
        typeof(model.delta) <: Array && (model.delta = model.delta[nx_min: nx_max, :])
        typeof(model.theta) <: Array && (model.theta = model.theta[nx_min: nx_max, :])
        typeof(model.rho) <: Array && (model.rho = model.rho[nx_min: nx_max, :])
        model.o = (ox, oz)
    else
        model.m = model.m[nx_min:nx_max,ny_min:ny_max,:]
        typeof(model.epsilon) <: Array && (model.epsilon = model.epsilon[nx_min:nx_max,ny_min:ny_max,:])
        typeof(model.delta) <: Array && (model.delta = model.delta[nx_min:nx_max,ny_min:ny_max,:])
        typeof(model.theta) <: Array && (model.theta = model.theta[nx_min:nx_max,ny_min:ny_max,:])
        typeof(model.phi) <: Array && (model.phi = model.phi[nx_min:nx_max,ny_min:ny_max,:])
        typeof(model.rho) <: Array && (model.rho = model.rho[nx_min: nx_max, :])
        model.o = (ox, oy, oz)
    end
    model.n = size(model.m)
    # println("N new: ", model.n)
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

function extend_gradient(model_full::Modelall,model::Modelall,gradient::Array)
    # Extend gradient back to full model size
    ndim = length(model.n)
    full_gradient = zeros(Float32,model_full.n)
    nx_start = trunc(Int, Float32(Float32(model.o[1] - model_full.o[1])/model.d[1]) + 1)
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

function remove_out_of_bounds_receivers(recGeometry::Geometry, model::Modelall)

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

function remove_out_of_bounds_receivers(recGeometry::Geometry, recData::Array, model::Modelall)

    # Only keep receivers within the model
    xmin = model.o[1]
    if typeof(recGeometry.xloc[1]) <: Array
        idx_xrec = findall(x -> x >= xmin, recGeometry.xloc[1])
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

function load_devito_jit(modelPy)
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "Python"))
    if modelPy.is_tti
        # return pyimport("TTI_operators")
        return pyimport("TTI_Staggered")
    else
        return pyimport("JAcoustic_codegen")
    end
end

function calculate_dt(model::Model_TTI)
    if length(model.n) == 3
        coeff = 0.38
    else
        coeff = 0.42
    end
    scale = sqrt(maximum(1 .+ 2 *model.epsilon))
    return coeff * minimum(model.d) / (scale*sqrt(1/minimum(model.m)))
end

function calculate_dt(model::Model)
    if length(model.n) == 3
        coeff = 0.38
    else
        coeff = 0.42
    end
    return coeff * minimum(model.d) / (sqrt(1/minimum(model.m)))
end
"""
    get_computational_nt(srcGeometry, recGeoemtry, model)

Estimate the number of computational time steps. Required for calculating the dimensions\\
of the matrix-free linear modeling operators. `srcGeometry` and `recGeometry` are source\\
and receiver geometries of type `Geometry` and `model` is the model structure of type \\
`Model`.

"""
function get_computational_nt(srcGeometry, recGeometry, model::Modelall)
    # Determine number of computational time steps
    if typeof(srcGeometry) == GeometryOOC
        nsrc = length(srcGeometry.container)
    else
        nsrc = length(srcGeometry.xloc)
    end
    nt = Array{Any}(undef, nsrc)
    dtComp = calculate_dt(model)
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
    return source_coords
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

function resample_model(array, inh, modelfull)
    # size in
    shape = size(array)
    ndim = length(shape)
    if ndim > 2
        # Axes
        x1 = inh[1] * linspace(0, shape[1] - 1)
        xnew = modelfull.d[1] * linspace(0, modelfull.n[1] - 1)
        y1 = inh[2] * linspace(0, shape[2] - 1)
        ynew = modelfull.d[2] * linspace(0, modelfull.n[2] - 1)
        z1 = inh[3] * linspace(0, shape[3]-1)
        znew = modelfull.d[3] * linspace(0, modelfull.n[3] - 1)
        interpolator = interp.RegularGridInterpolator((x1, y1, z1), array)
        gridnew = np.ix_(xnew, ynew, znew)
    else
        # Axes
        x1 = inh[1] * linspace(0, shape[1] - 1)
        xnew = modelfull.d[1] * linspace(0, modelfull.n[1] - 1)
        z1 = inh[2] * linspace(0, shape[2]-1)
        znew = modelfull.d[2] * linspace(0, modelfull.n[2] - 1)
        interpolator = interp.RegularGridInterpolator((x1, z1), array)
        gridnew = np.ix_(xnew, znew)
    end
    resampled = pycall(interpolator,  Array{Float32, ndim}, gridnew)
    return resampled
end

function gs_residual_trace(maxshift, dtComp, dPredicted, dObserved, normalize)
	#shifts
	nshift = round(Int64, maxshift/dtComp)
	nSamples = 2*nshift + 1

	data_size = size(dPredicted)
	residual = zeros(Float32, data_size)
	dPredicted = [zeros(Float32, nshift, size(dPredicted,2)); dPredicted; zeros(Float32, nshift, size(dPredicted,2))]
	dObserved = [zeros(Float32, nshift, size(dPredicted,2)); dObserved; zeros(Float32, nshift, size(dPredicted,2))]
	# residual_plot = zeros(Float32, size(dObserved))
	# syn_plot = zeros(Float32, size(dObserved))
    indnz = [i for i in 1:size(dPredicted,2) if (norm(dObserved[:,i])>0 && norm(dPredicted[:,i])>0)]

    weights = [norm(dObserved[:,i]) for i in 1:size(dPredicted,2)]

	for rr in indnz
		aux = zeros(Float32, size(dPredicted, 1))

		syn = dPredicted[:, rr]
		obs = dObserved[:, rr]

		if normalize
			syn /= norm(syn)
			obs /= norm(obs)
		end
		H =zeros(Float32, length(1:5:nSamples),  length(1:5:nSamples));

		iloc=0
		for i = 1:5:nSamples
			shift = i - nshift - 1
			iloc +=1
			jloc=iloc
			dshift = circshift(syn, shift) - obs
			H[iloc, iloc] = dot(dshift, dshift)
			for j = (i+5):5:nSamples
				jloc+=1
				shift2 = j - nshift - 1
				circshift!(aux, syn, shift2)
				broadcast!(-, aux, aux, obs)
				H[iloc, jloc] = dot(dshift, vec(aux))
				H[jloc, iloc] = H[iloc, jloc]
			end
		end

		x = 1:5:nSamples
		y = 1:5:nSamples
		spl = Spline2D(x, y, H)
		x0 = 1:nSamples
		y0 = 1:nSamples
		H = evalgrid(spl, x0, y0)

		# get coefficients

		H = Array{Float64}(H)
		H = .5*(H'+ H)
		A = ones(Float32, 1, nSamples)
		sol = quadprog(0, H, A, '=', 1., 0., 1., IpoptSolver(print_level=1))
		alphas = Array{Float32}(sol.sol)
		# Data misfit
		for i = 1:2*nshift+1
			shift = i - nshift - 1
			residual[:, rr] += alphas[i] * circshift(circshift(syn, shift) - obs, -2*shift)[nshift+1:(end-nshift)]
			# residual_plot[:, rr] +=  alphas[i] * (circshift(syn, shift) - obs)
			# syn_plot[:, rr] +=  alphas[i] * circshift(syn, shift)
			#residual += alphas[i] * circshift(circshift(dPredicted, (0,shift)) - dObserved, (0, -2*shift))[:, nshift+1:(end-nshift)]
		end
		# if rr%50==0
		# 	figure();plot(syn_plot[rr,:], "-b");plot(syn, "-r");plot(obs, "-g")
		# end
        residual[:, rr] = weights[rr] * residual[:, rr]
	end
	# figure();imshow(syn_plot', vmin=-1e1, vmax=1e1, cmap="seismic", aspect=.2)
	# figure();imshow(dPredicted', vmin=-1e1, vmax=1e1, cmap="seismic", aspect=.2)
	# figure();imshow(dObserved', vmin=-1e1, vmax=1e1, cmap="seismic", aspect=.2)
	# figure();imshow(residual_plot', vmin=-1e1, vmax=1e1, cmap="seismic", aspect=.2)
	# figure();imshow(dPredicted' - dObserved', vmin=-1e1, vmax=1e1, cmap="seismic", aspect=.2)
	#
	return residual
end

function gs_residual_shot(maxshift, dtComp, dPredicted, dObserved, normalize)
	#shifts
	nshift = round(Int64, maxshift/dtComp)
	# println(nshift, " ", dtComp)
	nSamples = 2*nshift + 1

	data_size = size(dPredicted)
	residual = zeros(Float32, data_size)
	dPredicted = [zeros(Float32, nshift, size(dPredicted,2)); dPredicted; zeros(Float32, nshift, size(dPredicted,2))]
	dObserved = [zeros(Float32, nshift, size(dPredicted,2)); dObserved; zeros(Float32, nshift, size(dPredicted,2))]
	if normalize
		dPredicted /= norm(vec(dPredicted))
		dObserved /= norm(vec(dObserved))
	end
	aux = zeros(Float32, size(dPredicted))

	H =zeros(Float32, length(1:5:nSamples),  length(1:5:nSamples));

	iloc=0
	for i = 1:5:nSamples
		shift = i - nshift - 1
		iloc +=1
		jloc=iloc
		dshift = vec(circshift(dPredicted, (shift, 0)) - dObserved)
		H[iloc, iloc] = dot(dshift, dshift)
		for j = (i+5):5:nSamples
			jloc+=1
			shift2 = j - nshift - 1
			circshift!(aux, dPredicted, (shift2, 0))
			broadcast!(-, aux, aux, dObserved)
			H[iloc, jloc] = dot(dshift, vec(aux))
			H[jloc, iloc] = H[iloc, jloc]
		end
	end

	x = 1:5:nSamples
	y = 1:5:nSamples
	spl = Spline2D(x, y, H)
	x0 = 1:nSamples
	y0 = 1:nSamples
	H = evalgrid(spl, x0, y0)

	# get coefficientsn

	H = Array{Float64}(H)
	H = .5*(H'+H)
	A = ones(Float32, 1, nSamples)
	sol = quadprog(0, H, A, '=', 1., 0., 1., IpoptSolver(print_level=1))
	alphas = Array{Float32}(sol.sol)
	# Data misfit
	for i = 1:2*nshift+1
		shift = i - nshift - 1
		residual += alphas[i] * circshift(abs.(circshift(dPredicted, (shift, 0))).*(circshift(dPredicted, (shift, 0)) - dObserved), (-2*shift, 0))[nshift+1:(end-nshift), :]
	end

	return residual
end

function gs_residual(gs, dtComp, dPredicted, dObserved, normalize)
	if gs["strategy"] == "shot"
		residual = gs_residual_shot(gs["maxshift"], dtComp, dPredicted, dObserved, normalize)
	else
		residual = gs_residual_trace(gs["maxshift"], dtComp, dPredicted, dObserved, normalize)
	end
	return residual
end
