# Auxiliary functions for TimeModeling module
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: September 2016
#
using PyPlot

export ricker_wavelet, get_computational_nt, smooth10, damp_boundary, calculate_dt, setup_grid, setup_3D_grid
export convertToCell, limit_model_to_receiver_area, extend_gradient
export time_resample, remove_padding, backtracking_linesearch
export generate_distribution, select_frequencies, process_physical_parameter

function limit_model_to_receiver_area(srcGeometry::Geometry,recGeometry::Geometry,model::Model,buffer;pert=[])
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
    min_x = max(model.o[1],min_x-buffer)
    max_x = min(model.o[1] + model.d[1]*(model.n[1]-1),max_x+buffer)
    if ndim == 3
        min_y = max(model.o[2],min_y-buffer)
        max_y = min(model.o[2] + model.d[2]*(model.n[2]-1),max_y+buffer)
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
        model.rho = model.rho[nx_min: nx_max, :]
        model.o = (ox, oz)
    else
        model.m = model.m[nx_min:nx_max,ny_min:ny_max,:]
        model.rho = model.rho[nx_min:nx_max,ny_min:ny_max,:]
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

"""
    convertToCell(x)

Convert an array `x` to a cell array (`Array{Any,1}`) with `length(x)` entries,\\
where the i-th cell contains the i-th entry of `x`.

"""
function convertToCell(x)
    n = length(x)
    y = Array{Any}(n)
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
    t = linspace(t0,tmax,nt)
    r = (pi * f0 * (t - 1 / f0))
    q = zeros(Float32,nt,1)
    q[:,1] = (1. - 2.*r.^2.).*exp.(-r.^2.)
    return q
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
function get_computational_nt(srcGeometry, recGeometry, model::Model)
    # Determine number of computational time steps
    if typeof(srcGeometry) == GeometryOOC
        nsrc = length(srcGeometry.container)
    else
        nsrc = length(srcGeometry.xloc)
    end
    nt = Array{Any}(nsrc)
    dtComp = calculate_dt(model)
    for j=1:nsrc
        ntRec = Int(trunc(recGeometry.dt[j]*(recGeometry.nt[j]-1))) / dtComp
        ntSrc = Int(trunc(srcGeometry.dt[j]*(srcGeometry.nt[j]-1))) / dtComp
        nt[j] = max(Int(trunc(ntRec)), Int(trunc(ntSrc)))
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
    xloc = Array{Any}(nsrc)
    yloc = Array{Any}(nsrc)
    zloc = Array{Any}(nsrc)
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
# Take input 1d x and y coordinate vectors and generate 3d grid. Input are arrays/linspace
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
            out[:,:,a] = sum(velocity[:,:,a-4:a+5],3) / 10
        end
    else
        out[:,:] = velocity[:,:]
        for a=5:nz-6
            out[:,a] = sum(velocity[:,a-4:a+5],2) / 10
        end
    end
    return out
end

function remove_padding(gradient::Array, nb::Integer; true_adjoint::Bool=false)
    if ndims(gradient) == 2
        if true_adjoint
            gradient[nb+1,:] = sum(gradient[1:nb,:],1)
            gradient[end-nb,:] = sum(gradient[end-nb+1:end,:],1)
            gradient[:,nb+1] = sum(gradient[:,1:nb],2)
            gradient[:,end-nb] = sum(gradient[:,end-nb+1:end],2)
        end
        return gradient[nb+1:end-nb,nb+1:end-nb]
    elseif ndims(gradient)==3
        if true_adjoint
            gradient[nb+1,:,:] = sum(gradient[1:nb,:,:],1)
            gradient[end-nb,:,:] = sum(gradient[end-nb+1:end,:,:],1)
            gradient[:,nb+1,:] = sum(gradient[:,1:nb,:],2)
            gradient[:,end-nb,:] = sum(gradient[:,end-nb+1:end,:],2)
            gradient[:,:,nb+1] = sum(gradient[:,:,1:nb],3)
            gradient[:,:,end-nb] = sum(gradient[:,:,end-nb+1:end],3)
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

function time_resample(data::Array,dt_in, geometry_out::Geometry;order=2)
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

	# invert probability distribution (interpolate)
	axis = Array{Float64,1}[]
	push!(axis,pd)
	L = Lininterp(convert(Array{Float64,1},f), axis)

	return L
end

function select_frequencies(L;fmin=0.,fmax=Inf,nf=1)
	freq = zeros(Float32,nf)
	for j=1:nf
		while (freq[j] <= fmin) || (freq[j] > fmax)
			freq[j] = getValue(L,rand(1)[1])[1]
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




function gs_residual_trace(maxshift, dtComp, dPredicted, dObserved, normalize)
	#shifts
	nshift = round(Int64, maxshift/dtComp)
	nSamples = 2*nshift + 1

	data_size = size(dPredicted)
	residual = zeros(Float32, data_size)
	dPredicted = [zeros(Float32, size(dPredicted,1), nshift) dPredicted zeros(Float32, size(dPredicted,1), nshift)]
	dObserved = [zeros(Float32, size(dObserved,1), nshift) dObserved zeros(Float32, size(dObserved,1), nshift)]
	residual_plot = zeros(Float32, size(dObserved))
	syn_plot = zeros(Float32, size(dObserved))

	for rr=1:size(dPredicted,1)
		aux = zeros(Float32, size(dPredicted, 2))

		syn = dPredicted[rr,:]
		obs = dObserved[rr,:]
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
		H = .5*(H'+H)
		figure();imshow(H)
		A = ones(Float32, 1, nSamples)
		sol = quadprog(0, H, A, '=', 1., 0., 1., IpoptSolver(print_level=1))
		alphas = Array{Float32}(sol.sol)
		# Data misfit
		for i = 1:2*nshift+1
			shift = i - nshift - 1
			residual[rr, :] += alphas[i] * circshift(circshift(syn, shift) - obs, -2*shift)[nshift+1:(end-nshift)]
			residual_plot[rr, :] +=  alphas[i] * (circshift(syn, shift) - obs)
			syn_plot[rr, :] +=  alphas[i] * circshift(syn, shift)
			#residual += alphas[i] * circshift(circshift(dPredicted, (0,shift)) - dObserved, (0, -2*shift))[:, nshift+1:(end-nshift)]
		end
		# if rr%50==0
		# 	figure();plot(syn_plot[rr,:], "-b");plot(syn, "-r");plot(obs, "-g")
		# end
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
	dPredicted = [zeros(Float32, size(dPredicted,1), nshift) dPredicted zeros(Float32, size(dPredicted,1), nshift)]
	dObserved = [zeros(Float32, size(dObserved,1), nshift) dObserved zeros(Float32, size(dObserved,1), nshift)]
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
		dshift = vec(circshift(dPredicted, (0,shift)) - dObserved)
		H[iloc, iloc] = dot(dshift, dshift)
		for j = (i+5):5:nSamples
			jloc+=1
			shift2 = j - nshift - 1
			circshift!(aux, dPredicted, (0,shift2))
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

	# get coefficients

	H = Array{Float64}(H)
	H = .5*(H'+H)
	A = ones(Float32, 1, nSamples)
	sol = quadprog(0, H, A, '=', 1., 0., 1., IpoptSolver(print_level=1))
	alphas = Array{Float32}(sol.sol)
	# Data misfit
	for i = 1:2*nshift+1
		shift = i - nshift - 1
		residual += alphas[i] * circshift(abs.(circshift(dPredicted, (0,shift))).*(circshift(dPredicted, (0,shift)) - dObserved), (0, -2*shift))[:, nshift+1:(end-nshift)]
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
