# Auxiliary functions for TimeModeling module
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: September 2016
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

export ricker_wavelet, get_computational_nt, calculate_dt, setup_grid, setup_3D_grid
export convertToCell, limit_model_to_receiver_area, extend_gradient, remove_out_of_bounds_receivers
export time_resample, remove_padding, subsample, process_input_data
export generate_distribution, select_frequencies
export load_pymodel, load_devito_jit, load_numpy, devito_model
export update_dm, pad_sizes, pad_array
export transducer


function update_dm(model::PyObject, dm, options)
    model.dm = pad_array(dm.data, pad_sizes(model, options))
end

function pad_sizes(model, options; so=nothing)
    isnothing(so) && (so = options.space_order)
    try
        return [(nbl + so, nbr + so) for (nbl, nbr)=model.padsizes]
    catch e
        padsizes = [(model.nb + so, model.nb + so) for i=1:length(model.n)]
        if options.free_surface
            padsizes[end] = (so, model.nb + so)
        end
        return padsizes
    end
end

function devito_model(model::Model, options)
    pm = load_pymodel()
    pad = pad_sizes(model, options)
    # Set up Python model structure
    m = pad_array(model[:m].data, pad)
    physpar = Dict((n, pad_array(v.data, pad)) for (n, v) in model.params if n != :m)
    modelPy = pm."Model"(model.o, model.d, model.n, m, fs=options.free_surface,
                         nbl=model.nb, space_order=options.space_order, dt=options.dt_comp;
                         physpar...)
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
        for (p, v) in model.params
            typeof(v) <: AbstractArray && (model.params[p] = PhysicalParameter(v.data[nx_min: nx_max, :],
                                                                               model.d, (ox, oz)))
        end
        model.o = (ox, oz)
    else
        for (p, v) in model.params
            typeof(v) <: AbstractArray && (model.params[p] = PhysicalParameter(v.data[nx_min:nx_max,ny_min:ny_max,:],
                                                                       model.d, (ox, oy, oz)))
        end
        model.o = (ox,oy,oz)
    end

    println("N old $(model.n)")
    model.n = model.m.n
    println("N new $(model.n)")
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

function extend_gradient(model_full::Model, model::Model, gradient::PhysicalParameter)
    # Extend gradient back to full model size
    ndim = length(model.n)
    full_gradient = zeros(Float32, model_full.n)
    nx_start = trunc(Int, Float32(Float32(model.o[1] - model_full.o[1])/model.d[1]) + 1)
    nx_end = nx_start + model.n[1] - 1
    if ndim == 2
        full_gradient[nx_start:nx_end,:] = gradient.data
    else
        ny_start = Int((model.o[2] - model_full.o[2])/model.d[2] + 1)
        ny_end = ny_start + model.n[2] - 1
        full_gradient[nx_start:nx_end,ny_start:ny_end,:] = gradient.data
    end
    return PhysicalParameter(full_gradient, model.d, model_full.o)
end

function extend_gradient(model_full::Model,model::Model, gradient::Array)
    # Extend gradient back to full model size
    ndim = length(model.n)
    full_gradient = zeros(Float32, model_full.n)
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
function ricker_wavelet(tmax, dt, f0; t0=nothing)
    R = typeof(dt)
    isnothing(t0) ? t0 = R(0) : tmax = R(tmax - t0)
    nt = trunc(Int64, tmax / dt) + 1
    t = range(t0, stop=tmax, length=nt)
    r = (pi * f0 * (t .- 1 / f0))
    q = zeros(Float32,nt,1)
    q[:,1] = (1f0 .- 2f0 .* r.^2f0) .* exp.(-r.^2f0)
    return q
end

function load_pymodel()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    return pyimport("models")
end

function load_devito_jit()
    pushfirst!(PyVector(pyimport("sys")."path"), joinpath(JUDIPATH, "pysource"))
    return pyimport("interface")
end

function calculate_dt(model::Model; dt=nothing)
    if ~isnothing(dt)
        return dt
    end
    pm = load_pymodel()
    m = minimum(model[:m])
    epsilon = maximum(get(model.params, :epsilon, 0))
    modelPy = pm."Model"(origin=model.o, spacing=model.d, shape=model.n,
                         m=m, epsilon=epsilon, nbl=0)
    return convert(Float32, modelPy.critical_dt)
end

"""
    get_computational_nt(srcGeometry, recGeoemtry, model; dt=nothing)

Estimate the number of computational time steps. Required for calculating the dimensions\\
of the matrix-free linear modeling operators. `srcGeometry` and `recGeometry` are source\\
and receiver geometries of type `Geometry` and `model` is the model structure of type \\
`Model`.

"""
function get_computational_nt(srcGeometry, recGeometry, model::Model; dt=nothing)
    # Determine number of computational time steps
    if typeof(srcGeometry) == GeometryOOC
        nsrc = length(srcGeometry.container)
    else
        nsrc = length(srcGeometry.xloc)
    end
    nt = Array{Any}(undef, nsrc)
    dtComp = calculate_dt(model; dt=dt)
    for j=1:nsrc
        ntRec = trunc(Int64, recGeometry.dt[j]*(recGeometry.nt[j]-1) / dtComp) + 1
        ntSrc = trunc(Int64, srcGeometry.dt[j]*(srcGeometry.nt[j]-1) / dtComp) + 1
        nt[j] = max(ntRec, ntSrc)
    end
    return nt
end

function get_computational_nt(Geometry, model::Model; dt=nothing)
    # Determine number of computational time steps
    if typeof(Geometry) == GeometryOOC
        nsrc = length(Geometry.container)
    else
        nsrc = length(Geometry.xloc)
    end
    nt = Array{Any}(undef, nsrc)
    dtComp = calculate_dt(model; dt=dt)
    for j=1:nsrc
        nt[j] = trunc(Int64, Geometry.dt[j]*(Geometry.nt[j]-1) / dtComp) + 1
    end
    return nt
end


function setup_grid(geometry, n)
    # 3D grid
    if length(n)==3
        if length(geometry.xloc[1]) > 1
            source_coords = Array{Float32,2}([vec(geometry.xloc[1]) vec(geometry.yloc[1]) vec(geometry.zloc[1])])
        else
            source_coords = Array{Float32,2}([geometry.xloc[1] geometry.yloc[1] geometry.zloc[1]])
        end
    else
    # 2D grid
        if length(geometry.xloc[1]) > 1
            source_coords = Array{Float32,2}([vec(geometry.xloc[1]) vec(geometry.zloc[1])])
        else
            source_coords = Array{Float32,2}([geometry.xloc[1] geometry.zloc[1]])
        end
    end
    return source_coords
end

function setup_3D_grid(xrec::Array{Any,1},yrec::Array{Any,1},zrec::Array{Any,1})
    # Take input 1d x and y coordinate vectors and generate 3d grid. Input are cell arrays
    nsrc = length(xrec)
    xloc = Array{Any}(undef, nsrc)
    yloc = Array{Any}(undef, nsrc)
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

pad_array(m::Number, nb::Array{Tuple{Int64,Int64},1}) = m

function pad_array(m::Array{DT}, nb::Array{Tuple{Int64,Int64},1}; mode::Symbol=:border) where {DT}
    n = size(m)
    new_size = Tuple([n[i] + sum(nb[i]) for i=1:length(nb)])
    Ei = []
    for i=length(nb):-1:1
        left, right = nb[i]
        push!(Ei, joExtend(n[i], mode;pad_upper=right, pad_lower=left, RDT=DT, DDT=DT))
    end
    padded = joKron(Ei...) * vec(m)
    return reshape(padded, new_size)
end

function remove_padding(gradient::Array{DT}, nb::Array{Tuple{Int64,Int64},1}; true_adjoint::Bool=false) where {DT}
    # Pad is applied x then y then z, so sum must be done in reverse order x then y then x
    true_adjoint ? mode = :border : mode = :zeros
    n = size(gradient)
    new_size = Tuple([n[i] - sum(nb[i]) for i=1:length(nb)])
    Ei = []
    for i=length(nb):-1:1
        left, right = nb[i]
        push!(Ei, joExtend(new_size[i], mode;pad_upper=right, pad_lower=left, RDT=DT, DDT=DT))
    end
    gradient = reshape(joKron(Ei...)' * vec(gradient), new_size)
    return collect(Float32, gradient)
end

# Vectorization of single variable (not defined in Julia)
vec(x::Float64) = x;
vec(x::Float32) = x;
vec(x::Int64) = x;
vec(x::Int32) = x;


function time_resample(data::Array, geometry_in::Geometry, dt_new;order=2)

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

function time_resample(data::Array, dt_in, geometry_out::Geometry;order=2)

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

process_input_data(input::judiVector, geometry::Geometry, info::Info) = input.data
process_input_data(input::judiWeights, model::Model, info::Info) = input.weights

function process_input_data(input::Array{Float32}, geometry::Geometry, info::Info)
    # Input data is pure Julia array: assume fixed no.
    # of receivers and reshape into data cube nt x nrec x nsrc
    nt = Int(geometry.nt[1])
    nrec = length(geometry.xloc[1])
    nsrc = info.nsrc
    data = reshape(input, nt, nrec, nsrc)
    dataCell = Array{Array}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = data[:,:,j]
    end
    return dataCell
end

function process_input_data(input::Array{Float32}, model::Model, info::Info)
    ndims = length(model.n)
    dataCell = Array{Array}(undef, info.nsrc)
    if ndims == 2
        input = reshape(input, model.n[1], model.n[2], info.nsrc)
        for j=1:info.nsrc
            dataCell[j] = input[:,:,j]
        end
    elseif ndims == 3
        input = reshape(input, model.n[1], model.n[2], model.n[3], info.nsrc)
        for j=1:info.nsrc
            dataCell[j] = input[:,:,:,j]
        end
    else
        throw("Number of dimensions not supported.")
    end
    return dataCell
end


function reshape(x::Array{Float32, 1}, geometry::Geometry)
    nt = geometry.nt[1]
    nrec = length(geometry.xloc[1])
    nsrc = Int(length(x) / nt / nrec)
    return reshape(x, nt, nrec, nsrc)
end


"""
    transducer(q, d, r, theta)

Create the JUDI soure for a circular transducer
Theta=0 points downward:

. . . . - - - . . . . . .

. . . . + + + . . . . . .

. . . . . . . . . . . . .

. . . . . . . . . . . . .


Theta=pi/2 points right:

. . . . - + . . . . . . .

. . . . - + . . . . . . .

. . . . - + . . . . . . .

. . . . . . . . . . . . .


2D only, to extend to 3D

"""
function transducer(q::judiVector, d::Tuple, r::Number, theta)
    length(theta) != length(q.geometry.xloc) && throw("Need one angle per source position")
    size(q.data[1], 2) > 1 && throw("Only point sources can be converted to transducer source")
    # Array of source
    nsrc_loc = 11
    nsrc = length(q.geometry.xloc)
    x_base = collect(range(-r, r, length=nsrc_loc))
    y_base = zeros(nsrc_loc)
    y_base_b = zeros(nsrc_loc) .- d[end]

    # New coords and data
    xloc = Array{Any}(undef, nsrc)
    yloc = Array{Any}(undef, nsrc)
    zloc = Array{Any}(undef, nsrc)
    data = Array{Any}(undef, nsrc)
    t = q.geometry.t[1]
    dt = q.geometry.dt[1]

    for i=1:nsrc
        # Build the rotated array of dipole
        R = [cos(theta[i] - pi/2) sin(theta[i] - pi/2);-sin(theta[i] - pi/2) cos(theta[i] - pi/2)]
        # +1 coords
        r_loc = R * [x_base';y_base']
        # -1 coords
        r_loc_b = R * [x_base';y_base_b']
        xloc[i] = q.geometry.xloc[i] .+ vec(vcat(r_loc[1, :], r_loc_b[1, :]))
        zloc[i] = q.geometry.zloc[i] .+ vec(vcat(r_loc[2, :], r_loc_b[2, :]))
        yloc[i] = zeros(2*nsrc_loc)
        data[i] = zeros(length(q.data[i]), 2*nsrc_loc)
        data[i][:, 1:nsrc_loc] .= q.data[i]/nsrc_loc
        data[i][:, nsrc_loc+1:end] .= -q.data[i]/nsrc_loc
    end
    return judiVector(Geometry(xloc, yloc, zloc; t=t, dt=dt), data)
end
