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
export devito_model, pad_sizes, pad_array
export transducer

"""
    devito_model(model, options;dm=nothing)

Creates a python side model strucutre for devito.

Parameters
* `model`: JUDI Model structure.
* `options`: JUDI Options structure.
* `dm`: Squared slowness perturbation (optional), Array or PhysicalParameter.
"""
function devito_model(model::Model, options::JUDIOptions, dm)
    pad = pad_sizes(model, options)
    # Set up Python model structure
    m = pad_array(model[:m].data, pad)
    dm = pad_array(dm, pad)
    physpar = Dict((n, pad_array(v.data, pad)) for (n, v) in model.params if n != :m)
    modelPy = pm."Model"(model.o, model.d, model.n, m, fs=options.free_surface,
                         nbl=model.nb, space_order=options.space_order, dt=options.dt_comp, dm=dm;
                         physpar...)
    return modelPy
end

devito_model(model::Model, options::JUDIOptions, dm::PhysicalParameter) = devito_model(model, options, reshape(dm.data, model.n))
devito_model(model::Model, options::JUDIOptions, dm::Vector{T}) where T = devito_model(model, options, reshape(dm, model.n))
devito_model(model::Model, options::JUDIOptions) = devito_model(model, options, nothing)

"""
    pad_sizes(model, options; so=nothing)

Computes ABC padding sizes according to the model's numbr of abc points and spatial order

Parameters
* `model`: JUDI or Python side Model.
* `options`: JUDI Options structure.
* `so`: Space order (optional) defaults to options.space_order.
"""
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

"""
    pad_array(m, nb; mode=:border)

Pads to the input array with either copying the edge value (:border) or zeros (:zeros)

Parameters
* `m`: Array to be padded.
* `nb`: Size of padding. Array of tuple with one (nb_left, nb_right) tuple per dimension.
* `mode`: Padding mode (optional), defaults to :border.
"""
function pad_array(m::Array{DT}, nb::Array{Tuple{Int64,Int64},1}; mode::Symbol=:border) where {DT}
    n = size(m)
    new_size = Tuple([n[i] + sum(nb[i]) for i=1:length(nb)])
    Ei = []
    for i=1:length(nb)
        left, right = nb[i]
        push!(Ei, joExtend(n[i], mode;pad_upper=right, pad_lower=left, RDT=DT, DDT=DT))
    end
    padded = joKron(Ei...) * PermutedDimsArray(m, length(n):-1:1)[:]
    return PyReverseDims(reshape(padded, reverse(new_size)))
end

pad_array(::Nothing, ::Array{Tuple{Int64,Int64},1}; s::Symbol=:border) = nothing
pad_array(m::Number, ::Array{Tuple{Int64,Int64},1}; s::Symbol=:border) = m

"""
    remove_padding(m, nb; true_adjoint=False)

Removes the padding from array `m`. This is the adjoint of [`pad_array`](@ref).

Parameters
* `m`: Array to remvove padding from.
* `nb`: Size of padding. Array of tuple with one (nb_left, nb_right) tuple per dimension.
* `true_adjoint`: Unpadding mode, defaults to False. Will sum the padding to the edge point with `true_adjoint=true`
 and should only be used this way for adjoint testing purpose.
"""
function remove_padding(gradient::AbstractArray{DT}, nb::Array{Tuple{Int64,Int64},1}; true_adjoint::Bool=false) where {DT}
    N = size(gradient)
    if true_adjoint
        for (dim, (nbl, nbr)) in enumerate(nb)
            selectdim(gradient, dim, nbl+1) .+= dropdims(sum(selectdim(gradient, dim, 1:nbl), dims=dim), dims=dim)
            selectdim(gradient, dim, N[dim]-nbr) .+= dropdims(sum(selectdim(gradient, dim, N[dim]-nbr+1:N[dim]), dims=dim), dims=dim)
        end
    end
    out = gradient[[nbl+1:nn-nbr for ((nbl, nbr), nn) in zip(nb, N)]...]
    return out
end

"""
    limit_model_to_receiver_area(srcGeometry, recGeometry, model, buffer; pert=nothing)

Crops the `model` to the area of the source an receiver with an extra buffer. This reduces the size
of the problem when the model si large and the source and receiver located in a small part of the domain.

In the cartoon below, the full model will be cropped to the center area containg the source (o) receivers (x) and
buffer area (*)

--------------------------------------------
| . . . . . . . . . . . . . . . . . . . . . |
| . . . . . . . . . . . . . . . . . . . . . |  - o Source position
| . . . . * * * * * * * * * * * * . . . . . |  - x receiver positions
| . . . . * x x x x x x x x x x * . . . . . |  - * Extra buffer (grid spacing in that simple case)
| . . . . * x x x x x x x x x x * . . . . . |
| . . . . * x x x x x x x x x x * . . . . . |
| . . . . * x x x x x o x x x x * . . . . . |
| . . . . * x x x x x x x x x x * . . . . . |
| . . . . * x x x x x x x x x x * . . . . . |
| . . . . * x x x x x x x x x x * . . . . . |
| . . . . * * * * * * * * * * * * . . . . . |
| . . . . . . . . . . . . . . . . . . . . . |
| . . . . . . . . . . . . . . . . . . . . . |
--------------------------------------------
Parameters
* `srcGeometry`: Geometry of the source.
* `recGeometry`: Geometry of the receivers.
* `model`: Model to be croped.
* `buffer`: Size of the buffer on each side.
* `pert`: Model perturbation (optional) to be cropped as well.
"""
function limit_model_to_receiver_area(srcGeometry::Geometry, recGeometry::Geometry,
                                      model::Model, buffer::Number; pert=nothing)
    # Restrict full velocity model to area that contains either sources and receivers
    ndim = length(model.n)
    # scan for minimum and maximum x and y source/receiver coordinates
    min_x = min(minimum(recGeometry.xloc[1]), minimum(srcGeometry.xloc[1]))
    max_x = max(maximum(recGeometry.xloc[1]), maximum(srcGeometry.xloc[1]))
    if ndim == 3
        min_y = min(minimum(recGeometry.yloc[1]), minimum(srcGeometry.yloc[1]))
        max_y = max(maximum(recGeometry.yloc[1]), maximum(srcGeometry.yloc[1]))
    end

    # add buffer zone if possible
    min_x = max(model.o[1], min_x - buffer) + 1
    max_x = min(model.o[1] + model.d[1]*(model.n[1]-1), max_x + buffer)
    if ndim == 3
        min_y = max(model.o[2], min_y - buffer)
        max_y = min(model.o[2] + model.d[2]*(model.n[2]-1), max_y + buffer)
    end

    # extract part of the model that contains sources/receivers
    nx_min = Int((min_x-model.o[1]) ÷ model.d[1]) + 1
    nx_max = Int((max_x-model.o[1]) ÷ model.d[1]) + 1
    inds = [max(1, nx_min):nx_max, 1:model.n[end]]
    if ndim == 3
        ny_min = Int((min_y-model.o[2]) ÷ model.d[2]) + 1
        ny_max = Int((max_y-model.o[2]) ÷ model.d[2]) + 1
        insert!(inds, 2, max(1, ny_min):ny_max)
    end

    # Extract relevant model part from full domain
    n_orig = model.n
    for (p, v) in model.params
        typeof(v) <: AbstractArray && (model.params[p] = v[inds...])
    end

    judilog("N old $(model.n)")
    model.n = model.m.n
    model.o = model.m.o
    judilog("N new $(model.n)")
    isnothing(pert) && (return model, nothing)

    pert = reshape(pert, n_orig)[inds...]
    return model, vec(pert)
end

"""
    extend_gradient(model_full, model, array)

This operation does the opposite of [`limit_model_to_receiver_area`](@ref) and put back a cropped array into
the full model.

Parameters
* `model_full`: Full domain model.
* `model`: Cropped model.
* `array`: Array to be extended (padded with zero) to the full model size.
"""
function extend_gradient(model_full::Model, model::Model, gradient::Union{Array, PhysicalParameter})
    # Extend gradient back to full model size
    ndim = length(model.n)
    full_gradient = similar(gradient, model_full)
    fill!(full_gradient, 0)
    nx_start = Int(Float32(Float32(model.o[1] - model_full.o[1]) ÷ model.d[1])) + 1
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
    remove_out_of_bounds_receivers(recGeometry, model)

Removes receivers that are positionned outside the computational domain defined by the model.

Parameters
* `recGeometry`: Geometry of receivers in which out of bounds will be removed.
* `model`: Model defining the computational domain.
"""
function remove_out_of_bounds_receivers(recGeometry::Geometry, model::Model)

    # Only keep receivers within the model
    xmin, xmax = model.o[1], model.o[1] + (model.n[1] - 1)*model.d[1]
    if typeof(recGeometry.xloc[1]) <: Array
        idx_xrec = findall(x -> xmax >= x >= xmin, recGeometry.xloc[1])
        recGeometry.xloc[1] = recGeometry.xloc[1][idx_xrec]
        length(recGeometry.yloc[1]) > 1 && (recGeometry.yloc[1] = recGeometry.yloc[1][idx_xrec])
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_xrec]
    end

    # For 3D shot records, scan also y-receivers
    if length(model.n) == 3 && typeof(recGeometry.yloc[1]) <: Array
        ymin, ymax = model.o[2], model.o[2] + (model.n[2] - 1)*model.d[2]
        idx_yrec = findall(x -> ymax >= x >= ymin, recGeometry.yloc[1])
        recGeometry.xloc[1] = recGeometry.xloc[1][idx_yrec]
        recGeometry.yloc[1] = recGeometry.yloc[1][idx_yrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_yrec]
    end
    return recGeometry
end

"""
    remove_out_of_bounds_receivers(recGeometry, recData, model)

Removes receivers that are positionned outside the computational domain defined by the model.

Parameters
* `recGeometry`: Geometry of receivers in which out of bounds will be removed.
* `recData`: Shot record for that geometry in which traces will be removed.
* `model`: Model defining the computational domain.
"""
function remove_out_of_bounds_receivers(recGeometry::Geometry, recData::Matrix{T}, model::Model) where T

    # Only keep receivers within the model
    xmin, xmax = model.o[1], model.o[1] + (model.n[1] - 1)*model.d[1]
    if typeof(recGeometry.xloc[1]) <: Array
        idx_xrec = findall(x -> xmax >= x >= xmin, recGeometry.xloc[1])
        recGeometry.xloc[1] = recGeometry.xloc[1][idx_xrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_xrec]
        recData = recData[:, idx_xrec]
    end

    # For 3D shot records, scan also y-receivers
    if length(model.n) == 3 && typeof(recGeometry.yloc[1]) <: Array
        ymin, ymax = model.o[2], model.o[2] + (model.n[2] - 1)*model.d[2]
        idx_yrec = findall(x -> ymax >= x >= ymin, recGeometry.yloc[1])
        recGeometry.yloc[1] = recGeometry.yloc[1][idx_yrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_yrec]
        recData = recData[:, idx_yrec]
    end
    return recGeometry, recData
end

remove_out_of_bounds_receivers(G::Geometry, ::Nothing, M::Model) = (remove_out_of_bounds_receivers(G, M), nothing)
remove_out_of_bounds_receivers(::Nothing, ::Nothing, M::Model) = (nothing, nothing)
remove_out_of_bounds_receivers(::Nothing, r::AbstractArray, M::Model) = (nothing, r)
remove_out_of_bounds_receivers(G::Geometry, r, M::Model) = remove_out_of_bounds_receivers(G, convert(Matrix{Float32}, r), M)
remove_out_of_bounds_receivers(w::AbstractArray, ::Nothing, M::Model) = (w, nothing)

"""
    convertToCell(x)

Convert an array `x` to a cell array (`Array{Any,1}`) with `length(x)` entries,\\
where the i-th cell contains the i-th entry of `x`.

Parameters
* `x`: Array to be converted into and array of array
"""
function convertToCell(x::Array{T, 1}) where T
    n = length(x)
    y = Array{Array{T, 1}, 1}(undef, n)
    for j=1:n
        y[j] = [x[j]]
    end
    return y
end

convertToCell(x::Array{Array{T, N}, 1}) where {T, N} = x
convertToCell(x::StepRangeLen) = convertToCell(Float32.(x))
convertToCell(x::Number) = convertToCell([x])

# 1D source time function
"""
    source(tmax, dt, f0)

Create seismic Ricker wavelet of length `tmax` (in milliseconds) with sampling interval `dt` (in milliseonds)\\
and central frequency `f0` (in kHz).

"""
function ricker_wavelet(tmax, dt, f0; t0=nothing)
    R = typeof(dt)
    isnothing(t0) ? t0 = R(0) : tmax = R(tmax - t0)
    nt = floor(Int, tmax / dt) + 1
    t = range(t0, stop=tmax, length=nt)
    r = (pi * f0 * (t .- 1 / f0))
    q = zeros(Float32,nt,1)
    q[:,1] = (1f0 .- 2f0 .* r.^2f0) .* exp.(-r.^2f0)
    return q
end

"""
    calculate_dt(model; dt=nothing)

Compute the computational time step based on the CFL condition and physical parameters
in the model.

Parameters
* `model`: Model structure
* `dt`: User defined time step (optional), will be the value returned if provided.
"""
function calculate_dt(model::Model; dt=nothing)
    if ~isnothing(dt)
        return dt
    end
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
    if typeof(srcGeometry) <: GeometryOOC
        nsrc = length(srcGeometry.container)
    else
        nsrc = length(srcGeometry.xloc)
    end
    nt = Array{Integer}(undef, nsrc)
    dtComp = calculate_dt(model; dt=dt)
    for j=1:nsrc
        ntRec = Int(recGeometry.dt[j]*(recGeometry.nt[j]-1) ÷ dtComp) + 1
        ntSrc = Int(srcGeometry.dt[j]*(srcGeometry.nt[j]-1) ÷ dtComp) + 1
        nt[j] = max(ntRec, ntSrc)
    end
    return nt
end

"""
    get_computational_nt(Geoemtry, model; dt=nothing)

Estimate the number of computational time steps. Required for calculating the dimensions\\
of the matrix-free linear modeling operators. `srcGeometry` and `recGeometry` are source\\
and receiver geometries of type `Geometry` and `model` is the model structure of type \\
`Model`.

"""
function get_computational_nt(Geometry, model::Model; dt=nothing)
    # Determine number of computational time steps
    if typeof(Geometry) <: GeometryOOC
        nsrc = length(Geometry.container)
    else
        nsrc = length(Geometry.xloc)
    end
    nt = Array{Integer}(undef, nsrc)
    dtComp = calculate_dt(model; dt=dt)
    for j=1:nsrc
        nt[j] = Int(Geometry.dt[j]*(Geometry.nt[j]-1) ÷ dtComp) + 1
    end
    return nt
end

"""
    setup_grid(geometry, n)

Sets up the coordinate arrays for Devito.

Parameters:
* `geometry`: Geometry containing the coordinates
* `n`: Domain size
"""
setup_grid(geometry, ::NTuple{3, T}) where T = hcat(geometry.xloc[1], geometry.yloc[1], geometry.zloc[1])
setup_grid(geometry, ::NTuple{2, T}) where T = hcat(geometry.xloc[1], geometry.zloc[1])
setup_grid(geometry, ::NTuple{1, T}) where T = geometry.xloc[1]

"""
    setup_3D_grid(x, y, z)

Converts one dimensional input (x, y, z) into three dimensional coordinates. The number of point created
is `length(x)*lenght(y)` with all the x/y pairs and each pair at depth z[idx[x]]. `x` and `z` must have the same size.

Parameters:
* `x`: X coordinates.
* `y`: Y coordinates.
* `z`: Z coordinates.
"""
function setup_3D_grid(xrec::Vector{<:AbstractVector{T}},yrec::Vector{<:AbstractVector{T}},zrec::AbstractVector{T}) where T<:Real
    # Take input 1d x and y coordinate vectors and generate 3d grid. Input are cell arrays
    nsrc = length(xrec)
    xloc = Vector{Array{T}}(undef, nsrc)
    yloc = Vector{Array{T}}(undef, nsrc)
    zloc = Vector{Array{T}}(undef, nsrc)
    for i=1:nsrc
        nxrec = length(xrec[i])
        nyrec = length(yrec[i])

        xloc[i] = zeros(T, nxrec*nyrec)
        yloc[i] = zeros(T, nxrec*nyrec)
        zloc[i] = zeros(T, nxrec*nyrec)

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

"""
    setup_3D_grid(x, y, z)

Converts one dimensional input (x, y, z) into three dimensional coordinates. The number of point created
is `length(x)*lenght(y)` with all the x/y pairs and each pair at same depth z.

Parameters:
* `x`: X coordinates.
* `y`: Y coordinates.
* `z`: Z coordinate.
"""
function setup_3D_grid(xrec::AbstractVector{T},yrec::AbstractVector{T}, zrec::T) where T<:Real
# Take input 1d x and y coordinate vectors and generate 3d grid. Input are arrays/ranges
    nxrec = length(xrec)
    nyrec = length(yrec)

    xloc = zeros(T, nxrec*nyrec)
    yloc = zeros(T, nxrec*nyrec)
    zloc = zeros(T, nxrec*nyrec)
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

setup_3D_grid(xrec, yrec, zrec) = setup_3D_grid(tof32(xrec), tof32(yrec), tof32(zrec))

function setup_3D_grid(xrec::Vector{Any}, yrec::Vector{Any}, zrec::Vector{Any})
    xrec, yrec, zrec = convert(Vector{typeof(xrec[1])},xrec), convert(Vector{typeof(yrec[1])}, yrec), convert(Vector{typeof(zrec[1])},zrec)
    setup_3D_grid(xrec, yrec, zrec)
end

"""
    time_resample(data, geometry_in, dt_new)

Resample the input data with sinc interpolation from the current time sampling (geometrty_in) to the
new time sampling `dt_new`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `geometry_in`: Geometry on which `data` is defined.
* `dt_new`: New time sampling rate to interpolate onto.
"""
function time_resample(data::AbstractArray{Float32, N}, geometry_in::Geometry, dt_new::AbstractFloat) where N
    @assert N<=2
    if dt_new==geometry_in.dt[1]
        return data, geometry_in
    else
        geometry = deepcopy(geometry_in)
        timeAxis = 0f0:geometry.dt[1]:geometry.t[1]
        timeInterp = 0f0:dt_new:geometry.t[1]
        dataInterp = SincInterpolation(data, timeAxis, timeInterp)

        geometry.dt[1] = dt_new
        geometry.nt[1] = length(timeInterp)
        geometry.t[1] = (geometry.nt[1] - 1)*geometry.dt[1]
        return dataInterp, geometry
    end
end


"""
    time_resample(data, dt_in, dt_new)

Resample the input data with sinc interpolation from the current time sampling dt_in to the 
new time sampling `dt_new`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `dt_in`: Time sampling of input
* `dt_new`: New time sampling rate to interpolate onto.
"""
function time_resample(data::Array, dt_in::T1, dt_new::T2) where {T1<:AbstractFloat, T2<:AbstractFloat}

    if dt_new==dt_in
        return data
    else
        nt = size(data, 1)
        timeAxis = 0:dt_in:(nt-1)*dt_in
        timeInterp = 0:dt_new:(nt-1)*dt_in
        dataInterp = Float32.(SincInterpolation(data, timeAxis, timeInterp))
        return dataInterp
    end
end


"""
    time_resample(data, dt_in, geometry_in)

Resample the input data with sinc interpolation from the current time sampling (dt_in) to the
new time sampling `geometry_out`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `geometry_out`: Geometry on which `data` is to be interpolated.
* `dt_in`: Time sampling rate of the `data.`
"""
function time_resample(data::AbstractArray{Float32, N}, dt_in::AbstractFloat, geometry_out::Geometry) where N
    @assert N<=2
    if dt_in == geometry_out.dt[1]
        return data
    else
        timeAxis = 0f0:dt_in:geometry_out.t[1]
        timeInterp = 0f0:geometry_out.dt[1]:geometry_out.t[1]
        return  SincInterpolation(data, timeAxis, timeInterp)
    end
end

"""
    generate_distribution(x; src_no=1)

Generates a probability distribution for the discrete input judiVector `x`.

Parameters
* `x`: judiVector. Usualy a source with a single trace per source position.
* `src_no`: Index of the source to select out of `x`
"""
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
	ns = nt ÷ 2 + 1
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

"""
    select_frequencies(q_dist; fmin=0f0, fmax=Inf, nf=1)

Selects `nf` frequencies based on the source distribution `q_dist` computed with [`generate_distribution`](@ref).

Parameters
* `q_dist`: Distribution to sample from.
* `f_min`: Minimum acceptable frequency to sample (defaults to 0).
* `f_max`: Maximum acceptable frequency to sample (defaults to Inf).
* `fd`: Number of frequnecies to sample (defaults to 1).
"""
function select_frequencies(q_dist; fmin=0f0, fmax=Inf, nf=1)
	freq = zeros(Float32, nf)
	for j=1:nf
		while (freq[j] <= fmin) || (freq[j] > fmax)
			freq[j] = q_dist(rand(1)[1])[1]
		end
	end
	return freq
end

"""
    process_input_data(input, geometry, nsrc)

Preprocesses input Array into an Array of Array for modeling

Parameters:
* `input`: Input to preprocess.
* `geometry`: Geometry containing physical parameters.
* `nsrc`: Number of sources
"""
function process_input_data(input::Array{Float32}, geometry::Geometry)
    # Input data is pure Julia array: assume fixed no.
    # of receivers and reshape into data cube nt x nrec x nsrc
    nt = Int(geometry.nt[1])
    nrec = length(geometry.xloc[1])
    nsrc = length(geometry.xloc)
    data = reshape(input, nt, nrec, nsrc)
    dataCell = Array{Array{Float32, 2}, 1}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = data[:,:,j]
    end
    return dataCell
end

"""
    process_input_data(input, model, nsrc)

Preprocesses input Array into an Array of Array for modeling

Parameters:
* `input`: Input to preprocess.
* `model`: Model containing physical parameters.
* `nsrc`: Number of sources
"""
function process_input_data(input::Array{Float32}, model::Model, nsrc::Integer)
    ndims = length(model.n)
    dataCell = Array{Array{Float32, ndims}, 1}(undef, nsrc)

    input = reshape(input, model.n..., nsrc)
    nd = ndims(input)
    for j=1:nsrc
        dataCell[j] = selectdim(input, nd, j)
    end
    return dataCell
end

process_input_data(input::Array{Float32}, model::Model) = process_input_data(input, model, length(input) ÷ prod(model.n))
process_input_data(input::judiVector, ::Geometry) = input
process_input_data(input::judiVector) = input.data
process_input_data(input::judiWeights, ::Model) = input.weights

function process_input_data(input::Array{T}, v::Vector{<:Array}) where T
    nsrc = length(v)
    dataCell = Vector{Vector{T}}(undef, nsrc)
    input = reshape(input, :, nsrc)
    for j=1:nsrc
        dataCell[j] = input[:, j]
    end
    return dataCell
end

"""
    reshape(x::Array{Float32, 1}, geometry::Geometry)

Reshapes input vector intu a 3D `nt x nrec x nsrc` Array.
"""
function reshape(x::Array{Float32, 1}, geometry::Geometry)
    nt = geometry.nt[1]
    nrec = length(geometry.xloc[1])
    nsrc = Int(length(x) / nt / nrec)
    return reshape(x, nt, nrec, nsrc)
end

"""
    reshape(V::Vector{T}, n::Tuple, nblock::Integer)

Reshapes input vector into a `Vector{Array{T, N}}` of length `nblock` with each subarray of size `n`
"""
function reshape(x::Vector{T}, d::Dims, nsrc::Integer) where {T<:Number}
    length(x) == prod(d)*nsrc || throw(judiMultiSourceException("Incompatible size"))
    as_nd = reshape(x, d..., nsrc)
    return [collect(selectdim(as_nd, ndims(as_nd), s)) for s=1:nsrc]
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
    xloc = Array{Array{Float32, 1}, 1}(undef, nsrc)
    yloc = Array{Array{Float32, 1}, 1}(undef, nsrc)
    zloc = Array{Array{Float32, 1}, 1}(undef, nsrc)
    data = Array{Array{Float32, 2}}(undef, nsrc)
    t = q.geometry.t[1]
    dt = q.geometry.dt[1]

    for i=1:nsrc
        # Build the rotated array of dipole
        R = Float32.([cos(theta[i] - pi/2) sin(theta[i] - pi/2);-sin(theta[i] - pi/2) cos(theta[i] - pi/2)])
        # +1 coords
        r_loc = R * [x_base';y_base']
        # -1 coords
        r_loc_b = R * [x_base';y_base_b']
        xloc[i] = q.geometry.xloc[i] .+ vec(vcat(r_loc[1, :], r_loc_b[1, :]))
        zloc[i] = q.geometry.zloc[i] .+ vec(vcat(r_loc[2, :], r_loc_b[2, :]))
        yloc[i] = zeros(Float32, 2*nsrc_loc)
        data[i] = zeros(Float32, length(q.data[i]), 2*nsrc_loc)
        data[i][:, 1:nsrc_loc] .= q.data[i]/nsrc_loc
        data[i][:, nsrc_loc+1:end] .= -q.data[i]/nsrc_loc
    end
    return judiVector(Geometry(xloc, yloc, zloc; t=t, dt=dt), data)
end

########################################### Misc defaults
# Vectorization of single variable (not defined in Julia)
vec(x::Float64) = x;
vec(x::Float32) = x;
vec(x::Int64) = x;
vec(x::Int32) = x;
vec(::Nothing) = nothing

SincInterpolation(Y::AbstractMatrix{T}, S::AbstractRange{T}, Up::AbstractRange{T}) where T<:AbstractFloat =
    sinc.( (Up .- S') ./ (S[2] - S[1]) ) * Y

"""
    as_vec(x, ::Val{Bool})
Vectorizes output when `return_array` is set to `true`.
"""
as_vec(x, ::Val) = x
as_vec(x::Tuple, v::Val) = tuple((as_vec(xi, v) for xi in x)...)
as_vec(x::Ref, ::Val) = x[]
as_vec(x::PhysicalParameter, ::Val{true}) = vec(x.data)
as_vec(x::judiMultiSourceVector, ::Val{true}) = vec(x)
