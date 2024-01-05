# Auxiliary functions for TimeModeling module
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: September 2016
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

export ricker_wavelet, get_computational_nt, calculate_dt, setup_grid, setup_3D_grid
export convertToCell, limit_model_to_receiver_area, remove_out_of_bounds_receivers, extend_gradient
export time_resample, remove_padding, subsample, process_input_data
export generate_distribution, select_frequencies
export devito_model, pad_sizes, pad_array
export transducer, as_vec

"""
    devito_model(model, options;dm=nothing)

Creates a python side model strucutre for devito.

Parameters
* `model`: JUDI Model structure.
* `options`: JUDI Options structure.
* `dm`: Squared slowness perturbation (optional), Array or PhysicalParameter.
"""
function devito_model(model::MT, options::JUDIOptions, dm) where {MT<:AbstractModel}
    pad = pad_sizes(model, options)
    # Set up Python model structure
    dm = pad_array(dm, pad)
    physpar = Dict((n, pad_array(v, pad)) for (n, v) in _params(model))

    modelPy = rlock_pycall(pm."Model", PyObject, origin(model), spacing(model), size(model), fs=options.free_surface,
                   nbl=nbl(model), space_order=options.space_order, dt=options.dt_comp, dm=dm;
                   physpar...)

    return modelPy
end

devito_model(model::AbstractModel, options::JUDIOptions, dm::PhysicalParameter) = devito_model(model, options, reshape(dm.data, size(model)))
devito_model(model::AbstractModel, options::JUDIOptions, dm::Vector{T}) where T = devito_model(model, options, reshape(dm, size(model)))
devito_model(model::AbstractModel, options::JUDIOptions) = devito_model(model, options, nothing)

"""
    pad_sizes(model, options; so=nothing)

Computes ABC padding sizes according to the model's numbr of abc points and spatial order

Parameters
* `model`: JUDI or Python side Model.
* `options`: JUDI Options structure.
* `so`: Space order (optional) defaults to options.space_order.
"""
function pad_sizes(model::PyObject, options; so=nothing)
    isnothing(so) && (so = options.space_order)
    N = model.grid.dim
    return tuple([(nbl + so, nbr + so) for (nbl, nbr)=model.padsizes]...)
end

function pad_sizes(model::AbstractModel{T, N}, options; so=nothing) where {T, N}
    isnothing(so) && (so = options.space_order)
    padsizes = [(nbl(model) + so, nbl(model) + so) for i=1:N]
    if options.free_surface
        padsizes[end] = (so, nbl(model) + so)
    end
    return tuple(padsizes...)
end

"""
    pad_array(m, nb; mode=:border)

Pads to the input array with either copying the edge value (:border) or zeros (:zeros)

Parameters
* `m`: Array to be padded.
* `nb`: Size of padding. Array of tuple with one (nb_left, nb_right) tuple per dimension.
* `mode`: Padding mode (optional), defaults to :border.
"""
function pad_array(m::Array{DT, N}, nb::NTuple{N, NTuple{2, Int64}}; mode::Symbol=:border) where {DT, N}
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

pad_array(::Nothing, ::NTuple{N, NTuple{2, Int64}}; s::Symbol=:border) where N = nothing
pad_array(m::Number, ::NTuple{N, NTuple{2, Int64}}; s::Symbol=:border) where N = m
pad_array(m::PhysicalParameter, nb::NTuple{N, NTuple{2, Int64}}; mode::Symbol=:border) where {N} = pad_array(m.data, nb; mode=mode)

"""
    remove_padding(m, nb; true_adjoint=False)

Removes the padding from array `m`. This is the adjoint of [`pad_array`](@ref).

Parameters
* `m`: Array to remvove padding from.
* `nb`: Size of padding. Array of tuple with one (nb_left, nb_right) tuple per dimension.
* `true_adjoint`: Unpadding mode, defaults to False. Will sum the padding to the edge point with `true_adjoint=true`
 and should only be used this way for adjoint testing purpose.
"""
function remove_padding(gradient::AbstractArray{DT, ND}, nb::NTuple{ND, Tuple{Int64, Int64}}; true_adjoint::Bool=false) where {ND, DT}
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

- o Source position
- x receiver positions
- * Extra buffer (grid spacing in that simple case)

--------------------------------------------  \n
| . . . . . . . . . . . . . . . . . . . . . | \n
| . . . . . . . . . . . . . . . . . . . . . | \n
| . . . . * * * * * * * * * * * * . . . . . | \n
| . . . . * x x x x x x x x x x * . . . . . | \n
| . . . . * x x x x x x x x x x * . . . . . | \n
| . . . . * x x x x x x x x x x * . . . . . | \n
| . . . . * x x x x x o x x x x * . . . . . | \n
| . . . . * x x x x x x x x x x * . . . . . | \n
| . . . . * x x x x x x x x x x * . . . . . | \n
| . . . . * x x x x x x x x x x * . . . . . | \n
| . . . . * * * * * * * * * * * * . . . . . | \n
| . . . . . . . . . . . . . . . . . . . . . | \n
| . . . . . . . . . . . . . . . . . . . . . | \n
--------------------------------------------  \n

Parameters
* `srcGeometry`: Geometry of the source.
* `recGeometry`: Geometry of the receivers.
* `model`: Model to be croped.
* `buffer`: Size of the buffer on each side.
* `pert`: Model perturbation (optional) to be cropped as well.
"""
function limit_model_to_receiver_area(srcGeometry::Geometry{T}, recGeometry::Geometry{T},
                                      model::MT, buffer::Number; pert=nothing) where {MT<:AbstractModel, T<:Real}
    # Restrict full velocity model to area that contains either sources and receivers
    ndim = length(size(model))
    n_orig = size(model)
    # scan for minimum and maximum x and y source/receiver coordinates
    min_x = min(minimum(recGeometry.xloc[1]), minimum(srcGeometry.xloc[1]))
    max_x = max(maximum(recGeometry.xloc[1]), maximum(srcGeometry.xloc[1]))
    if ndim == 3
        min_y = min(minimum(recGeometry.yloc[1]), minimum(srcGeometry.yloc[1]))
        max_y = max(maximum(recGeometry.yloc[1]), maximum(srcGeometry.yloc[1]))
    end

    # add buffer zone if possible
    min_x = max(origin(model, 1), min_x - buffer)
    max_x = min(origin(model, 1) + spacing(model, 1)*(size(model, 1)-1), max_x + buffer)
    if ndim == 3
        min_y = max(origin(model, 2), min_y - buffer)
        max_y = min(origin(model, 2) + spacing(model, 2)*(size(model, 2)-1), max_y + buffer)
    end

    # extract part of the model that contains sources/receivers
    nx_min = Int((min_x-origin(model, 1)) ÷ spacing(model, 1)) + 1
    nx_max = Int((max_x-origin(model, 1)) ÷ spacing(model, 1)) + 1
    inds = [max(1, nx_min):nx_max, 1:size(model)[end]]
    if ndim == 3
        ny_min = Int((min_y-origin(model, 2)) ÷ spacing(model, 2)) + 1
        ny_max = Int((max_y-origin(model, 2)) ÷ spacing(model, 2)) + 1
        insert!(inds, 2, max(1, ny_min):ny_max)
    end

    # Extract relevant model part from full domain
    newp = OrderedDict()
    for (p, v) in _params(model)
        newp[p] = isa(v, AbstractArray) ? v[inds...] : v
    end
    p = findfirst(x->isa(x, PhysicalParameter), newp)
    o, n = newp[p].o, newp[p].n

    new_model = MT(DiscreteGrid(n, spacing(model), o, nbl(model)), values(newp)...)
    isnothing(pert) && (return new_model, nothing)

    newpert = reshape(pert, n_orig)[inds...]
    return new_model, newpert[1:end]
end

"""
    remove_out_of_bounds_receivers(recGeometry, model)

Removes receivers that are positionned outside the computational domain defined by the model.

Parameters
* `recGeometry`: Geometry of receivers in which out of bounds will be removed.
* `model`: Model defining the computational domain.
"""
function remove_out_of_bounds_receivers(recGeometry::Geometry{T}, model::AbstractModel{T, N}) where {T<:Real, N}

    # Only keep receivers within the model
    xmin, xmax = origin(model, 1), origin(model, 1) + (size(model)[1] - 1)*spacing(model, 1)
    if typeof(recGeometry.xloc[1]) <: Array
        idx_xrec = findall(x -> xmax >= x >= xmin, recGeometry.xloc[1])
        recGeometry.xloc[1] = recGeometry.xloc[1][idx_xrec]
        length(recGeometry.yloc[1]) > 1 && (recGeometry.yloc[1] = recGeometry.yloc[1][idx_xrec])
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_xrec]
    end

    # For 3D shot records, scan also y-receivers
    if length(size(model)) == 3 && typeof(recGeometry.yloc[1]) <: Array
        ymin, ymax = origin(model, 2), origin(model, 2) + (size(model, 2) - 1)*spacing(model, 2)
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
function remove_out_of_bounds_receivers(recGeometry::Geometry{T}, recData::Matrix{T}, model::AbstractModel) where {T<:Real}

    # Only keep receivers within the model
    xmin, xmax = origin(model, 1), origin(model, 1) + (size(model)[1] - 1)*spacing(model, 1)
    if typeof(recGeometry.xloc[1]) <: Array
        idx_xrec = findall(x -> xmax >= x >= xmin, recGeometry.xloc[1])
        recGeometry.xloc[1] = recGeometry.xloc[1][idx_xrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_xrec]
        recData = recData[:, idx_xrec]
    end

    # For 3D shot records, scan also y-receivers
    if length(size(model)) == 3 && typeof(recGeometry.yloc[1]) <: Array
        ymin, ymax = origin(model, 2), origin(model, 2) + (size(model, 2) - 1)*spacing(model, 2)
        idx_yrec = findall(x -> ymax >= x >= ymin, recGeometry.yloc[1])
        recGeometry.yloc[1] = recGeometry.yloc[1][idx_yrec]
        recGeometry.zloc[1] = recGeometry.zloc[1][idx_yrec]
        recData = recData[:, idx_yrec]
    end
    return recGeometry, recData
end

remove_out_of_bounds_receivers(G::Geometry, ::Nothing, M::AbstractModel) = (remove_out_of_bounds_receivers(G, M), nothing)
remove_out_of_bounds_receivers(::Nothing, ::Nothing, M::AbstractModel) = (nothing, nothing)
remove_out_of_bounds_receivers(w, r::AbstractArray, ::AbstractModel) = (w, r)
remove_out_of_bounds_receivers(G::Geometry, r::AbstractArray, M::AbstractModel) = remove_out_of_bounds_receivers(G, convert(Matrix{Float32}, r), M)
remove_out_of_bounds_receivers(w::AbstractArray, ::Nothing, M::AbstractModel) = (w, nothing)

"""
    convertToCell(x)

Convert an array `x` to a cell array (`Array{Any,1}`) with `length(x)` entries,\\
where the i-th cell contains the i-th entry of `x`.

Parameters
* `x`: Array to be converted into and array of array
"""
function convertToCell(x::Vector{T}) where T<:Number
    n = length(x)
    y = Array{Array{T, 1}, 1}(undef, n)
    for j=1:n
        y[j] = [x[j]]
    end
    return y
end

convertToCell(x::Vector{Array{T, N}}) where {T<:Number, N} = x
convertToCell(x::StepRangeLen) = convertToCell(Float32.(x))
convertToCell(x::Number) = convertToCell([x])

# 1D source time function
"""
    ricker_wavelet(tmax, dt, f0)

Create seismic Ricker wavelet of length `tmax` (in milliseconds) with sampling interval `dt` (in milliseonds)\\
and central frequency `f0` (in kHz).

"""
function ricker_wavelet(tmax::T, dt::T, f0::T; t0=nothing) where {T<:Real}
    isnothing(t0) ? t0 = T(0) : tmax = T(tmax - t0)
    nt = floor(Int, tmax / dt) + 1
    t = range(t0, stop=tmax, length=nt)
    r = (pi * f0 * (t .- 1 / f0))
    q = zeros(T, nt, 1)
    q[:,1] .= (T(1) .- T(2) .* r.^T(2)) .* exp.(-r.^T(2))
    return q
end

ricker_wavelet(tmax::T, dt, f0; t0=nothing) where {T<:Real} = ricker_wavelet(tmax, T(dt), T(f0); t0=t0)


"""
    calculate_dt(model; dt=nothing)

Compute the computational time step based on the CFL condition and physical parameters
in the model.

Parameters
* `model`: Model structure
* `dt`: User defined time step (optional), will be the value returned if provided.
"""
function calculate_dt(model::Union{ViscIsoModel{T, N}, IsoModel{T, N}}; dt=nothing) where {T<:Real, N}
    if ~isnothing(dt)
        return dt
    end
    m = minimum(model.m)

    modelPy = rlock_pycall(pm."Model", PyObject, origin=origin(model), spacing=spacing(model), shape=ntuple(_ -> 11, N),
               m=m, nbl=0)

    return calculate_dt(modelPy)
end

function calculate_dt(model::IsoElModel{T, N}; dt=nothing) where {T<:Real, N}
    if ~isnothing(dt)
        return dt
    end
    lam = maximum(model.lam)
    mu = maximum(model.mu)

    modelPy = rlock_pycall(pm."Model", PyObject, origin=origin(model), spacing=spacing(model), shape=ntuple(_ -> 11, N),
               lam=lam, mu=mu, nbl=0)

    return calculate_dt(modelPy)
end

function calculate_dt(model::TTIModel{T, N}; dt=nothing) where {T<:Real, N}
    if ~isnothing(dt)
        return dt
    end
    m = minimum(model.m)

    epsilon = maximum(model.epsilon)
    modelPy = rlock_pycall(pm."Model", PyObject, origin=origin(model), spacing=spacing(model), shape=ntuple(_ -> 11, N),
               m=m, epsilon=epsilon, nbl=0)

    return calculate_dt(modelPy)
end

calculate_dt(modelPy::PyObject) = convert(Float32, modelPy."critical_dt")

"""
    get_computational_nt(srcGeometry, recGeoemtry, model; dt=nothing)

Estimate the number of computational time steps. Required for calculating the dimensions\\
of the matrix-free linear modeling operators. `srcGeometry` and `recGeometry` are source\\
and receiver geometries of type `Geometry` and `model` is the model structure of type \\
`Model`.

"""
function get_computational_nt(srcGeometry::Geometry{T}, recGeometry::Geometry{T}, model::AbstractModel; dt=nothing) where {T<:Real}
    # Determine number of computational time steps
    nsrc = get_nsrc(srcGeometry)
    nt = Vector{Int64}(undef, nsrc)
    dtComp = calculate_dt(model; dt=dt)
    for j=1:nsrc
        ntRec = length(0:dtComp:(dtComp*ceil(recGeometry.t[j]/dtComp)))
        ntSrc = length(0:dtComp:(dtComp*ceil(srcGeometry.t[j]/dtComp)))
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
function get_computational_nt(Geometry::Geometry{T}, model::AbstractModel; dt=nothing) where {T<:Real}
    # Determine number of computational time steps
    nsrc = get_nsrc(Geometry)
    nt = Array{Integer}(undef, nsrc)
    dtComp = calculate_dt(model; dt=dt)
    for j=1:nsrc
        nt[j] = length(0:dtComp:(dtComp*ceil(Geometry.t[j]/dtComp)))
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
setup_grid(geometry::GeometryIC{T}, ::NTuple{3, <:Integer}) where {T<:Real} = hcat(geometry.xloc[1], geometry.yloc[1], geometry.zloc[1])
setup_grid(geometry::GeometryIC{T}, ::NTuple{2, <:Integer}) where {T<:Real} = hcat(geometry.xloc[1], geometry.zloc[1])
setup_grid(geometry::GeometryIC{T}, ::NTuple{1, <:Integer}) where {T<:Real} = geometry.xloc[1]

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
    xloc = Vector{Vector{T}}(undef, nsrc)
    yloc = Vector{Vector{T}}(undef, nsrc)
    zloc = Vector{Vector{T}}(undef, nsrc)
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
time_resample(data::AbstractArray{T, N}, G_in::Geometry, dt_new::Real) where {T<:Real, N} = time_resample(data, G_in.dt[1], dt_new, G_in.t[1])

"""
    time_resample(data, dt_in, dt_new)

Resample the input data with sinc interpolation from the current time sampling dt_in to the 
new time sampling `dt_new`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `dt_in`: Time sampling of input
* `dt_new`: New time sampling rate to interpolate onto.
"""
function time_resample(data::AbstractArray{T, N}, dt_in::T, dt_new::T, t::T) where {T<:Real, N}

    if dt_new==dt_in
        return data
    elseif (dt_new % dt_in) == 0
        rate = Int64(div(dt_new, dt_in))
        return _time_resample(data, rate)
    else
        @juditime "Data time sinc-interpolation" begin
            nt = size(data, 1)
            timeAxis = StepRangeLen(0f0, T(dt_in), nt)
            timeInterp = 0:dt_new:(dt_new*ceil(t/dt_new))
            dataInterp = Float32.(SincInterpolation(data, timeAxis, timeInterp))
        end
        return dataInterp
    end
end

time_resample(data::AbstractArray{T, N}, dt_in::Number, dt_new::Number, t::Number) where {T<:Real, N} = time_resample(data, T(dt_in), T(dt_new), T(t))


"""
    time_resample(data, dt_in, geometry_in)

Resample the input data with sinc interpolation from the current time sampling (dt_in) to the
new time sampling `geometry_out`.

Parameters
* `data`: Data to be reampled. If data is a matrix, resamples each column.
* `geometry_out`: Geometry on which `data` is to be interpolated.
* `dt_in`: Time sampling rate of the `data.`
"""
time_resample(data::AbstractArray{T, N}, dt_in::Real, G_out::Geometry{T}) where {T<:Real, N} = time_resample(data, dt_in, G_out.dt[1], G_out.t[1])

_time_resample(data::Matrix{T}, rate::Integer) where T = data[1:rate:end, :]
_time_resample(data::PermutedDimsArray{T, 2, (2, 1), (2, 1), Matrix{T}}, rate::Integer) where {T<:Real} = data.parent[:, 1:rate:end]'

SincInterpolation(Y::Matrix{T}, S::StepRangeLen{T}, Up::StepRangeLen{T}) where T<:Real = sinc.( (Up .- S') ./ (S[2] - S[1]) ) * Y
SincInterpolation(Y::PermutedDimsArray{T, 2, (2, 1), (2, 1), Matrix{T}}, S::StepRangeLen{T}, Up::StepRangeLen{T}) where T<:Real = (Y.parent * sinc.( (Up' .- S) ./ (S[2] - S[1]) ))'

"""
    generate_distribution(x; src_no=1)

Generates a probability distribution for the discrete input judiVector `x`.

Parameters
* `x`: judiVector. Usualy a source with a single trace per source position.
* `src_no`: Index of the source to select out of `x`
"""
function generate_distribution(x::judiVector{T, Matrix{T}}; src_no=1) where {T<:Real}
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
function select_frequencies(q_dist::Spline1D; fmin=0f0, fmax=Inf, nf=1)
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
function process_input_data(input::DenseArray{T}, geometry::Geometry{T}) where {T<:Real}
    # Input data is pure Julia array: assume fixed no.
    # of receivers and reshape into data cube nt x nrec x nsrc
    nt = Int(geometry.nt[1])
    nrec = geometry.nrec[1]
    nsrc = length(geometry.xloc)
    data = reshape(input, nt, nrec, nsrc)
    dataCell = Vector{Matrix{T}}(undef, nsrc)
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
function process_input_data(input::DenseArray{T}, model::AbstractModel{T, N}, nsrc::Integer) where {T<:Real, N}
    dataCell = Vector{Array{T, N}}(undef, nsrc)

    input = reshape(input, size(model)..., nsrc)
    nd = ndims(input)
    for j=1:nsrc
        dataCell[j] = selectdim(input, nd, j)
    end
    return dataCell
end

process_input_data(input::DenseArray{T}, model::AbstractModel{T, N}) where {T<:Real, N} = process_input_data(input, model, length(input) ÷ prod(size(model)))
process_input_data(input::judiVector{T, AT}, ::Geometry{T}) where {T<:Real, AT} = input
process_input_data(input::judiVector{T, AT}) where {T<:Real, AT} = input.data
process_input_data(input::judiWeights{T}, ::AbstractModel{T, N}) where {T<:Real, N} = input.weights

function process_input_data(input::DenseArray{T}, v::Vector{<:Array}) where T
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

Reshapes input vector into a 3D `nt x nrec x nsrc` Array.
"""
function reshape(x::AbstractArray{T}, geometry::Geometry{T}) where {T<:Real}
    nt = geometry.nt[1]
    nrec = geometry.nrec[1]
    nsrc = Int(length(x) / nt / nrec)
    return reshape(x, nt, nrec, nsrc)
end

"""
    reshape(V::Vector{T}, n::Tuple, nblock::Integer)

Reshapes input vector into a `Vector{Array{T, N}}` of length `nblock` with each subarray of size `n`
"""
function reshape(x::Vector{T}, d::Dims, nsrc::Integer) where {T<:Real}
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
function transducer(q::judiVector{T, AT}, d::Tuple, r::Number, theta) where {T<:Real, AT}
    length(theta) != length(q.geometry.xloc) && throw("Need one angle per source position")
    size(q.data[1], 2) > 1 && throw("Only point sources can be converted to transducer source")
    # Array of source
    nsrc_loc = 11
    nsrc = length(q.geometry.xloc)
    x_base = collect(range(-r, r, length=nsrc_loc))
    y_base = zeros(nsrc_loc)
    y_base_b = zeros(nsrc_loc) .- d[end]

    # New coords and data
    xloc = Vector{Vector{T}}(undef, nsrc)
    yloc = Vector{Vector{T}}(undef, nsrc)
    zloc = Vector{Vector{T}}(undef, nsrc)
    data = Vector{Matrix{T}}(undef, nsrc)
    t = q.geometry.t[1]
    dt = q.geometry.dt[1]

    for i=1:nsrc
        # Build the rotated array of dipole
        R = T.([cos(theta[i] - pi/2) sin(theta[i] - pi/2);-sin(theta[i] - pi/2) cos(theta[i] - pi/2)])
        # +1 coords
        r_loc = R * [x_base';y_base']
        # -1 coords
        r_loc_b = R * [x_base';y_base_b']
        xloc[i] = q.geometry.xloc[i] .+ vec(vcat(r_loc[1, :], r_loc_b[1, :]))
        zloc[i] = q.geometry.zloc[i] .+ vec(vcat(r_loc[2, :], r_loc_b[2, :]))
        yloc[i] = zeros(T, 2*nsrc_loc)
        data[i] = zeros(T, length(q.data[i]), 2*nsrc_loc)
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

"""
    as_vec(x, ::Val{Bool})
Vectorizes output when `return_array` is set to `true`.
"""
as_vec(x, ::Val) = x
as_vec(x::Tuple, v::Val) = tuple((as_vec(xi, v) for xi in x)...)
as_vec(x::Ref, ::Val) = x[]
as_vec(x::PhysicalParameter, ::Val{true}) = vec(x.data)
as_vec(x::judiMultiSourceVector, ::Val{true}) = vec(x)


######### backward compat
extend_gradient(model_full, model, array) = array

### Filter out PyObject none and nothing
pynone(::AbstractArray) = false
pynone(m) = (m == PyObject(nothing) || isnothing(m))

function filter_none(args::Tuple)
    out = filter(m-> ~pynone(m), args)
    out = length(out) == 1 ? out[1] : out
    return out
end

filter_none(x) = x
