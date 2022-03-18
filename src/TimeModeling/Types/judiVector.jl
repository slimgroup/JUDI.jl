############################################################
# judiVector # #################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiVector, judiVector_to_SeisBlock, src_to_SeisBlock
export time_resample, time_resample!, judiTimeInterpolation
export write_shot_record, get_data, convert_to_array, rebuild_jv

############################################################

# structure for seismic data as an abstract vector
mutable struct judiVector{T, AT} <: judiMultiSourceVector{T}
    nsrc::Integer
    geometry::Geometry
    data::Vector{AT}
end

############################################################

## outer constructors

"""
    judiVector
        geometry::Geometry
        data

Abstract vector for seismic data. This vector-like structure contains the geometry and data for either\\
receiver data (shot records) or source data (wavelets).

Constructors
============

Construct vector from `Geometry` structure and cell array of shot records or wavelets. The `data` keyword\\
can also be a single (non-cell) array, in which case the data is the same for all source positions:

    judiVector(geometry, data)

Construct vector for observed data from `SegyIO.SeisBlock`. `segy_depth_key` is the `SegyIO` keyword \\
that contains the receiver depth coordinate:

    judiVector(SegyIO.SeisBlock; segy_depth_key="RecGroupElevation")

Construct vector for observed data from out-of-core data container of type `SegyIO.SeisCon`:

    judiVector(SegyIO.SeisCon; segy_depth_key="RecGroupElevation")

Examples
========

(1) Construct data vector from `Geometry` structure and a cell array of shot records:

    dobs = judiVector(rec_geometry, shot_records)

(2) Construct data vector for a seismic wavelet (can be either cell arrays of individual\\
wavelets or a single wavelet as an array):

    q = judiVector(src_geometry, wavelet)

(3) Construct data vector from `SegyIO.SeisBlock` object:

    using SegyIO
    seis_block = segy_read("test_file.segy")
    dobs = judiVector(seis_block; segy_depth_key="RecGroupElevation")

(4) Construct out-of-core data vector from `SegyIO.SeisCon` object (for large SEG-Y files):

    using SegyIO
    seis_container = segy_scan("/path/to/data/directory","filenames",["GroupX","GroupY","RecGroupElevation","SourceDepth","dt"])
    dobs = judiVector(seis_container; segy_depth_key="RecGroupElevation")

"""
function judiVector(geometry::Geometry, data::Array{T, N}) where {T, N}
    T == Float32 || (data = tof32(data))
    N < 3 || throw(judiMultiSourceException("Only 1D (trace) and 2D (record) input data supported"))
    nsrc = get_nsrc(geometry)
    dataCell = Vector{Array{T, N}}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = deepcopy(data)
    end
    return judiVector{T, Array{T, N}}(nsrc, geometry, dataCell)
end

# constructor if data is passed as a cell array
function judiVector(geometry::Geometry, data::Vector{Array{T, N}}) where {T, N}
    T == Float32 || (data = tof32.(data))
    nsrcd = length(data)
    nsrcg = get_nsrc(geometry)
    nsrcd == nsrcg || throw(judiMultiSourceException("Number of sources in geometry and data don't match $(nsrcd) != $(nsrcg)"))
    return judiVector{T, Array{T, N}}(nsrcd, geometry, data)
end

# contructor for in-core data container and given geometry
function judiVector(geometry::Geometry, data::SegyIO.SeisBlock)
    # length of data vector
    src = get_header(data,"FieldRecord")
    nsrc = length(unique(src))
    # fill data vector with pointers to data location
    dataCell = Vector{Array{Float32, 2}}(undef, nsrc)
    for j=1:nsrc
        traces = findall(src .== unique(src)[j])
        dataCell[j] = convert(Array{Float32, 2}, data.data[:,traces])
    end
    return judiVector{Float32, Array{Float32, 2}}(nsrc, geometry, dataCell)
end

# contructor for single out-of-core data container and given geometry
function judiVector(geometry::Geometry, data::SegyIO.SeisCon)
    # length of data vector
    nsrc = length(data)
    # fill data vector with pointers to data location
    dataCell = Vector{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = split(data,j)
    end
    return judiVector{Float32, SegyIO.SeisCon}(nsrc, geometry,dataCell)
end

judiVector(data::SegyIO.SeisBlock; segy_depth_key="RecGroupElevation") = judiVector(Geometry(data; key="receiver", segy_depth_key=segy_depth_key), data)
judiVector(data::SegyIO.SeisCon; segy_depth_key="RecGroupElevation")= judiVector(Geometry(data; key="receiver", segy_depth_key=segy_depth_key), data)
judiVector(data::Vector{SegyIO.SeisCon}; segy_depth_key="RecGroupElevation") = judiVector(Geometry(data; key="receiver", segy_depth_key=segy_depth_key), data)
judiVector(geometry::Geometry, data::Vector{SegyIO.SeisCon}) =  judiVector{Float32, SegyIO.SeisCon}(length(data), geometry, data)

############################################################
## overloaded multi_source functions
time_sampling(jv::judiVector) = jv.geometry.dt

############################################################
# JOLI conversion
jo_convert(::Type{T}, jv::judiVector{T, Array{T, N}}, ::Bool) where {T<:Real, N} = jv
jo_convert(::Type{T}, jv::judiVector{vT, Array{vT, N}}, B::Bool) where {T<:Real, vT, N} = judiVector{T, Array{T, N}}(jv.nsrc, jv.geometry, jo_convert.(T, jv.data, B))
zero(::Type{T}, v::judiVector{vT, AT}) where {T, vT, AT} = judiVector{T, AT}(v.nsrc, deepcopy(v.geometry), T(0) .* v.data)
function copy!(jv::judiVector, jv2::judiVector)
    jv.geometry = deepcopy(jv2.geometry)
    jv.data .= jv2.data
    jv
end

copyto!(jv::judiVector, jv2::judiVector) = copy!(jv, jv2)
make_input(jv::judiVector) = jv.data[1]

check_compat(ms::Vararg{judiVector, N}) where N = all(y -> compareGeometry(y.geometry, first(ms).geometry), ms)
##########################################################

# Overload needed base function for SegyIO objects
vec(x::SegyIO.SeisCon) = vec(x[1].data)
abs(x::SegyIO.IBMFloat32) = abs(Float32(x))
*(n::Number, s::SegyIO.SeisCon) = copy(s)

# push!
function push!(a::judiVector{T, mT}, b::judiVector{T, mT}) where {T, mT}
    typeof(a.geometry) == typeof(b.geometry) || throw(judiMultiSourceException("Geometry type mismatch"))
    append!(a.data, b.data)
    a.nsrc += b.nsrc
    push!(a.geometry, b.geometry)
end

# getindex data container
"""
    getindex(x,source_numbers)

getindex seismic data vectors or matrix-free linear operators and extract the entries that correspond\\
to the shot positions defined by `source_numbers`. Works for inputs of type `judiVector`, `judiModeling`, \\
`judiProjection`, `judiJacobian`, `Geometry`, `judiRHS`, `judiPDE`, `judiPDEfull`.

Examples
========

(1) Extract 2 shots from `judiVector` vector:

    dsub = getindex(dobs,[1,2])

(2) Extract geometry for shot location 100:

    geometry_sub = getindex(dobs.geometry,100)

(3) Extract Jacobian for shots 10 and 20:

    Jsub = getindex(J,[10,20])

"""
function getindex(a::judiVector{avDT, AT}, srcnum::RangeOrVec) where {avDT, AT}
    geometry = getindex(a.geometry, srcnum)     # Geometry of getindexd data container
    return judiVector{avDT, AT}(length(srcnum), geometry, a.data[srcnum])
end

# Create SeisBlock from judiVector container to write to file
function judiVector_to_SeisBlock(d::judiVector{avDT, AT}, q::judiVector{avDT, QT};
                                 source_depth_key="SourceSurfaceElevation",
                                 receiver_depth_key="RecGroupElevation") where {avDT, AT, QT}

    typeof(d.geometry) <: GeometryOOC && (d.geometry = Geometry(d.geometry))
    typeof(q.geometry) <: GeometryOOC && (q.geometry = Geometry(q.geometry))

    blocks = Array{Any}(undef, d.nsrc)
    count = 0
    for j=1:d.nsrc

        # create SeisBlock
        blocks[j] = SeisBlock(d.data[j])
        numTraces = size(d.data[j],2)
        traceNumbers = convert(Array{Integer,1},count+1:count+numTraces)

        # set headers
        set_header!(blocks[j], "GroupX", Int.(round.(d.geometry.xloc[j]*1f3)))
        if length(d.geometry.yloc[j]) == 1
            set_header!(blocks[j], "GroupY", Int(round(d.geometry.yloc[j][1]*1f3)))
        else
            set_header!(blocks[j], "GroupY", convert(Array{Integer,1},round.(d.geometry.yloc[j]*1f3)))
        end
        set_header!(blocks[j], receiver_depth_key, Int.(round.(d.geometry.zloc[j]*1f3)))
        set_header!(blocks[j], "SourceX", Int.(round.(q.geometry.xloc[j][1]*1f3)))
        set_header!(blocks[j], "SourceY", Int.(round.(q.geometry.yloc[j][1]*1f3)))
        set_header!(blocks[j], source_depth_key, Int.(round.(q.geometry.zloc[j][1]*1f3)))

        set_header!(blocks[j], "dt", Int(d.geometry.dt[j]*1f3))
        set_header!(blocks[j], "FieldRecord",j)
        set_header!(blocks[j], "TraceNumWithinLine", traceNumbers)
        set_header!(blocks[j], "TraceNumWithinFile", traceNumbers)
        set_header!(blocks[j], "TraceNumber", traceNumbers)
        set_header!(blocks[j], "ElevationScalar", -1000)
        set_header!(blocks[j], "RecSourceScalar", -1000)
        count += numTraces
    end

    # merge into single block
    fullblock = blocks[1]
    for j=2:d.nsrc
        fullblock = merge(fullblock,blocks[j])
        blocks[j] = []
    end
    return fullblock
end

function src_to_SeisBlock(q::judiVector{avDT, QT};
                                 source_depth_key="SourceSurfaceElevation",
                                 receiver_depth_key="RecGroupElevation") where {avDT, QT}

    return judiVector_to_SeisBlock(q, q;
        source_depth_key=source_depth_key,
        receiver_depth_key=receiver_depth_key)
end

function write_shot_record(srcGeometry, srcData, recGeometry, recData,options)
    q = judiVector(srcGeometry, srcData)
    d = judiVector(recGeometry, recData)
    pos = [srcGeometry.xloc[1][1], srcGeometry.yloc[1][1],  srcGeometry.zloc[1][1]]
    pos = join(["_"*string(trunc(p; digits=2)) for p in pos])
    file = join([string(options.file_name), pos,".segy"])
    block_out = judiVector_to_SeisBlock(d, q)
    segy_write(join([options.file_path,"/",file]), block_out)
    container = scan_file(join([options.file_path,"/",file]),
                          ["GroupX", "GroupY", "dt", "SourceSurfaceElevation", "RecGroupElevation"];
                          chunksize=256)
    return container
end


function time_resample!(x::judiVector, dt_new; order=2)
    for j=1:x.nsrc
        dataInterp, geom = time_resample(x.data[j], getindex(x.geometry, j), dt_new)
        x.data[j] = dataInterp
        x.geometry.dt[j] = dt_new
        x.geometry.nt[j] = geom.nt[1]
        x.geometry.t[j] = geom.t[1]
    end
    return x
end

function time_resample(x::judiVector, dt_new; order=2)
    xout = deepcopy(x)
    time_resample!(xout, dt_new; order=order)
    return xout
end

function judiTimeInterpolation(geometry::Geometry, dt_coarse, dt_fine)
# Time interpolation as a linear operator (copies input data)

    nsrc = length(geometry.xloc)
    I = joLinearFunctionFwd_T(nsrc, nsrc,
                              v -> time_resample(v, dt_fine),
                              w -> time_resample(w, dt_coarse),
                              Float32,Float32,name="Time resampling")
    return I
end

####################################################################################################

# Integration/differentiation of shot records

function cumsum(x::judiVector;dims=1)
    y = deepcopy(x)
    cumsum!(y, x; dims=dims)
    return y
end

function cumsum!(y::judiVector, x::judiVector;dims=1)
    dims == 1 || dims == 2 || throw(judiMultiSourceException("Dimension $(dims) is out of range for a 2D array"))
    h = dims == 1 ? x.geometry.dt[1] : 1f0              # receiver dimension is non-dimensional
    for i = 1:x.nsrc
        cumsum!(y.data[i], x.data[i], dims=dims)
    end
    rmul!(y, h)
    return y
end

function diff(x::judiVector;dims=1)
    # note that this is not the same as default diff in julia, the first entry stays in the diff result
    dims == 1 || dims == 2 || throw(judiMultiSourceException("Dimension $(dims) is out of range for a 2D array"))
    y = (dims == 1 ? 1f0 / x.geometry.dt[1] : 1f0) * x        # receiver dimension is non-dimensional
    for i = 1:x.nsrc
        copy!(selectdim(y.data[i], dims, 2:size(y.data[i],dims)), diff(y.data[i],dims=dims))
    end
    return y
end


function get_data(x::judiVector{T, SeisCon}) where T
    shots = Array{Array{Float32, 2}, 1}(undef, x.nsrc)
    rec_geometry = Geometry(x.geometry)
    for j=1:x.nsrc
        shots[j] = convert(Array{Float32, 2}, x.data[j][1].data)
    end
    return judiVector(rec_geometry, shots)
end

get_data(x::judiVector{T, Array{Float32, 2}}) where T = x

convert_to_array(x::judiVector) = vcat(vec.(x.data)...)

## in place
function A_mul_B!(x::judiWeights, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiVector)
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.weights)
        x.weights[j] .= z.weights[j]
    end
end

function A_mul_B!(x::judiVector, F::Union{joAbstractLinearOperator, joLinearFunction}, y::Union{Array, PhysicalParameter})
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.data)
        x.data[j] .= z.data[j]
    end
end

function A_mul_B!(x::Union{Array, PhysicalParameter}, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiVector)
    F.m == size(y, 1) ? x[:] .= (adjoint(F)*y)[:] : x[:] .= (F*y)[:]
end

function A_mul_B!(x::judiVector, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiWeights)
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.data)
        x.data[j] .= z.data[j]
    end
end

function A_mul_B!(x::judiVector, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiVector)
    F.m == size(y, 1) ? z = adjoint(F)*y : z = F*y
    for j=1:length(x.data)
        x.data[j] .= z.data[j]
    end
end

mul!(x::judiWeights, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiVector) = A_mul_B!(x, F, y)
mul!(x::judiVector, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiWeights) = A_mul_B!(x, F, y)
mul!(x::judiVector, F::Union{joAbstractLinearOperator, joLinearFunction}, y::judiVector) = A_mul_B!(x, F, y)
mul!(x::Union{Array, PhysicalParameter}, J::Union{joAbstractLinearOperator, joLinearFunction}, y::judiVector) = A_mul_B!(x, J, y)
mul!(x::judiVector, J::Union{joAbstractLinearOperator, joLinearFunction}, y::Union{Array, PhysicalParameter}) = A_mul_B!(x, J, y)

"""
    reuild_jv(v)
rebuild a judiVector from previous version type or JLD2 reconstructed type
"""
rebuild_jv(v::judiVector{T, AT}) where {T, AT} = v
rebuild_jv(v) = judiVector(convgeom(v), convdata(v))

# Rebuild for backward compatinility
convgeom(x) = GeometryIC{Float32}([getfield(x.geometry, s) for s=fieldnames(GeometryIC)]...)
convdata(x) = convert(Array{Array{Float32, 2}, 1}, x.data)
