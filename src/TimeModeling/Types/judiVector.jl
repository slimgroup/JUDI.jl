############################################################
# judiVector # #################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiVector, judiVectorException, subsample, judiVector_to_SeisBlock, src_to_SeisBlock
export time_resample, time_resample!, judiTimeInterpolation
export write_shot_record, get_data, convert_to_array, rebuild_jv

############################################################

# structure for seismic data as an abstract vector
mutable struct judiVector{vDT<:Number, AT} <: joAbstractLinearOperator{vDT,vDT}
    name::String
    m::Integer
    n::Integer
    nsrc::Integer
    geometry::Geometry
    data::Array{AT, 1}
end

mutable struct judiVectorException <: Exception
    msg :: String
end

############################################################

## outer constructors

"""
    judiVector
        name::String
        m::Integer
        n::Integer
        nsrc::Integer
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
    # length of vector
    n = 1
    nsrc = get_nsrc(geometry)
    m = n_samples(geometry, nsrc)

    dataCell = Array{Array{T, N}, 1}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = deepcopy(data)
    end
    return judiVector{T, Array{T, N}}("Seismic data vector", m, n, nsrc, geometry, dataCell)
end

function judiVector(geometry::Geometry, data::Vector{T}) where {T, N}
    return judiVector(geometry, reshape(data, length(data),1))
end

# constructor if data is passed as a cell array
function judiVector(geometry::Geometry, data::Vector{Array{T, N}}) where {T, N}
    T == Float32 || (data = tof32.(data))

    # length of vector
    nsrc = get_nsrc(geometry)
    m = n_samples(geometry, nsrc)

    n = 1
    return judiVector{T, Array{T, N}}("Seismic data vector",m,n,nsrc,geometry,data)
end


############################################################
# constructors from SEGY files or out-of-core containers

# contructor for in-core data container
function judiVector(data::SegyIO.SeisBlock; segy_depth_key="RecGroupElevation")
    # length of data vector
    src = get_header(data,"FieldRecord")
    nsrc = length(unique(src))

    numTraces = get_header(data,"TraceNumber")[end] - get_header(data,"TraceNumber")[1] + 1
    numSamples = get_header(data,"ns")[end]
    m = numTraces*numSamples
    n = 1

    # extract geometry from data container
    geometry = Geometry(data; key="receiver", segy_depth_key=segy_depth_key)

    # fill data vector with pointers to data location
    dataCell = Array{Array{Float32, 2}, 1}(undef, nsrc)
    for j=1:nsrc
        traces = findall(src .== unique(src)[j])
        dataCell[j] = convert(Array{Float32, 2}, data.data[:,traces])
    end

    return judiVector{Float32, Array{Float32, 2}}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for in-core data container and given geometry
function judiVector(geometry::Geometry, data::SegyIO.SeisBlock)

    # length of data vector
    src = get_header(data,"FieldRecord")
    nsrc = length(unique(src))

    numTraces = get_header(data,"TraceNumber")[end] - get_header(data,"TraceNumber")[1] + 1
    numSamples = get_header(data,"ns")[end]
    m = numTraces*numSamples
    n = 1

    # fill data vector with pointers to data location
    dataCell = Array{Array{Float32, 2}, 1}(undef, nsrc)
    for j=1:nsrc
        traces = findall(src .== unique(src)[j])
        dataCell[j] = convert(Array{Float32, 2}, data.data[:,traces])
    end

    return judiVector{Float32, Array{Float32, 2}}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from single container
function judiVector(data::SegyIO.SeisCon; segy_depth_key="RecGroupElevation")
    # length of data vector
    nsrc = length(data)
    numTraces = 0
    for j=1:nsrc
        numTraces += Int((data.blocks[j].endbyte - data.blocks[j].startbyte)/(240 + data.ns*4))
    end
    m = numTraces*data.ns
    n = 1

    # extract geometry from data container
    geometry = Geometry(data; key="receiver", segy_depth_key=segy_depth_key)

    # fill data vector with pointers to data location
    dataCell = Array{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = split(data,j)
    end

    return judiVector{Float32, SegyIO.SeisCon}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for single out-of-core data container and given geometry
function judiVector(geometry::Geometry, data::SegyIO.SeisCon; vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain and range types not supported"))

    # length of data vector
    nsrc = length(data)
    numTraces = 0
    for j=1:nsrc
        numTraces += Int((data.blocks[j].endbyte - data.blocks[j].startbyte)/(240 + data.ns*4))
    end
    m = numTraces*data.ns
    n = 1

    # fill data vector with pointers to data location
    dataCell = Array{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = split(data,j)
    end

    return judiVector{Float32, SegyIO.SeisCon}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from cell array of containers
function judiVector(data::Array{SegyIO.SeisCon,1}; segy_depth_key="RecGroupElevation")

    # length of data vector
    nsrc = length(data)
    numTraces = 0
    for j=1:nsrc
        numTraces += Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data[j].ns*4))
    end
    m = numTraces*data[1].ns    # SEGY requires same number of samples for every trace
    n = 1

    # extract geometry from data container
    geometry = Geometry(data; key="receiver", segy_depth_key=segy_depth_key)

    # fill data vector with pointers to data location
    dataCell = Array{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = data[j]
    end

    return judiVector{Float32, SegyIO.SeisCon}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from cell array of containers and given geometry
function judiVector(geometry::Geometry, data::Vector{SegyIO.SeisCon})
    # length of data vector
    nsrc = length(data)
    numTraces = 0
    for j=1:nsrc
        numTraces += Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data[j].ns*4))
    end
    m = numTraces*data[1].ns
    n = 1

    # fill data vector with pointers to data location
    dataCell = Array{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = data[j]
    end

    return judiVector{Float32, SegyIO.SeisCon}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

############################################################
## overloaded Base functions

# conj(jo)
conj(a::judiVector{vDT, AT}) where {vDT, AT} =
    judiVector{vDT, AT}("conj("*a.name*")",a.m,a.n,a.nsrc,a.geometry,a.data)

# transpose(jo)
transpose(a::judiVector{vDT, AT}) where {vDT, AT} =
    judiVector{vDT, AT}(""*a.name*".'",a.n,a.m,a.nsrc,a.geometry,a.data)

# adjoint(jo)
adjoint(a::judiVector{vDT, AT}) where {vDT, AT} =
        judiVector{vDT, AT}(""*a.name*".'",a.n,a.m,a.nsrc,a.geometry,a.data)

##########################################################

# Overload needed base function for SegyIO objects
vec(x::SegyIO.SeisCon) = vec(x[1].data)
abs(x::SegyIO.IBMFloat32) = abs(Float32(x))

##########################################################

for opo=[:+, :-, :*, :/]
    @eval begin
        $opo(a::judiVector{avDT, SeisCon}, ::T) where {avDT, T<:Real} = throw(DomainError(a, "Addition for OOC judiVectors not supported."))
        $opo(::T, b::judiVector{bvDT, SeisCon}) where {bvDT, T<:Real} = throw(DomainError(b, "Addition for OOC judiVectors not supported."))
        $opo(::judiVector{avDT, SeisCon}, b::judiVector{bvDT, SeisCon}) where {avDT, bvDT} = throw(DomainError(b, "Addition for OOC judiVectors not supported."))
    end
end

for ipop=[:lmul!, :rmul!, :rdiv!, :ldiv!]
    @eval begin
        $ipop(a::judiVector{avDT, SeisCon}, ::T) where {avDT, T<:Real} = throw(DomainError(a, "Addition for OOC judiVectors not supported."))
        $ipop(::T, b::judiVector{bvDT, SeisCon}) where {bvDT, T<:Real} = throw(DomainError(b, "Addition for OOC judiVectors not supported."))
        $ipop(::judiVector{avDT, SeisCon}, b::judiVector{bvDT, SeisCon}) where {avDT, bvDT} = throw(DomainError(b, "Addition for OOC judiVectors not supported."))
    end
end

# *(joLinearFunction, judiVector)
function *(A::joLinearFunction{ADDT,ARDT},v::judiVector{avDT, AT}) where {ADDT, ARDT, avDT, AT}
    A.n == size(v,1) || throw(judiVectorException("Shape mismatch: A:$((A.m, A.n)), v: $(size(v))"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiVector):",A.name,typeof(A),avDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(joLinearOperator, judiVector)
function *(A::joLinearOperator{ADDT,ARDT},v::judiVector{avDT, AT}) where {ADDT, ARDT, avDT, AT}
    A.n == size(v,1) || throw(judiVectorException("Shape mismatch: A:$((A.m, A.n)), v: $(size(v))"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiVector):",A.name,typeof(A),avDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# vcat
function vcat(ai::Vararg{judiVector{avDT, AT}, N}) where {avDT, AT, N}
    N == 1 && (return ai[1])
    N > 2 && (return vcat(ai[1], vcat(ai[2:end]...)))
    a, b = ai
    typeof(a.geometry) == typeof(b.geometry) || throw(judiVectorException("Geometry type mismatch"))
    m = a.m + b.m
    n = 1
    nsrc = a.nsrc + b.nsrc

    data = vcat(a.data, b.data)

    if AT == SegyIO.SeisCon
        nsamples = vcat(a.geometry.nsamples, b.geometry.nsamples)
    else
        xloc = vcat(a.geometry.xloc, b.geometry.xloc)
        yloc = vcat(a.geometry.yloc, b.geometry.yloc)
        zloc = vcat(a.geometry.zloc, b.geometry.zloc)
    end

    dt = vcat(a.geometry.dt, b.geometry.dt)
    nt = vcat(a.geometry.nt, b.geometry.nt)
    t = vcat(a.geometry.t, b.geometry.t)

    if AT == SegyIO.SeisCon
        geometry = GeometryOOC{Float32}(data,dt,nt,t,nsamples,a.geometry.key,a.geometry.segy_depth_key)
    else
        geometry = GeometryIC{Float32}(xloc, yloc, zloc, dt, nt, t)
    end

    return judiVector{avDT, AT}(a.name, m, n, nsrc, geometry, data)
end

# push!
function push!(a::judiVector{T, mT}, b::judiVector{T, mT}) where {T, mT}
    append!(a.data, b.data)
    a.nsrc += b.nsrc
    a.m += b.m
    push!(a.geometry, b.geometry)
end

# dot product
function dot(a::judiVector{avDT, AT}, b::judiVector{bvDT, AT}) where {avDT, bvDT, AT}
# Dot product for data containers
    size(a) == size(b) || throw(judiVectorException("Size mismatch: a:$(size(a)) b:$(size(b))"))
    compareGeometry(a.geometry, b.geometry) == 1 || throw(judiVectorException("Geometry mismatch"))
    dotprod = 0f0
    for j=1:a.nsrc
        dotprod += a.geometry.dt[j] * dot(vec(a.data[j]),vec(b.data[j]))
    end
    return dotprod
end

# norm
function norm(a::judiVector{avDT, AT}, p::Real=2) where {avDT, AT}
    if p == Inf
        return max([maximum(abs.(a.data[i])) for i=1:a.nsrc]...)
    end
    x = 0.f0
    for j=1:a.nsrc
        x += a.geometry.dt[j] * sum(abs.(vec(a.data[j])).^p)
    end
    return x^(1.f0/p)
end


# abs
function abs(a::judiVector{avDT, AT}) where {avDT, AT}
    b = deepcopy(a)
    for j=1:a.nsrc
        b.data[j] = abs.(a.data[j])
    end
    return b
end

# Subsample data container
"""
    subsample(x,source_numbers)

Subsample seismic data vectors or matrix-free linear operators and extract the entries that correspond\\
to the shot positions defined by `source_numbers`. Works for inputs of type `judiVector`, `judiModeling`, \\
`judiProjection`, `judiJacobian`, `Geometry`, `judiRHS`, `judiPDE`, `judiPDEfull`.

Examples
========

(1) Extract 2 shots from `judiVector` vector:

    dsub = subsample(dobs,[1,2])

(2) Extract geometry for shot location 100:

    geometry_sub = subsample(dobs.geometry,100)

(3) Extract Jacobian for shots 10 and 20:

    Jsub = subsample(J,[10,20])

"""
function subsample(a::judiVector{avDT, AT},srcnum) where {avDT, AT}
    geometry = subsample(a.geometry,srcnum)     # Geometry of subsampled data container
    return judiVector(geometry,a.data[srcnum])
end

getindex(x::judiVector,a) = subsample(x,a)

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
        set_header!(blocks[j], "GroupX", convert(Array{Integer,1},round.(d.geometry.xloc[j]*1f3)))
        if length(d.geometry.yloc[j]) == 1
            set_header!(blocks[j], "GroupY", Int(round(d.geometry.yloc[j][1]*1f3)))
        else
            set_header!(blocks[j], "GroupY", convert(Array{Integer,1},round.(d.geometry.yloc[j]*1f3)))
        end
        set_header!(blocks[j], receiver_depth_key, convert(Array{Integer,1},round.(d.geometry.zloc[j]*1f3)))
        set_header!(blocks[j], "SourceX", Int(round.(q.geometry.xloc[j][1]*1f3)))
        set_header!(blocks[j], "SourceY", Int(round.(q.geometry.yloc[j][1]*1f3)))
        set_header!(blocks[j], source_depth_key, Int(round.(q.geometry.zloc[j][1]*1f3)))

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

# Create SeisBlock from a single judiVector (source)
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
    x.m = 0
    for j=1:x.nsrc
        dataInterp, geom = time_resample(x.data[j], subsample(x.geometry, j), dt_new)
        x.data[j] = dataInterp
        x.geometry.dt[j] = dt_new
        x.geometry.nt[j] = geom.nt[1]
        x.geometry.t[j] = geom.t[1]
        x.m += prod(size(dataInterp))
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
    m = 0
    n = 0
    for j=1:nsrc
        nt_coarse = Int(trunc(geometry.t[j]/dt_coarse + 1))
        nt_fine = Int(trunc(geometry.t[j]/dt_fine + 1))
        n += length(geometry.xloc[j])*nt_coarse
        m += length(geometry.xloc[j])*nt_fine
    end
    I = joLinearFunctionFwd_T(m,n,
                              v -> time_resample(v, dt_fine),
                              w -> time_resample(w, dt_coarse),
                              Float32,Float32,name="Time resampling")
    return I
end

####################################################################################################
# Indexing

setindex!(x::judiVector, y, i) = x.data[i][:] = y

firstindex(x::judiVector) = 1

lastindex(x::judiVector) = x.nsrc

axes(x::judiVector) = Base.OneTo(x.nsrc)

ndims(x::judiVector) = length(size(x))

similar(x::judiVector) = 0f0*x

similar(x::judiVector, element_type::DataType, dims::Union{AbstractUnitRange, Integer}...) = 0f0*x

function fill!(x::judiVector{vDT, AT}, val) where {vDT, AT}
    for j=1:length(x.data)
        fill!(x.data[j], val)
    end
end

function sum(x::judiVector)
    s = 0f0
    for j=1:length(x.data)
        s += sum(vec(x.data[j]))
    end
    return s
end

Base.IteratorSize(d::judiVector) = Base.SizeUnknown()
isfinite(v::judiVector) = all(all(isfinite.(v.data[i])) for i=1:v.nsrc)
iterate(S::judiVector, state::Integer=1) = state > S.nsrc ? nothing : (S.data[state], state+1)

####################################################################################################

# Integration/differentiation of shot records

function cumsum(x::judiVector;dims=1)
    y = deepcopy(x)
    cumsum!(y, x; dims=dims)
    return y
end

function cumsum!(y::judiVector, x::judiVector;dims=1)
    dims == 1 || dims == 2 || throw(judiVectorException("Dimension $(dims) is out of range for a 2D array"))
    h = dims == 1 ? x.geometry.dt[1] : 1f0              # receiver dimension is non-dimensional
    for i = 1:x.nsrc
        cumsum!(y.data[i], x.data[i], dims=dims)
    end
    rmul!(y, h)
    return y
end

function diff(x::judiVector;dims=1)
    # note that this is not the same as default diff in julia, the first entry stays in the diff result
    dims == 1 || dims == 2 || throw(judiVectorException("Dimension $(dims) is out of range for a 2D array"))
    y = (dims == 1 ? 1f0 / x.geometry.dt[1] : 1f0) * x        # receiver dimension is non-dimensional
    for i = 1:x.nsrc
        copy!(selectdim(y.data[i], dims, 2:size(y.data[i],dims)), diff(y.data[i],dims=dims))
    end
    return y
end

####################################################################################################

BroadcastStyle(::Type{judiVector}) = Base.Broadcast.DefaultArrayStyle{1}()

ndims(::Type{judiVector{Float32, Array{Float32, 2}}}) = 1

### +/- ####
broadcasted(::typeof(+), x::judiVector, y::judiVector) = x + y
broadcasted(::typeof(-), x::judiVector, y::judiVector) = x - y

broadcasted(::typeof(+), x::judiVector, y::Number) = x + y
broadcasted(::typeof(-), x::judiVector, y::Number) = x - y

broadcasted(::typeof(+), x::Number, y::judiVector) = x + y
broadcasted(::typeof(-), x::Number, y::judiVector) = x - y

### * ####
function broadcasted(::typeof(*), x::judiVector, y::judiVector)
    size(x) == size(y) || throw(judiVectorException("Size mismatch: x:$(size(x)), y:$(size(y))"))
    compareGeometry(x.geometry, y.geometry) == 1 || throw(judiVectorException("Geometry mismatch"))
    typeof(x.data[1]) == SeisCon && throw("Addition for OOC judiVectors not supported.")
    typeof(y.data[1]) == SeisCon && throw("Addition for OOC judiVectors not supported.")
    z = deepcopy(x)
    for j=1:length(x.data)
        z.data[j] = x.data[j] .* y.data[j]
    end
    return z
end

function broadcasted(::typeof(*), x::judiVector, y::Number)
    z = deepcopy(x)
    for j=1:length(x.data)
        z.data[j] .*= y
    end
    return z
end

broadcasted(::typeof(*), x::Number, y::judiVector) = broadcasted(*, y, x)

### / ####
function broadcasted(::typeof(/), x::judiVector, y::judiVector)
    size(x) == size(y) || throw(judiVectorException("Size mismatch: x:$(size(x)), y:$(size(y))"))
    compareGeometry(x.geometry, y.geometry) == 1 || throw(judiVectorException("geometry mismatch"))
    typeof(x.data[1]) == SeisCon && throw("Addition for OOC judiVectors not supported.")
    typeof(y.data[1]) == SeisCon && throw("Addition for OOC judiVectors not supported.")
    z = deepcopy(x)
    for j=1:length(x.data)
        z.data[j] = x.data[j] ./ y.data[j]
    end
    return z
end

broadcasted(::typeof(/), x::judiVector, y::Number) = broadcasted(*, x, 1/y)

# Materialize for broadcasting
function materialize!(x::judiVector, y::judiVector)
    for j=1:length(x.data)
        x.data[j] .= y.data[j]
    end
end

function broadcast!(identity, x::judiVector, y::judiVector)
    copy!(x,y)
end

function broadcasted(identity, x::judiVector)
    return x
end


function copy!(x::judiVector, y::judiVector)
    for j=1:x.nsrc
        x.data[j] .= y.data[j]
    end
    x.geometry = deepcopy(y.geometry)
end

copy(x::judiVector) = 1f0 * x

function get_data(x::judiVector{T, SeisCon}) where T
    shots = Array{Array{Float32, 2}, 1}(undef, x.nsrc)
    rec_geometry = Geometry(x.geometry)
    for j=1:x.nsrc
        shots[j] = convert(Array{Float32, 2}, x.data[j][1].data)
    end
    return judiVector(rec_geometry, shots)
end

get_data(x::judiVector{T, Array{Float32, 2}}) where T = x

function convert_to_array(x::judiVector)
    y = vec(x.data[1])
    if x.nsrc > 1
        for j=2:x.nsrc
            y = [y; vec(x.data[j])]
        end
    end
    return y
end

function isapprox(x::judiVector, y::judiVector; rtol::Real=sqrt(eps()), atol::Real=0)
    compareGeometry(x.geometry, y.geometry) == 1 || throw(judiVectorException("Geometry mismatch"))
    isapprox(x.data, y.data; rtol=rtol, atol=atol)
end


############################################################

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

# Rebuild for backward compatinility
convgeom(x) = GeometryIC{Float32}([getfield(x.geometry, s) for s=fieldnames(GeometryIC)]...)
convdata(x) = convert(Array{Array{Float32, 2}, 1}, x.data)

"""
    reuild_jv(v)
rebuild a judiVector from previous version type or JLD2 reconstructed type
"""
rebuild_jv(v::judiVector{T, AT}) where {T, AT} = v
rebuild_jv(v) = judiVector(convgeom(v), convdata(v))