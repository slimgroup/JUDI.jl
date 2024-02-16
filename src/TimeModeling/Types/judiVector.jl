############################################################
# judiVector # #################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiVector, judiVector_to_SeisBlock, src_to_SeisBlock
export write_shot_record, get_data, convert_to_array, rebuild_jv, simsource

import Base: mergewith!

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
        nsrc::Integer
        geometry::Geometry
        data::Vector

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
    check_geom(geometry, data)
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
    check_geom(geometry, data)
    T == Float32 || (data = tof32.(data))
    nsrcd = length(data)
    nsrcg = get_nsrc(geometry)
    nsrcd == nsrcg || throw(judiMultiSourceException("Number of sources in geometry and data don't match $(nsrcd) != $(nsrcg)"))
    return judiVector{T, Array{T, N}}(nsrcd, geometry, data)
end

# contructor for in-core data container and given geometry
function judiVector(geometry::Geometry, data::SegyIO.SeisBlock)
    check_geom(geometry, data)
    # length of data vector
    src = get_header(data,"FieldRecord")
    nsrc = length(unique(src))
    # fill data vector with pointers to data location
    dataCell = Vector{Array{Float32, 2}}(undef, nsrc)
    for j=1:nsrc
        traces = findall(src .== unique(src)[j])
        dataCell[j] = convert(Array{Float32, 2}, data.data[1:get_nt(geometry, j), traces])
    end
    return judiVector{Float32, Array{Float32, 2}}(nsrc, geometry, dataCell)
end

# contructor for single out-of-core data container and given geometry
function judiVector(geometry::Geometry, data::SegyIO.SeisCon)
    check_geom(geometry, data)
    # length of data vector
    nsrc = length(data)
    # fill data vector with pointers to data location
    dataCell = Vector{SegyIO.SeisCon}(undef, nsrc)
    for j=1:nsrc
        dataCell[j] = split(data,j)
    end
    return judiVector{Float32, SegyIO.SeisCon}(nsrc, geometry,dataCell)
end

judiVector(data::SegyIO.SeisBlock; kw...) = judiVector(Geometry(data; key="receiver", kw...), data)
judiVector(data::SegyIO.SeisCon; kw...)= judiVector(Geometry(data; key="receiver", kw...), data)
judiVector(data::Vector{SegyIO.SeisCon}; kw...) = judiVector(Geometry(data; key="receiver", kw...), data)
judiVector(geometry::Geometry, data::Vector{SegyIO.SeisCon}) =  judiVector{Float32, SegyIO.SeisCon}(length(data), geometry, data)

############################################################
## overloaded multi_source functions
time_sampling(jv::judiVector) = get_dt(jv.geometry)

############################################################
# JOLI conversion
jo_convert(::Type{T}, jv::judiVector{T, Array{T, N}}, ::Bool) where {T<:AbstractFloat, N} = jv
jo_convert(::Type{T}, jv::judiVector{vT, Array{vT, N}}, B::Bool) where {T<:AbstractFloat, vT, N} = judiVector{T, Array{T, N}}(jv.nsrc, jv.geometry, jo_convert.(T, jv.data, B))
zero(::Type{T}, v::judiVector{vT, AT}; nsrc::Integer=v.nsrc) where {T, vT, AT} = judiVector{T, AT}(nsrc, deepcopy(v.geometry[1:nsrc]), T(0) .* v.data[1:nsrc])

function copy!(jv::judiVector, jv2::judiVector)
    jv.geometry = deepcopy(jv2.geometry)
    jv.data .= jv2.data
    jv
end

copyto!(jv::judiVector, jv2::judiVector) = copy!(jv, jv2)
make_input(jv::judiVector{T, Matrix{T}}) where T = jv.data[1]
make_input(jv::judiVector{T, Vector{T}}) where T = reshape(jv.data[1], :, 1)
make_input(jv::judiVector{T, SeisCon}) where T = get_data(jv).data[1]

check_compat(ms::Vararg{judiVector, N}) where N = all(y -> compareGeometry(y.geometry, first(ms).geometry), ms)

function as_ordered_dict(x::judiVector{T, AT}; v=1) where {T, AT}
    x.nsrc == 1 || throw(ArgumentError("as_ordered_dict only supported for single shot records"))
    jdict = OrderedDict(zip(as_coord_set(x.geometry[1]), collect.(eachcol(v.*make_input(x)))))
    return jdict
end

function from_ordered_dict(d::OrderedDict{NTuple{N, T}}, t::AbstractRange{T}) where {N, T}
    Gnew = GeometryIC{T}(coords_from_keys(d.keys)..., [t])
    return judiVector{T, Matrix{T}}(1, Gnew, [hcat(d.vals...)])
end

function pad_zeros(zpad::OrderedDict, m::T, d::judiVector{T, AT}) where {T, AT}
    @assert d.nsrc == 1
    dloc = as_ordered_dict(d; v=m)
    sort!(merge!(.+, dloc, zpad))
    return from_ordered_dict(dloc, d.geometry.taxis[1])
end

function simsource(M::Vector{T}, x::judiVector{T, AT}; reduction=+, minimal::Bool=false) where {T, AT}
    reduction in [+, nothing] || throw(ArgumentError("$(reduction): Only + and nothing supported for reduction (supershot or supershot before sum)"))
    # Without reduction, add zero trace for all missing coords
    if isnothing(reduction)
        sgeom = as_coord_set(x.geometry)
        zpad = OrderedDict(zip(sgeom, [zeros(Float32, get_nt(x.geometry, 1)) for s=1:length(sgeom)]))
        dOut = vcat([pad_zeros(zpad, M[s], x[s]) for s=1:x.nsrc]...)
    # With reduction. COuld reuse the previous but cheaper to compute the sum directly
    else
        reduction = minimal ? PopZero() : reduction
        sdict = mergewith!(reduction, map((d, m)->as_ordered_dict(d; v=m), x, M)...)
        sort!(sdict)
        dOut = from_ordered_dict(sdict, x.geometry.taxis[1])
    end
    return dOut
end

function simsource(M::Matrix{T}, x::judiVector{T, AT}; reduction=+, minimal::Bool=false) where {T, AT}
    isnothing(reduction) && @assert size(M, 1) == 1
    return vcat([simsource(vec(M[si,:]), x; reduction=reduction, minimal=minimal) for si=1:size(M, 1)]...)
end

## Merging utility
struct PopZero end

no_sim_msg = """
No common receiver poisiton found to construct a super shot. You can

- Input shot records with at least one receiver position in common
- Use `simsource(M, data; minimal=false)`or `M*data` to construct a supershot with zero-padded missing traces.
"""

function mergewith!(::PopZero, d1::OrderedDict{K, V}, d2::OrderedDict{K, V}) where {K, V}
    k1 = keys(d1)
    k2 = keys(d2)
    mergekeys = intersect(k1, k2)
    if isempty(mergekeys)
	@show k1, k2
        throw(ArgumentError(no_sim_msg))
    end
    for k in mergekeys
        d1[k] .+= d2[k]
    end
    Base.filter!(p->p.first in mergekeys, d1)
    return d1
end

##########################################################
# Overload needed base function for SegyIO objects

vec(x::SegyIO.SeisCon) = vec(x[1].data)
dot(x::SegyIO.SeisCon, y::SegyIO.SeisCon) = dot(x[1].data, y[1].data)
norm(x::SegyIO.SeisCon, p::Real=2) = norm(x[1].data, p)
abs(x::SegyIO.IBMFloat32) = abs(Float32(x))
*(::Number, ::SeisCon) = throw(judiMultiSourceException("Cannot multiply out of core SeisCon byt scalar"))

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

    d.geometry = Geometry(d.geometry)
    q.geometry = Geometry(q.geometry)

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

        set_header!(blocks[j], "dt", Int(get_dt(d.geometry, j)*1f3))
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


function write_shot_record(srcGeometry::GeometryIC, srcData, recGeometry::GeometryIC, recData, options)
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


####################################################################################################
# Load OOC
function get_data(x::judiVector{T, AT}; rel_origin=(0, 0, 0), project=nothing) where {T, AT}
    shots = Vector{Matrix{Float32}}(undef, x.nsrc)
    rec_geometry = Geometry(x.geometry; rel_origin=rel_origin, project=project)
    for j=1:x.nsrc
        shots[j] = data_matrix(x, j)
    end
    return judiVector{T, Array{Float32, 2}}(x.nsrc, rec_geometry, shots)
end

convert_to_array(x::judiVector) = vcat(vec.(x.data)...)

data_matrix(x::judiVector{T, SeisCon}, i::Integer) where T = convert(Matrix{Float32}, x.data[i][1].data)[1:get_nt(x.geometry, i), :]
data_matrix(x::judiVector{T, Matrix{T}}, i::Integer) where T = x.data[i]

##### Rebuild bad vector
"""
    reuild_jv(v)
rebuild a judiVector from previous version type or JLD2 reconstructed type
"""
rebuild_jv(v::judiVector{T, AT}) where {T, AT} = v
rebuild_jv(v) = judiVector(convgeom(v), convdata(v))


function rebuild_maybe_jld(x::Vector{Any})
    try
        return tof32(x)
    catch e
        if hasproperty(x[1], :offset)
            return [Float32.(StepRangeLen(xi.ref, xi.step, xi.len, xi.offset)) for xi in x]
        end
        return x
    end
end


# Rebuild for backward compatinility
convgeom(x) = GeometryIC{Float32}([getfield(x.geometry, s) for s=fieldnames(GeometryIC)]...)
convdata(x) = convert(Array{Array{Float32, 2}, 1}, x.data)
