# Source/receiver geometry structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

export Geometry, compareGeometry, GeometryIC, GeometryOOC, get_nsrc, n_samples, super_shot_geometry

abstract type Geometry{T} end

mutable struct GeometryException <: Exception
    msg :: String
end


const CoordT{T} = Union{Vector{T}, Vector{Vector{T}}} where T<:Number
(::Type{CoordT{T}})(x::Vector{Any}) where {T<:Real} = rebuild_maybe_jld(x)
Base.convert(::Type{CoordT}, a::Vector{Any}) = Vector{typeof(a[1])}(a)

# In-core geometry structure for seismic header information
mutable struct GeometryIC{T} <: Geometry{T}
    xloc::CoordT{T}  # Array of receiver positions (fixed for all experiments)
    yloc::CoordT{T}
    zloc::CoordT{T}
    dt::Vector{T}
    nt::Vector{Integer}
    t::Vector{T}
end

getproperty(G::GeometryIC, s::Symbol) = s == :nrec ? length.(G.xloc) : getfield(G, s)

# Out-of-core geometry structure, contains look-up table instead of coordinates
mutable struct GeometryOOC{T} <: Geometry{T}
    container::Vector{SegyIO.SeisCon}
    dt::Vector{T}
    nt::Vector{<:Integer}
    t::Vector{T}
    nrec::Vector{<:Integer}
    key::String
    segy_depth_key::String
end


display(G::Geometry) = println("$(typeof(G)) wiht $(length(G.nt)) sources")
show(io::IO, G::Geometry) = print(io, "$(typeof(G)) wiht $(length(G.nt)) sources")
show(io::IO, ::MIME{Symbol("text/plain")}, G::Geometry) = println(io, "$(typeof(G)) wiht $(length(G.nt)) sources")

######################## shapes easy access ################################
get_nsrc(g::GeometryIC) = length(g.xloc)
get_nsrc(g::GeometryOOC) = length(g.container)
get_nsrc(S::SeisCon) = length(S)
get_nsrc(S::Vector{SeisCon}) = length(S)
get_nsrc(S::SeisBlock) = length(unique(get_header(S, "FieldRecord")))

n_samples(g::GeometryOOC, nsrc::Integer) = sum([g.nrec[j]*g.nt[j] for j=1:nsrc])
n_samples(g::GeometryIC, nsrc::Integer) = sum([length(g.xloc[j])*g.nt[j] for j=1:nsrc])
n_samples(g::Geometry) = n_samples(g, get_nsrc(g))

rec_space(G::Geometry) = AbstractSize((:src, :time, :rec), (get_nsrc(G), G.nt, G.nrec))
################################################ Constructors ####################################################################

"""
    GeometryIC
        xloc::Array{Array{T, 1},1}
        yloc::Array{Array{T, 1},1}
        zloc::Array{Array{T, 1},1}
        dt::Array{T,1}
        nt::Array{Integer,1}
        t::Array{T,1}

Geometry structure for seismic sources or receivers. Each field is a cell array, where individual cell entries
contain values or arrays with coordinates and sampling information for the corresponding shot position. The 
first three entries are in meters and the last three entries in milliseconds.

GeometryOOC{T} <: Geometry{T}
    container::Array{SegyIO.SeisCon,1}
    dt::Array{T,1}
    nt::Array{Integer,1}
    t::Array{T,1}
    nrec::Array{Integer,1}
    key::String
    segy_depth_key::String

Constructors
============

Only pass `dt` and `n` and automatically set `t`:

    Geometry(xloc, yloc, zloc; dt=[], nt=[])

Pass single array as coordinates/parameters for all `nsrc` experiments:

    Geometry(xloc, yloc, zloc, dt=[], nt=[], nsrc=1)

Create geometry structure for either source or receivers from a SegyIO.SeisBlock object.
`segy_depth_key` is the SegyIO keyword that contains the depth coordinate and `key` is 
set to either `source` for source geometry or `receiver` for receiver geometry:

    Geometry(SeisBlock; key="source", segy_depth_key="")

Create geometry structure for from a SegyIO.SeisCon object (seismic data container):

    Geometry(SeisCon; key="source", segy_depth_key="")

Examples
========

(1) Set up receiver geometry for 2D experiment with 4 source locations and 80 fixed receivers:

    xrec = range(100,stop=900,length=80)
    yrec = range(0,stop=0,length=80)
    zrec = range(50,stop=50,length=80)
    dt = 4f0
    t = 1000f0

    rec_geometry = Geometry(xrec, yrec, zrec; dt=dt, t=t, nsrc=4)

(2) Set up corresponding source geometry (coordinates can be of type `linspace` or regular arrays):

    xsrc = [200,400,600,800]
    ysrc = [0,0,0,0]
    zsrc = [50,50,50,50]

    src_geometry = Geometry(xsrc, ysrc, zsrc; dt=dt, t=t, nsrc=4)

(3) Read source and receiver geometries from SEG-Y file:

    using SegyIO
    seis_block = segy_read("test_file.segy")
    rec_geometry = Geometry(seis_block; key="receiver", segy_depth_key="RecGroupElevation")
    src_geometry = Geometry(seis_block; key="source", segy_depth_key="SourceDepth")

Check the seis_block's header entries to findall out which keywords contain the depth coordinates.
The source depth keyword is either `SourceDepth` or `SourceSurfaceElevation`. The receiver depth 
keyword is typically `RecGroupElevation`.

(4) Read source and receiver geometries from out-of-core SEG-Y files (for large data sets). Returns an out-of-core 
geometry object `GeometryOOC` without the source/receiver coordinates, but a lookup table instead:

    using SegyIO
    seis_container = segy_scan("/path/to/data/directory","filenames",["GroupX","GroupY","RecGroupElevation","SourceDepth","dt"])
    rec_geometry = Geometry(seis_container; key="receiver", segy_depth_key="RecGroupElevation")
    src_geometry = Geometry(seis_container; key="source", segy_depth_key="SourceDepth")

"""
function Geometry(xloc, yloc, zloc; dt=nothing, t=nothing, nsrc=nothing)
    check_time(dt, t)
    if any(typeof(x) <: AbstractRange for x=[xloc, yloc, zloc])
        args = [typeof(x) <: AbstractRange ? collect(x) : x for x=[xloc, yloc, zloc]]
        isnothing(nsrc) && (return Geometry(args...; dt=dt, t=t))
        return Geometry(args...; dt=dt, t=t, nsrc=nsrc)
    end
    isnothing(nsrc) && (return Geometry(tof32(xloc), tof32(yloc), tof32(zloc); dt=dt, t=t))
    return Geometry(tof32(xloc), tof32(yloc), tof32(zloc); dt=dt, t=t, nsrc=nsrc)
end

Geometry(xloc::CoordT, yloc::CoordT, zloc::CoordT, dt::Array{T,1}, nt::Array{Integer,1}, t::Array{T,1}) where {T<:Real} = GeometryIC{T}(xloc,yloc,zloc,dt,nt,t)

# Fallback constructors for non standard input types 

# Constructor if nt is not passed
function Geometry(xloc::Array{Array{T, 1},1}, yloc::CoordT, zloc::Array{Array{T, 1},1}; dt=nothing, t=nothing) where {T<:Real}
    check_time(dt, t)
    nsrc = length(xloc)
    # Check if single dt was passed
    dtCell = isa(dt, Real) ? [T(dt) for j=1:nsrc] : T.(dt)
    # Check if single t was passed
    tCell = isa(t, Real) ? [T(t) for j=1:nsrc] : T.(t)

    # Calculate number of time steps
    ntCell = floor.(Int, tCell ./ dtCell) .+ 1
    return GeometryIC{T}(xloc, yloc, zloc, dtCell, ntCell, tCell)
end

# Constructor if coordinates are not passed as a cell arrays
function Geometry(xloc::Array{T, 1}, yloc::CoordT, zloc::Array{T, 1}; dt=nothing, t=nothing, nsrc::Integer=1) where {T<:Real}
    check_time(dt, t)
    xlocCell = [xloc for j=1:nsrc]
    ylocCell = [yloc for j=1:nsrc]
    zlocCell = [zloc for j=1:nsrc]
    dtCell = [T(dt) for j=1:nsrc]
    tCell = [T(t) for j=1:nsrc]
    ntCell = floor.(Int, tCell ./ dtCell) .+ 1
    return GeometryIC{T}(xlocCell, ylocCell, zlocCell, dtCell, ntCell, tCell)
end

################################################ Constructors from SEGY data  ####################################################

# Utility function to prepare dtCell, ntCell, tCell from SEGY or based on user defined dt and t.
# Useful when creating geometry for Forward Modeling with custom timings.

_get_p(v, S, nsrc, P) = throw(GeometryException("User defined `dt` is neither: Real, Array of Real or the length of Array doesn't match the number of sources in SEGY"))
_get_p(::Nothing, S::SeisBlock, nsrc::Integer, p, ::Type{T}, s::T) where T = fill(T(get_header(S, p)[1]/s), nsrc)
_get_p(::Nothing, S::SeisCon, nsrc::Integer, p, ::Type{T}, s::T) where T = [T(_get_p_SeisCon(S, p, j)/s) for j=1:nsrc]
_get_p(::Nothing, S::Vector{SeisCon}, nsrc::Integer, p, ::Type{T}, s::T) where T  = [T(_get_p_SeisCon(S[j], p, 1)/s) for j=1:nsrc]
_get_p(v::Real, data, nsrc::Integer, p, ::Type{T}, s::T) where T  = fill(T(v), nsrc)
_get_p(v::Vector, data, nsrc::Integer, p, ::Type{T}, s::T) where T  = convert(Vector{T}, v)


_get_p_SeisCon(S::SeisCon, p::String, b::Integer) = try S.blocks[b].summary[p][1] catch; getfield(S, Symbol(p)); end

function timings_from_segy(data, dt=nothing, t::T=nothing) where T
    # Get nsrc
    nsrc = get_nsrc(data)
    dtCell = _get_p(dt, data, nsrc, "dt", Float32, 1f3)

    try
        tCell = T <: Number ? fill(Float32(t), nsrc) : convert(Vector{Float32}, t)
        ntCell = floor.(Integer, tCell ./ dtCell) .+ 1
        return dtCell, ntCell, tCell
    catch e
        ntCell = _get_p(nothing, data, nsrc, "ns", Int, 1)
        tCell = Float32.((ntCell .- 1) .* dtCell)
        return dtCell, ntCell, tCell
    end
end

# Set up source geometry object from in-core data container
function Geometry(data::SegyIO.SeisBlock; key="source", segy_depth_key="", dt=nothing, t=nothing)
    check_time(dt, t, true)
    src = get_header(data,"FieldRecord")
    usrc = unique(src)
    nsrc = length(usrc)
    if key=="source"
        isempty(segy_depth_key) && (segy_depth_key="SourceSurfaceElevation")
        params = ["SourceX","SourceY",segy_depth_key]
        gt = Float32
    elseif key=="receiver"
        isempty(segy_depth_key) && (segy_depth_key="RecGroupElevation")
        params = ["GroupX","GroupY",segy_depth_key]
        gt = Array{Float32, 1}
    else
        throw(GeometryException("Specified keyword not supported"))
    end
    xloc = Vector{gt}(undef, nsrc)
    yloc = Vector{gt}(undef, nsrc)
    zloc = Vector{gt}(undef, nsrc)

    xloc_full = get_header(data, params[1])
    yloc_full = get_header(data, params[2])
    zloc_full = get_header(data, params[3])

    for j=1:nsrc
        traces = findall(src .== usrc[j])
        if key=="source"    # assume same source location for all traces within one shot record
            xloc[j] = convert(gt, xloc_full[traces][1])
            yloc[j] = convert(gt, yloc_full[traces][1])
            zloc[j] = abs.(convert(gt, zloc_full[traces][1]))
        else
            xloc[j] = convert(gt, xloc_full[traces])
            yloc[j] = convert(gt, yloc_full[traces])
            zloc[j] = abs.(convert(gt, zloc_full[traces]))
        end
    end

    if key == "source"
        xloc = convertToCell(xloc)
        yloc = convertToCell(yloc)
        zloc = convertToCell(zloc)
    end

    dtCell, ntCell, tCell = timings_from_segy(data, dt, t)
    return GeometryIC{Float32}(xloc,yloc,zloc,dtCell,ntCell,tCell)
end

# Set up geometry summary from out-of-core data container
function Geometry(data::SegyIO.SeisCon; key="source", segy_depth_key="", dt=nothing, t=nothing)
    check_time(dt, t, true)

    if key=="source"
        isempty(segy_depth_key) && (segy_depth_key="SourceSurfaceElevation")
    elseif key=="receiver"
        isempty(segy_depth_key) && (segy_depth_key="RecGroupElevation")
    else
        throw(GeometryException("Specified keyword not supported"))
    end

    # read either source or receiver geometry
    nsrc = length(data)
    container = Array{SegyIO.SeisCon}(undef, nsrc)
    nrec = Array{Integer}(undef, nsrc)
    for j=1:nsrc
        container[j] = split(data,j)
        nrec[j] = key=="source" ? 1 : Int((data.blocks[j].endbyte - data.blocks[j].startbyte)/(240 + data.ns*4))
    end

    dtCell,ntCell,tCell = timings_from_segy(data, dt, t)
    return GeometryOOC{Float32}(container,dtCell,ntCell,tCell,nrec,key,segy_depth_key)
end

# Set up geometry summary from out-of-core data container passed as cell array
function Geometry(data::Array{SegyIO.SeisCon,1}; key="source", segy_depth_key="", dt=nothing, t=nothing)
    check_time(dt, t, true)

    if key=="source"
        isempty(segy_depth_key) && (segy_depth_key="SourceSurfaceElevation")
    elseif key=="receiver"
        isempty(segy_depth_key) && (segy_depth_key="RecGroupElevation")
    else
        throw(GeometryException("Specified keyword not supported"))
    end

    nsrc = length(data)
    container = Array{SegyIO.SeisCon}(undef, nsrc)
    nrec = Array{Integer}(undef, nsrc)
    for j=1:nsrc
        container[j] = data[j]
        nrec[j] = key=="source" ? 1 : Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data[j].ns*4))
    end

    dtCell,ntCell,tCell = timings_from_segy(data, dt, t)
    return GeometryOOC{Float32}(container,dtCell,ntCell,tCell,nrec,key,segy_depth_key)
end

# Load geometry from out-of-core Geometry container
function Geometry(geometry::GeometryOOC)
    nsrc = length(geometry.container)

    # read either source or receiver geometry
    if geometry.key=="source"
        params = ["SourceX","SourceY",geometry.segy_depth_key,"dt","ns"]
        gt = Float32
    elseif geometry.key=="receiver"
        params = ["GroupX","GroupY",geometry.segy_depth_key,"dt","ns"]
        gt = Array{Float32, 1}
    else
        throw(GeometryException("Specified keyword not supported"))
    end
    xloc = Array{gt, 1}(undef, nsrc)
    yloc = Array{gt, 1}(undef, nsrc)
    zloc = Array{gt, 1}(undef, nsrc)
    dt = Array{Float32}(undef, nsrc)
    nt = Array{Integer}(undef, nsrc)
    t = Array{Float32}(undef, nsrc)

    for j=1:nsrc

        header = read_con_headers(geometry.container[j], params, 1)
        if geometry.key=="source"
            xloc[j] = convert(gt, get_header(header, params[1])[1])
            yloc[j] = convert(gt, get_header(header, params[2])[1])
            zloc[j] = abs.(convert(gt,get_header(header, params[3])[1]))
        else
            xloc[j] = convert(gt, get_header(header, params[1]))
            yloc[j] = convert(gt, get_header(header, params[2]))
            zloc[j] = abs.(convert(gt, get_header(header, params[3])))
        end
        dt[j] = geometry.dt[j]
        nt[j] = geometry.nt[j]
        t[j] = geometry.t[j]
    end
    if geometry.key == "source"
        xloc = convertToCell(xloc)
        yloc = convertToCell(yloc)
        zloc = convertToCell(zloc)
    end
    return GeometryIC(xloc,yloc,zloc,dt,nt,t)
end

Geometry(geometry::GeometryIC) = geometry
Geometry(v::Array{T}) where T = v
Geometry(::Nothing) = nothing

###########################################################################################################################################
subsample(g::Geometry, I) = getindex(g, I)

# getindex in-core geometry structure
function getindex(geometry::GeometryIC{T}, srcnum::RangeOrVec) where T
    xsub = geometry.xloc[srcnum]
    ysub = geometry.yloc[srcnum]
    zsub = geometry.zloc[srcnum]
    dtsub = geometry.dt[srcnum]
    ntsub = geometry.nt[srcnum]
    tsub = geometry.t[srcnum]
    geometry = GeometryIC{T}(xsub, ysub, zsub, dtsub, ntsub, tsub)
    return geometry
end

getindex(geometry::GeometryIC{T}, srcnum::Integer) where T = getindex(geometry, srcnum:srcnum)

# getindex out-of-core geometry structure
getindex(geometry::GeometryOOC, srcnum) = Geometry(geometry.container[srcnum]; key=geometry.key, segy_depth_key=geometry.segy_depth_key, dt=geometry.dt[srcnum], t=geometry.t[srcnum])

###########################################################################################################################################
# Compare if geometries match
function compareGeometry(geometry_A::Geometry, geometry_B::Geometry)
    if isequal(geometry_A.xloc, geometry_B.xloc) && isequal(geometry_A.yloc, geometry_B.yloc) && isequal(geometry_A.zloc, geometry_B.zloc) &&
    isequal(geometry_A.dt, geometry_B.dt) && isequal(geometry_A.nt, geometry_B.nt)
        return true
    else
        return false
    end
end

==(geometry_A::Geometry, geometry_B::Geometry) = compareGeometry(geometry_A, geometry_B)
isapprox(geometry_A::Geometry, geometry_B::Geometry; kw...) = compareGeometry(geometry_A, geometry_B)

function compareGeometry(geometry_A::GeometryOOC, geometry_B::GeometryOOC)
    check = true
    for j=1:length(geometry_A.container)
        if ~isequal(geometry_A.container[j].blocks[1].summary["GroupX"], geometry_B.container[j].blocks[1].summary["GroupX"]) ||
        ~isequal(geometry_A.container[j].blocks[1].summary["GroupY"], geometry_B.container[j].blocks[1].summary["GroupY"]) ||
        ~isequal(geometry_A.container[j].blocks[1].summary["SourceX"], geometry_B.container[j].blocks[1].summary["SourceX"]) ||
        ~isequal(geometry_A.container[j].blocks[1].summary["SourceY"], geometry_B.container[j].blocks[1].summary["SourceY"]) ||
        ~isequal(geometry_A.container[j].blocks[1].summary["dt"], geometry_B.container[j].blocks[1].summary["dt"])
            check = false
        end
    end
    return check
end

==(geometry_A::GeometryOOC, geometry_B::GeometryOOC) = compareGeometry(geometry_A, geometry_B)

compareGeometry(geometry_A::GeometryOOC, geometry_B::Geometry) = true
compareGeometry(geometry_A::Geometry, geometry_B::GeometryOOC) = true

###########################################################################################################################################

for G in [GeometryOOC, GeometryIC]
    @eval function push!(G1::$G, G2::$G)
        for k in fieldnames($G)
            pushfield!(getfield(G1, k), getfield(G2, k))
        end
    end
end

pushfield!(a::Array, b::Array) = append!(a, b)
pushfield!(a, b) = nothing

# Gets called by judiVector constructor to be sure that geometry is consistent with the data.
# Data may be any of: Array, Array of Array, SeisBlock, SeisCon
check_geom(geom::Geometry, data::Array{T}) where T = all([check_geom(geom[s], data[s]) for s=1:get_nsrc(geom)])
check_geom(geom::Geometry, data::Array{T}) where {T<:Number} = _check_geom(geom.nt[1],  size(data, 1)) && _check_geom(geom.nrec[1],  size(data, 2))
check_geom(geom::Geometry, data::SeisBlock) = _check_geom(geom.nt[1], data.fileheader.bfh.ns)
check_geom(geom::Geometry, data::SeisCon) = _check_geom(geom.nt[1], data.ns)
_check_geom(nt::Integer, ns::Integer) = nt == ns || _geom_missmatch(nt, ns)

check_time(dt::Number, t::Number, segy::Bool=false) = (t/dt == div(t, dt, RoundNearest)) || throw(GeometryException("Recording time t=$(t) not divisible by sampling rate dt=$(dt)"))
check_time(::Nothing, ::Nothing, segy::Bool=false) = segy || throw(GeometryException("Recording time `t` and sampling rate `dt` must be provided"))
check_time(dt::AbstractVector, t::AbstractVector, segy::Bool=false) = check_time.(dt, t)
_geom_missmatch(nt::Integer, ns::Integer) = throw(judiMultiSourceException("Geometry's number of samples doesn't match the data: $(nt), $(ns)"))


################################# Merge geometries ##############################################################
allsame(x, val=first(x)) = all(y->y==val, x)

as_coord_set(x::Vector{T}, y::T, z::Vector{T}) where T = OrderedSet(zip(x, z))
as_coord_set(x::Vector{T}, y::Vector{T}, z::Vector{T}) where T = OrderedSet(zip(x, y, z))

yloc(y::Vector{T}, ::Val{1}) where T<:Number = y[1]
yloc(y::T, ::Val{1}) where T<:Number = y
yloc(y, ::Val) = y

_get_coords(G::Geometry, ny::Val) = begin gloc = Geometry(G); return tuple(gloc.xloc[1], yloc(gloc.yloc[1], ny), gloc.zloc[1]) end

function as_coord_set(G::Geometry)
    @assert allsame(G.nt)
    @assert allsame(G.dt)
    @assert allsame(G.t)
    G0 = Geometry(G[1])
    ny = Val(length(G0.yloc[1]))
    s = as_coord_set(_get_coords(G0, ny::Val)...)
    nsrc = get_nsrc(G)
    if nsrc > 1
        map(i->union!(s, as_coord_set(_get_coords(G[i], ny::Val)...)), 2:nsrc)
    end
    sort!(s)
    return s
end

coords_from_set(S::OrderedSet{Tuple{T, T}}) where T = tuple([first.(S)], [[0f0]], [last.(S)])
coords_from_set(S::OrderedSet{Tuple{T, T, T}}) where T = tuple([first.(S)], [getindex.(S, 2)], [last.(S)])
coords_from_keys(S::Vector{Tuple{T, T}}) where T = tuple([first.(S)], [[0f0]], [last.(S)])
coords_from_keys(S::Vector{Tuple{T, T, T}}) where T = tuple([first.(S)], [getindex.(S, 2)], [last.(S)])

function super_shot_geometry(G::Geometry{T}) where T
    as_set = coords_from_set(as_coord_set(G))
    return GeometryIC{T}(as_set..., [G.dt[1]], [G.nt[1]], [G.t[1]])
end
