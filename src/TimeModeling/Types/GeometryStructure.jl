# Source/receiver geometry structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

export Geometry, compareGeometry, GeometryIC, GeometryOOC, get_nsrc, n_samples

abstract type Geometry{T} end

const CoordT = Union{Vector{T}, Vector{Vector{T}}} where T<:Number
(::Type{CoordT})(x::Vector{Any}) = rebuild_maybe_jld(x)

# In-core geometry structure for seismic header information
mutable struct GeometryIC{T} <: Geometry{T}
    xloc::CoordT  # Array of receiver positions (fixed for all experiments)
    yloc::CoordT
    zloc::CoordT
    dt::Array{T,1}
    nt::Array{Integer,1}
    t::Array{T,1}
end

# Out-of-core geometry structure, contains look-up table instead of coordinates
mutable struct GeometryOOC{T} <: Geometry{T}
    container::Array{SegyIO.SeisCon,1}
    dt::Array{T,1}
    nt::Array{Integer,1}
    t::Array{T,1}
    nsamples::Array{Integer,1}
    key::String
    segy_depth_key::String
end

######################## shapes easy access ################################
get_nsrc(g::GeometryIC) = length(g.xloc)
get_nsrc(g::GeometryOOC) = length(g.container)

n_samples(g::GeometryOOC, ::Info) = sum(g.nsamples)
n_samples(g::GeometryIC, info::Info) = sum([length(g.xloc[j])*g.nt[j] for j=1:info.nsrc])
n_samples(g::GeometryOOC, ::Integer) = sum(g.nsamples)
n_samples(g::GeometryIC, nsrc::Integer) = sum([length(g.xloc[j])*g.nt[j] for j=1:nsrc])

################################################ Constructors ####################################################################

"""
    Geometry
        xloc::Array{Array{T, 1},1}
        yloc::Array{Array{T, 1},1}
        zloc::Array{Array{T, 1},1}
        dt::Array{T,1}
        nt::Array{Integer,1}
        t::Array{T,1}

Geometry structure for seismic sources or receivers. Each field is a cell array, where individual cell entries\\
contain values or arrays with coordinates and sampling information for the corresponding shot position. The \\
first three entries are in meters and the last three entries in milliseconds.


Constructors
============

Only pass `dt` and `n` and automatically set `t`:

    Geometry(xloc, yloc, zloc; dt=[], nt=[])

Pass single array as coordinates/parameters for all `nsrc` experiments:

    Geometry(xloc, yloc, zloc, dt=[], nt=[], nsrc=1)

Create geometry structure for either source or receivers from a SegyIO.SeisBlock object.\\
`segy_depth_key` is the SegyIO keyword that contains the depth coordinate and `key` is \\
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

Check the seis_block's header entries to findall out which keywords contain the depth coordinates.\\
The source depth keyword is either `SourceDepth` or `SourceSurfaceElevation`. The receiver depth \\
keyword is typically `RecGroupElevation`.

(4) Read source and receiver geometries from out-of-core SEG-Y files (for large data sets). Returns an out-of-core \\
geometry object `GeometryOOC` without the source/receiver coordinates, but a lookup table instead:

    using SegyIO
    seis_container = segy_scan("/path/to/data/directory","filenames",["GroupX","GroupY","RecGroupElevation","SourceDepth","dt"])
    rec_geometry = Geometry(seis_container; key="receiver", segy_depth_key="RecGroupElevation")
    src_geometry = Geometry(seis_container; key="source", segy_depth_key="SourceDepth")

"""
Geometry(xloc::CoordT, yloc::CoordT, zloc::CoordT, dt::Array{T,1}, nt::Array{Integer,1}, t::Array{T,1}) where T = GeometryIC{T}(xloc,yloc,zloc,dt,nt,t)

# Fallback constructors for non standard input types 
function Geometry(xloc, yloc, zloc; dt=[], t=[], nsrc=nothing)
    if any(typeof(x) <: AbstractRange for x=[xloc, yloc, zloc])
        args = [typeof(x) <: AbstractRange ? collect(x) : x for x=[xloc, yloc, zloc]]
        isnothing(nsrc) && (return Geometry(args...; dt=dt, t=t))
        return Geometry(args...; dt=dt, t=t, nsrc=nsrc)
    end
    isnothing(nsrc) && (return Geometry(tof32(xloc), tof32(yloc), tof32(zloc); dt=dt, t=t))
    return Geometry(tof32(xloc), tof32(yloc), tof32(zloc); dt=dt, t=t, nsrc=nsrc)
end

# Constructor if nt is not passed
function Geometry(xloc::Array{Array{T, 1},1}, yloc::CoordT, zloc::Array{Array{T, 1},1};dt=[],t=[]) where T
    nsrc = length(xloc)
    # Check if single dt was passed
    dtCell = typeof(t) <: Real ? [T(dt) for j=1:nsrc] : T.(dt)
    # Check if single t was passed
    tCell = typeof(t) <: Real ? [T(t) for j=1:nsrc] : T.(t)

    # Calculate number of time steps
    ntCell = typeof(t) <: Real ? [floor(Int, t / dt) + 1 for j=1:nsrc] : floor.(Int, tCell ./ dtCell) .+ 1
    return GeometryIC{T}(xloc, yloc, zloc, dtCell, ntCell, tCell)
end

# Constructor if coordinates are not passed as a cell arrays
function Geometry(xloc::Array{T, 1}, yloc::CoordT, zloc::Array{T, 1}; dt=[], t=[], nsrc::Integer=1) where T
    xlocCell = [xloc for j=1:nsrc]
    ylocCell = [yloc for j=1:nsrc]
    zlocCell = [zloc for j=1:nsrc]
    dtCell = [T(dt) for j=1:nsrc]
    ntCell = [floor(Int, t/dt)+1 for j=1:nsrc]
    tCell = [T(t) for j=1:nsrc]
    return GeometryIC{T}(xlocCell, ylocCell, zlocCell, dtCell, ntCell, tCell)
end

################################################ Constructors from SEGY data  ####################################################

# Set up source geometry object from in-core data container
function Geometry(data::SegyIO.SeisBlock; key="source", segy_depth_key="")
    src = get_header(data,"FieldRecord")
    nsrc = length(unique(src))
    if key=="source"
        isempty(segy_depth_key) && (segy_depth_key="SourceSurfaceElevation")
        params = ["SourceX","SourceY",segy_depth_key]
        gt = Float32
    elseif key=="receiver"
        isempty(segy_depth_key) && (segy_depth_key="RecGroupElevation")
        params = ["GroupX","GroupY",segy_depth_key]
        gt = Array{Float32, 1}
    else
        throw("Specified keyword not supported")
    end
    xloc = Array{gt, 1}(undef, nsrc)
    yloc = Array{gt, 1}(undef, nsrc)
    zloc = Array{gt, 1}(undef, nsrc)
    dt = Array{Float32}(undef, nsrc)
    nt = Array{Integer}(undef, nsrc)
    t = Array{Float32}(undef, nsrc)

    xloc_full = get_header(data, params[1])
    yloc_full = get_header(data, params[2])
    zloc_full = get_header(data, params[3])
    dt_full = get_header(data, "dt")[1]
    nt_full = get_header(data, "ns")[1]

    for j=1:nsrc
        traces = findall(src .== unique(src)[j])
        if key=="source"    # assume same source location for all traces within one shot record
            xloc[j] = convert(gt, xloc_full[traces][1])
            yloc[j] = convert(gt,yloc_full[traces][1])
            zloc[j] = abs.(convert(gt,zloc_full[traces][1]))
        else
            xloc[j] = convert(gt, xloc_full[traces])
            yloc[j] = convert(gt, yloc_full[traces])
            zloc[j] = abs.(convert(gt, zloc_full[traces]))
        end
        dt[j] = Float32(dt_full/1f3)
        nt[j] = convert(Integer,nt_full)
        t[j] =  Float32((nt[j]-1)*dt[j])
    end

    if key == "source"
        xloc = convertToCell(xloc)
        yloc = convertToCell(yloc)
        zloc = convertToCell(zloc)
    end
    return GeometryIC{Float32}(xloc,yloc,zloc,dt,nt,t)
end

# Set up geometry summary from out-of-core data container
function Geometry(data::SegyIO.SeisCon; key="source", segy_depth_key="")

    if key=="source"
        isempty(segy_depth_key) && (segy_depth_key="SourceSurfaceElevation")
    elseif key=="receiver"
        isempty(segy_depth_key) && (segy_depth_key="RecGroupElevation")
    else
        throw("Specified keyword not supported")
    end

    # read either source or receiver geometry
    nsrc = length(data)
    container = Array{SegyIO.SeisCon}(undef, nsrc)
    dt = Array{Float32}(undef, nsrc)
    nt = Array{Integer}(undef, nsrc)
    t = Array{Float32}(undef, nsrc)
    nsamples = Array{Integer}(undef, nsrc)
    for j=1:nsrc
        container[j] = split(data,j)
        dt[j] = data.blocks[j].summary["dt"][1]/1f3
        nt[j] = data.ns
        t[j] = (nt[j]-1)*dt[j]
        key=="source" ? nsamples[j] = data.ns : nsamples[j] = Int((data.blocks[j].endbyte - data.blocks[j].startbyte)/(240 + data.ns*4)*data.ns)
    end
    return  GeometryOOC{Float32}(container,dt,nt,t,nsamples,key,segy_depth_key)
end

# Set up geometry summary from out-of-core data container passed as cell array
function Geometry(data::Array{SegyIO.SeisCon,1}; key="source", segy_depth_key="")

    if key=="source"
        isempty(segy_depth_key) && (segy_depth_key="SourceSurfaceElevation")
    elseif key=="receiver"
        isempty(segy_depth_key) && (segy_depth_key="RecGroupElevation")
    else
        throw("Specified keyword not supported")
    end

    nsrc = length(data)
    container = Array{SegyIO.SeisCon}(undef, nsrc)
    dt = Array{Float32}(undef, nsrc); nt = Array{Integer}(undef, nsrc); t = Array{Float32}(undef, nsrc)
    nsamples = Array{Integer}(undef, nsrc)
    for j=1:nsrc
        container[j] = data[j]
        dt[j] = data[j].blocks[1].summary["dt"][1]/1f3
        nt[j] = data[j].ns
        t[j] = (nt[j]-1)*dt[j]
        key=="source" ? nsamples[j] = data[j].ns : nsamples[j] = Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data[j].ns*4)*data[j].ns)
    end
    return  GeometryOOC{Float32}(container,dt,nt,t,nsamples,key,segy_depth_key)
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
        throw("Specified keyword not supported")
    end
    xloc = Array{gt, 1}(undef, nsrc)
    yloc = Array{gt, 1}(undef, nsrc)
    zloc = Array{gt, 1}(undef, nsrc)
    dt = Array{Float32}(undef, nsrc); nt = Array{Integer}(undef, nsrc); t = Array{Float32}(undef, nsrc)

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
        dt[j] = get_header(header, params[4])[1]/1f3
        nt[j] = convert(Integer, get_header(header, params[5])[1])
        t[j] =  (nt[j]-1)*dt[j]
    end
    if geometry.key == "source"
        xloc = convertToCell(xloc)
        yloc = convertToCell(yloc)
        zloc = convertToCell(zloc)
    end
    return GeometryIC(xloc,yloc,zloc,dt,nt,t)
end

Geometry(geometry::GeometryIC) = geometry
Geometry(::Nothing) = nothing

###########################################################################################################################################

# Subsample in-core geometry structure
function subsample(geometry::GeometryIC,srcnum)
    if length(srcnum)==1
        srcnum = srcnum[1]
        geometry = Geometry(geometry.xloc[srcnum], geometry.yloc[srcnum], geometry.zloc[srcnum];
                            dt=geometry.dt[srcnum],t=geometry.t[srcnum],nsrc=1)
    else
        geometry = Geometry(geometry.xloc[srcnum], geometry.yloc[srcnum], geometry.zloc[srcnum],
                            geometry.dt[srcnum], geometry.nt[srcnum], geometry.t[srcnum])
    end
    return geometry
end

# Subsample out-of-core geometry structure
subsample(geometry::GeometryOOC, srcnum) = Geometry(geometry.container[srcnum]; key=geometry.key, segy_depth_key=geometry.segy_depth_key)

# Compare if geometries match
function compareGeometry(geometry_A::Geometry, geometry_B::Geometry)
    if isequal(geometry_A.xloc, geometry_B.xloc) && isequal(geometry_A.yloc, geometry_B.yloc) && isequal(geometry_A.zloc, geometry_B.zloc) &&
    isequal(geometry_A.dt, geometry_B.dt) && isequal(geometry_A.nt, geometry_B.nt)
        return true
    else
        return false
    end
end

isequal(geometry_A::Geometry, geometry_B::Geometry) = compareGeometry(geometry_A, geometry_B)

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

isequal(geometry_A::GeometryOOC, geometry_B::GeometryOOC) = compareGeometry(geometry_A, geometry_B)

compareGeometry(geometry_A::GeometryOOC, geometry_B::Geometry) = true
compareGeometry(geometry_A::Geometry, geometry_B::GeometryOOC) = true

for G in [GeometryOOC, GeometryIC]
    @eval function push!(G1::$G, G2::$G)
        for k in fieldnames($G)
            pushfield!(getfield(G1, k), getfield(G2, k))
        end
    end
end

pushfield!(a::Array, b::Array) = append!(a, b)
pushfield!(a, b) = nothing

########## merge judiVector ###########

"""
    merge(geometry)

Merge the geometry of multi-source judiVector \\
The trace with same locations will be added, \\
and with different locations will be appended. \\
This merge will be largely used to generate simultaneous judiVectors with random weights.

"""

# merge(GeometryIC)
function merge(geometry::GeometryIC{T}) where T

    (norm(diff(v.geometry.dt))+norm(diff(v.geometry.nt))+norm(diff(v.geometry.t)) == 0) || throw(judiVectorException("nt/dt/t mismatch in judiVector"))

    loc = Vector{Tuple{T,T,T}}()
    for i = get_nsrc(geometry)
        for j = 1:length(geometry.xloc[i])
            xloc = geometry.xloc[i][j]
            if length(geometry.yloc[i]) == 1
                yloc = geometry.yloc[i][1]
            else
                yloc = geometry.yloc[i][j]
            end
            zloc = geometry.zloc[i][j]
            push!(loc, (xloc, yloc, zloc))
        end
    end

    loc_merge = sort(unique(loc))

    # set geometry
    xloc = @. getindex(loc_merge, 1)
    yloc = @. getindex(loc_merge, 2)
    if length(geometry.yloc[1]) == 1
        yloc = geometry.yloc[1]
    end
    zloc = @. getindex(key, 3)

    return Geometry(xloc,yloc,zloc; dt=v.geometry.dt[1], t=v.geometry.t[1])

end