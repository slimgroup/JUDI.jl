# Source/receiver geometry structure
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

export Geometry, compareGeometry, GeometryIC, GeometryOOC

abstract type Geometry end

# In-core geometry structure for seismic header information
mutable struct GeometryIC <: Geometry
    xloc::Array{Any,1}  # Array of receiver positions (fixed for all experiments)
    yloc::Array{Any,1}
    zloc::Array{Any,1}
    dt::Array{Any,1}
    nt::Array{Any,1}
    t::Array{Any,1}
end

# Out-of-core geometry structure, contains look-up table instead of coordinates
mutable struct GeometryOOC <: Geometry
    container::Array{SegyIO.SeisCon,1}
    dt::Array{Any,1}
    nt::Array{Any,1}
    t::Array{Any,1}
    nsamples::Array{Any,1}
    key::String
    segy_depth_key::String
end

################################################ Constructors ####################################################################

"""
    Geometry
        xloc::Array{Any,1}
        yloc::Array{Any,1}
        zloc::Array{Any,1}
        dt::Array{Any,1}
        nt::Array{Any,1}
        t::Array{Any,1}

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
Geometry(xloc::Array{Any,1},yloc::Array{Any,1},zloc::Array{Any,1},dt::Array{Any,1},nt::Array{Any,1},t::Array{Any,1}) = GeometryIC(xloc,yloc,zloc,dt,nt,t)

# Constructor if nt is not passed
function Geometry(xloc::Array{Any,1},yloc::Array{Any,1},zloc::Array{Any,1};dt=[],t=[])
    nsrc = length(xloc)
    # Check if single dt was passed
    if typeof(dt) <: Real
        dtCell = Array{Any}(undef, nsrc)
        for j=1:nsrc
            dtCell[j] = dt
        end
    else
        dtCell = dt
    end
    # Check if single t was passed
    if typeof(t) <: Real
        tCell = Array{Any}(undef, nsrc)
        for j=1:nsrc
            tCell[j] = t
        end
    else
        tCell = t
    end
    # Calculate number of time steps
    ntCell = Array{Any}(undef, nsrc)
    for j=1:nsrc
        ntCell[j] = Int(trunc(tCell[j]/dtCell[j] + 1))
    end
    return GeometryIC(xloc,yloc,zloc,dtCell,ntCell,tCell)
end

# Constructor if coordinates are not passed as a cell arrays
function Geometry(xloc,yloc,zloc;dt=[],t=[],nsrc::Integer=1)
    xlocCell = Array{Any}(undef, nsrc)
    ylocCell = Array{Any}(undef, nsrc)
    zlocCell = Array{Any}(undef, nsrc)
    dtCell = Array{Any}(undef, nsrc)
    ntCell = Array{Any}(undef, nsrc)
    tCell = Array{Any}(undef, nsrc)
    for j=1:nsrc
        xlocCell[j] = xloc
        ylocCell[j] = yloc
        zlocCell[j] = zloc
        dtCell[j] = dt
        ntCell[j] = Int(trunc(t/dt + 1))
        tCell[j] = t
    end
    return GeometryIC(xlocCell,ylocCell,zlocCell,dtCell,ntCell,tCell)
end


################################################ Constructors from SEGY data  ####################################################

# Set up source geometry object from in-core data container
function Geometry(data::SegyIO.SeisBlock; key="source", segy_depth_key="")
    src = get_header(data,"FieldRecord")
    nsrc = length(unique(src))
    if key=="source"
        isempty(segy_depth_key) && (segy_depth_key="SourceSurfaceElevation")
        params = ["SourceX","SourceY",segy_depth_key]
    elseif key=="receiver"
        isempty(segy_depth_key) && (segy_depth_key="RecGroupElevation")
        params = ["GroupX","GroupY",segy_depth_key]
    else
        throw("Specified keyword not supported")
    end
    xloc = Array{Any}(undef, nsrc); yloc = Array{Any}(undef, nsrc); zloc = Array{Any}(undef, nsrc)
    dt = Array{Any}(undef, nsrc); nt = Array{Any}(undef, nsrc); t = Array{Any}(undef, nsrc)

    xloc_full = get_header(data, params[1])
    yloc_full = get_header(data, params[2])
    zloc_full = get_header(data, params[3])
    dt_full = get_header(data, "dt")[1]
    nt_full = get_header(data, "ns")[1]

    for j=1:nsrc
        traces = findall(src .== unique(src)[j])
        if key=="source"    # assume same source location for all traces within one shot record
            xloc[j] = convert(Float32,xloc_full[traces][1])
            yloc[j] = convert(Float32,yloc_full[traces][1])
            zloc[j] = abs.(convert(Float32,zloc_full[traces][1]))
        else
            xloc[j] = convert(Array{Float32,1}, xloc_full[traces])
            yloc[j] = convert(Array{Float32,1}, yloc_full[traces])
            zloc[j] = abs.(convert(Array{Float32,1}, zloc_full[traces]))
        end
        dt[j] = dt_full/1f3
        nt[j] = convert(Integer,nt_full)
        t[j] =  (nt[j]-1)*dt[j]
    end
    return  GeometryIC(xloc,yloc,zloc,dt,nt,t)
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
    dt = Array{Any}(undef, nsrc); nt = Array{Any}(undef, nsrc); t = Array{Any}(undef, nsrc)
    nsamples = Array{Any}(undef, nsrc)
    for j=1:nsrc
        container[j] = split(data,j)
        dt[j] = data.blocks[j].summary["dt"][1]/1f3
        nt[j] = data.ns
        t[j] = (nt[j]-1)*dt[j]
        key=="source" ? nsamples[j] = data.ns : nsamples[j] = Int((data.blocks[j].endbyte - data.blocks[j].startbyte)/(240 + data.ns*4)*data.ns)
    end
    return  GeometryOOC(container,dt,nt,t,nsamples,key,segy_depth_key)
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
    dt = Array{Any}(undef, nsrc); nt = Array{Any}(undef, nsrc); t = Array{Any}(undef, nsrc)
    nsamples = Array{Any}(undef, nsrc)
    for j=1:nsrc
        container[j] = data[j]
        dt[j] = data[j].blocks[1].summary["dt"][1]/1f3
        nt[j] = data[j].ns
        t[j] = (nt[j]-1)*dt[j]
        key=="source" ? nsamples[j] = data[j].ns : nsamples[j] = Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data[j].ns*4)*data[j].ns)
    end
    return  GeometryOOC(container,dt,nt,t,nsamples,key,segy_depth_key)
end

# Load geometry from out-of-core Geometry container
function Geometry(geometry::GeometryOOC)
    nsrc = length(geometry.container)

    # read either source or receiver geometry
    if geometry.key=="source"
        params = ["SourceX","SourceY",geometry.segy_depth_key,"dt","ns"]
    elseif geometry.key=="receiver"
        params = ["GroupX","GroupY",geometry.segy_depth_key,"dt","ns"]
    else
        throw("Specified keyword not supported")
    end
    xloc = Array{Any}(undef, nsrc); yloc = Array{Any}(undef, nsrc); zloc = Array{Any}(undef, nsrc)
    dt = Array{Any}(undef, nsrc); nt = Array{Any}(undef, nsrc); t = Array{Any}(undef, nsrc)

    for j=1:nsrc

        header = read_con_headers(geometry.container[j], params, 1)
        if geometry.key=="source"
            xloc[j] = convert(Float32, get_header(header, params[1])[1])
            yloc[j] = convert(Float32, get_header(header, params[2])[1])
            zloc[j] = abs.(convert(Float32,get_header(header, params[3])[1]))
        else
            xloc[j] = convert(Array{Float32,1}, get_header(header, params[1]))
            yloc[j] = convert(Array{Float32,1}, get_header(header, params[2]))
            zloc[j] = abs.(convert(Array{Float32,1}, get_header(header, params[3])))
        end
        dt[j] = get_header(header, params[4])[1]/1f3
        nt[j] = convert(Integer, get_header(header, params[5])[1])
        t[j] =  (nt[j]-1)*dt[j]
    end
    return  GeometryIC(xloc,yloc,zloc,dt,nt,t)
end


###########################################################################################################################################

# Subsample in-core geometry structure
function subsample(geometry::GeometryIC,srcnum)
    if length(srcnum)==1
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
