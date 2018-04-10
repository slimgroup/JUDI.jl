############################################################
# judiVector # #################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiVector, judiVectorException, subsample, judiVector_to_SeisBlock, time_resample, time_resample!, judiTimeInterpolation, write_shot_record

############################################################

# structure for seismic data as an abstract vector
mutable struct judiVector{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
    name::String
    m::Integer
    n::Integer
    nsrc::Integer
    geometry::Geometry
    data
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

Construct vector for observed data from `SeisIO.SeisBlock`. `segy_depth_key` is the `SeisIO` keyword \\
that contains the receiver depth coordinate:

    judiVector(SeisIO.SeisBlock; segy_depth_key="RecGroupElevation")

Construct vector for observed data from out-of-core data container of type `SeisIO.SeisCon`:

    judiVector(SeisIO.SeisCon; segy_depth_key="RecGroupElevation")

Examples
========

(1) Construct data vector from `Geometry` structure and a cell array of shot records:

    dobs = judiVector(rec_geometry, shot_records)

(2) Construct data vector for a seismic wavelet (can be either cell arrays of individual\\
wavelets or a single wavelet as an array):

    q = judiVector(src_geometry, wavelet)

(3) Construct data vector from `SeisIO.SeisBlock` object:

    using SeisIO
    seis_block = segy_read("test_file.segy")
    dobs = judiVector(seis_block; segy_depth_key="RecGroupElevation")

(4) Construct out-of-core data vector from `SeisIO.SeisCon` object (for large SEG-Y files):

    using SeisIO
    seis_container = segy_scan("/path/to/data/directory","filenames",["GroupX","GroupY","RecGroupElevation","SourceDepth","dt"])
    dobs = judiVector(seis_container; segy_depth_key="RecGroupElevation")

"""
function judiVector(geometry::Geometry,data::Array; vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain type not supported"))

    # length of vector
    n = 1
    if typeof(geometry) == GeometryOOC
        nsrc = length(geometry.container)
        m = sum(geometry.nsamples)
    else
        nsrc = length(geometry.xloc)
        m = 0
        for j=1:nsrc
            m += length(geometry.xloc[j])*geometry.nt[j]
        end
    end
    dataCell = Array{Array}(nsrc)
    for j=1:nsrc
        dataCell[j] = data
    end
    return judiVector{Float32}("Seismic data vector",m,n,nsrc,geometry,dataCell)
end

# constructor if data is passed as a cell array
function judiVector(geometry::Geometry,data::Union{Array{Any,1},Array{Array,1}}; vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain and range types not supported"))

    # length of vector
    if typeof(geometry) == GeometryOOC
        nsrc = length(geometry.container)
        m = sum(geometry.nsamples)
    else
        nsrc = length(geometry.xloc)
        m = 0
        for j=1:nsrc
            m += length(geometry.xloc[j])*geometry.nt[j]
        end
    end
    n = 1
    return judiVector{Float32}("Seismic data vector",m,n,nsrc,geometry,data)
end


############################################################
# constructors from SEGY files or out-of-core containers

# contructor for in-core data container
function judiVector(data::SeisIO.SeisBlock; segy_depth_key="RecGroupElevation", vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain and range types not supported"))

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
    dataCell = Array{Array}(nsrc)
    full_data = convert(Array{Float32,2},data.data)
    for j=1:nsrc
        traces = find(src .== unique(src)[j])
        dataCell[j] = full_data[:,traces]
    end

    return judiVector{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for in-core data container and given geometry
function judiVector(geometry::Geometry, data::SeisIO.SeisBlock; vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain and range types not supported"))

    # length of data vector
    src = get_header(data,"FieldRecord")
    nsrc = length(unique(src))

    numTraces = get_header(data,"TraceNumber")[end] - get_header(data,"TraceNumber")[1] + 1
    numSamples = get_header(data,"ns")[end]
    m = numTraces*numSamples
    n = 1

    # fill data vector with pointers to data location
    dataCell = Array{Array}(nsrc)
    for j=1:nsrc
        traces = find(src .== unique(src)[j])
        dataCell[j] = convert(Array{Float32,2},data.data[:,traces])
    end

    return judiVector{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from single container
function judiVector(data::SeisIO.SeisCon; segy_depth_key="RecGroupElevation", vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain and range types not supported"))

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
    dataCell = Array{SeisIO.SeisCon}(nsrc)
    for j=1:nsrc
        dataCell[j] = split(data,j)
    end

    return judiVector{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for single out-of-core data container and given geometry
function judiVector(geometry::Geometry, data::SeisIO.SeisCon; vDT::DataType=Float32)
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
    dataCell = Array{SeisIO.SeisCon}(nsrc)
    for j=1:nsrc
        dataCell[j] = split(data,j)
    end

    return judiVector{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from cell array of containers
function judiVector(data::Array{SeisIO.SeisCon,1}; segy_depth_key="RecGroupElevation", vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain and range types not supported"))

    # length of data vector
    nsrc = length(data)
    numTraces = 0
    for j=1:nsrc
        numTraces += Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data.ns*4))
    end
    m = numTraces*data.ns
    n = 1

    # extract geometry from data container
    geometry = Geometry(data; key="receiver", segy_depth_key=segy_depth_key)

    # fill data vector with pointers to data location
    dataCell = Array{SeisIO.SeisCon}(nsrc)
    for j=1:nsrc
        dataCell[j] = data[j]
    end

    return judiVector{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end

# contructor for out-of-core data container from cell array of containers and given geometry
function judiVector(geometry::Geometry, data::Array{SeisIO.SeisCon}; vDT::DataType=Float32)
    vDT == Float32 || throw(judiVectorException("Domain and range types not supported"))

    # length of data vector
    nsrc = length(data)
    numTraces = 0
    for j=1:nsrc
        numTraces += Int((data[j].blocks[1].endbyte - data[j].blocks[1].startbyte)/(240 + data[j].ns*4))
    end
    m = numTraces*data[1].ns
    n = 1

    # fill data vector with pointers to data location
    dataCell = Array{SeisIO.SeisCon}(nsrc)
    for j=1:nsrc
        dataCell[j] = data[j]
    end

    return judiVector{Float32}("Julia seismic data container",m,n,nsrc,geometry,dataCell)
end



############################################################
## overloaded Base functions

# conj(jo)
conj{vDT}(a::judiVector{vDT}) =
    judiVector{vDT}("conj("*a.name*")",a.m,a.n,a.nsrc,a.geometry,a.data)

# transpose(jo)
transpose{vDT}(a::judiVector{vDT}) =
    judiVector{vDT}(""*a.name*".'",a.n,a.m,a.nsrc,a.geometry,a.data)

# ctranspose(jo)
ctranspose{vDT}(a::judiVector{vDT}) =
    judiVector{vDT}(""*a.name*"'",a.n,a.m,a.nsrc,a.geometry,a.data)

##########################################################


# +(judiVector, judiVector)
function +{avDT,bvDT}(a::judiVector{avDT}, b::judiVector{bvDT})
    size(a) == size(b) || throw(judiVectorException("dimension mismatch"))
    compareGeometry(a.geometry, b.geometry) == 1 || throw(judiVectorException("geometry mismatch"))
    c = deepcopy(a)
    c.data = a.data + b.data
    return c
end

# -(judiVector, judiVector)
function -{avDT,bvDT}(a::judiVector{avDT}, b::judiVector{bvDT})
    size(a) == size(b) || throw(judiVectorException("dimension mismatch"))
    compareGeometry(a.geometry, b.geometry) == 1 || throw(judiVectorException("geometry mismatch"))
    c = deepcopy(a)
    c.data = a.data - b.data
    return c
end

# +(judiVector, number)
function +{avDT}(a::judiVector{avDT},b::Number)
    c = deepcopy(a)
    c.data = c.data+b
    return c
end

# -(judiVector, number)
function -{avDT}(a::judiVector{avDT},b::Number)
    c = deepcopy(a)
    c.data = c.data-b
    return c
end

# *(judiVector, number)
function *{avDT}(a::judiVector{avDT},b::Number)
    c = deepcopy(a)
    c.data = c.data*b
    return c
end

# *(number, judiVector)
function *{bvDT}(a::Number,b::judiVector{bvDT})
    c = deepcopy(b)
    c.data = a*c.data
    return c
end

# /(judiVector, number)
function /{avDT}(a::judiVector{avDT},b::Number)
    c = deepcopy(a)
    if iszero(b)
        error("Division by zero")
    else
        c.data = c.data/b
    end
    return c
end

# *(joLinearFunction, judiVector)
function *{ADDT,ARDT,avDT}(A::joLinearFunction{ADDT,ARDT},v::judiVector{avDT})
    A.n == size(v,1) || throw(judiVectorException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiVector):",A.name,typeof(A),avDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# *(joLinearOperator, judiVector)
function *{ADDT,ARDT,avDT}(A::joLinearOperator{ADDT,ARDT},v::judiVector{avDT})
    A.n == size(v,1) || throw(judiVectorException("shape mismatch"))
    jo_check_type_match(ADDT,avDT,join(["DDT for *(joLinearFunction,judiVector):",A.name,typeof(A),avDT]," / "))
    V = A.fop(v)
    jo_check_type_match(ARDT,eltype(V),join(["RDT from *(joLinearFunction,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V
end

# vcat
function vcat{avDT,bvDT}(a::judiVector{avDT},b::judiVector{bvDT})
    typeof(a.geometry) == typeof(b.geometry) || throw(judiVectorException("Geometry type mismatch"))
    m = a.m + b.m
    n = 1
    nsrc = a.nsrc + b.nsrc

    if typeof(a.data) == Array{SeisIO.SeisCon,1} && typeof(b.data) == Array{SeisIO.SeisCon,1}
        data = Array{SeisIO.SeisCon}(nsrc)
    else
        data = Array{Array}(nsrc)
    end

    dt = Array{Any}(nsrc)
    nt = Array{Any}(nsrc)
    t = Array{Any}(nsrc)
    if typeof(data) == Array{SeisIO.SeisCon,1}
        nsamples = Array{Any}(nsrc)
    else
        xloc = Array{Any}(nsrc)
        yloc = Array{Any}(nsrc)
        zloc = Array{Any}(nsrc)
    end

    # Merge data sets and geometries
    for j=1:a.nsrc
        data[j] = a.data[j]
        if typeof(data) == Array{SeisIO.SeisCon,1}
            nsamples[j] = a.geometry.nsamples[j]
        else
            xloc[j] = a.geometry.xloc[j]
            yloc[j] = a.geometry.yloc[j]
            zloc[j] = a.geometry.zloc[j]
        end
        dt[j] = a.geometry.dt[j]
        nt[j] = a.geometry.nt[j]
        t[j] = a.geometry.t[j]
    end
    for j=a.nsrc+1:nsrc
        data[j] = b.data[j-a.nsrc]
        if typeof(data) == Array{SeisIO.SeisCon,1}
            nsamples[j] = b.geometry.nsamples[j-a.nsrc]
        else
            xloc[j] = b.geometry.xloc[j-a.nsrc]
            yloc[j] = b.geometry.yloc[j-a.nsrc]
            zloc[j] = b.geometry.zloc[j-a.nsrc]
        end
        dt[j] = b.geometry.dt[j-a.nsrc]
        nt[j] = b.geometry.nt[j-a.nsrc]
        t[j] = b.geometry.t[j-a.nsrc]
    end

    if typeof(data) == Array{SeisIO.SeisCon,1}
        geometry = GeometryOOC(data,dt,nt,t,nsamples,a.geometry.key,a.geometry.segy_depth_key)
    else
        geometry = Geometry(xloc,yloc,zloc,dt,nt,t)
    end
    nvDT = promote_type(avDT,bvDT)
    return judiVector{nvDT}(a.name,m,n,nsrc,geometry,data)
end

# dot product
function dot{avDT,bvDT}(a::judiVector{avDT}, b::judiVector{bvDT})
# Dot product for data containers
    size(a) == size(b) || throw(judiVectorException("dimension mismatch"))
    compareGeometry(a.geometry, b.geometry) == 1 || throw(judiVectorException("geometry mismatch"))
    dotprod = 0f0
    for j=1:a.nsrc
        dotprod += a.geometry.dt[j] * dot(vec(a.data[j]),vec(b.data[j]))
    end
    return dotprod
end

# norm
function norm{avDT}(a::judiVector{avDT}; p=2)
    x = 0.f0
    for j=1:a.nsrc
        x += (a.geometry.dt[j] * norm(vec(a.data[j]),p))^p
    end
    return x^(1.f0/p)
end

# abs
function abs{avDT}(a::judiVector{avDT})
    b = deepcopy(a)
    for j=1:a.nsrc
        b.data[j] = abs(a.data[j])
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
function subsample{avDT}(a::judiVector{avDT},srcnum)
    geometry = subsample(a.geometry,srcnum)     # Geometry of subsampled data container
    return judiVector(geometry,a.data[srcnum];vDT=avDT)
end

getindex(x::judiVector,a) = subsample(x,a)

# Create SeisBlock from judiVector container to write to file
function judiVector_to_SeisBlock{avDT}(d::judiVector{avDT},q::judiVector{avDT}; source_depth_key="SourceSurfaceElevation", receiver_depth_key="RecGroupElevation")

    typeof(d.geometry) == GeometryOOC && (d.geometry = Geometry(d.geometry))
    typeof(q.geometry) == GeometryOOC && (q.geometry = Geometry(q.geometry))

    blocks = Array{Any}(d.nsrc)
    count = 0
    for j=1:d.nsrc

        # create SeisBlock
        blocks[j] = SeisBlock(d.data[j])
        numTraces = size(d.data[j],2)
        traceNumbers = convert(Array{Integer,1},count+1:count+numTraces)

        # set headers
        set_header!(blocks[j], "GroupX", convert(Array{Integer,1},round.(d.geometry.xloc[j]*1f3)))
        if length(d.geometry.yloc[j]) == 1
            set_header!(blocks[j], "GroupY", Int(round(d.geometry.yloc[j]*1f3)))
        else
            set_header!(blocks[j], "GroupY", convert(Array{Integer,1},round.(d.geometry.yloc[j]*1f3)))
        end
        set_header!(blocks[j], receiver_depth_key, convert(Array{Integer,1},round.(d.geometry.zloc[j]*1f3)))
        set_header!(blocks[j], "SourceX", Int(round.(q.geometry.xloc[j]*1f3)))
        set_header!(blocks[j], "SourceY", Int(round.(q.geometry.yloc[j]*1f3)))
        set_header!(blocks[j], source_depth_key, Int(round.(q.geometry.zloc[j]*1f3)))

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

function write_shot_record(srcGeometry,srcData,recGeometry,recData,options)
    q = judiVector(srcGeometry,srcData)
    d = judiVector(recGeometry,recData)
    file = join([string(options.file_name),"_",string(srcGeometry.xloc[1][1]),"_",string(srcGeometry.yloc[1][1]),".segy"])
    block_out = judiVector_to_SeisBlock(d,q)
    segy_write(join([options.file_path,"/",file]), block_out)
    container = scan_file(join([options.file_path,"/",file]),["GroupX","GroupY","dt","SourceSurfaceElevation","RecGroupElevation"])
    return container
end


function time_resample!(x::judiVector,dt_new;order=2)
    for j=1:x.nsrc
        numTraces = size(x.data[j],2)
        timeAxis = 0:x.geometry.dt[j]:x.geometry.t[j]
        timeInterp = 0:dt_new:x.geometry.t[j]
        dataInterp = zeros(Float32,length(timeInterp),numTraces)
        for k=1:numTraces
            spl = Spline1D(timeAxis,x.data[j][:,k];k=order)
            dataInterp[:,k] = spl(timeInterp)
        end
        x.data[j] = dataInterp
        x.geometry.dt[j] = dt_new
        x.geometry.nt[j] = length(timeInterp)
        x.geometry.t[j] = (x.geometry.nt[j] - 1)*x.geometry.dt[j]
    end
    return judiVector(x.geometry,x.data)
end

function time_resample(x::judiVector,dt_new;order=2)
    xout = deepcopy(x)
    for j=1:x.nsrc
        numTraces = size(x.data[j],2)
        timeAxis = 0:x.geometry.dt[j]:x.geometry.t[j]
        timeInterp = 0:dt_new:x.geometry.t[j]
        dataInterp = zeros(Float32,length(timeInterp),numTraces)
        for k=1:numTraces
            spl = Spline1D(timeAxis,x.data[j][:,k];k=order)
            dataInterp[:,k] = spl(timeInterp)
        end
        xout.data[j] = dataInterp
        xout.geometry.dt[j] = dt_new
        xout.geometry.nt[j] = length(timeInterp)
        xout.geometry.t[j] = (x.geometry.nt[j] - 1)*x.geometry.dt[j]
    end
    return judiVector(xout.geometry,xout.data)
end

function judiTimeInterpolation(geometry::Geometry,dt_coarse,dt_fine)
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
    I = joLinearFunctionFwdT(m,n,
                             v -> time_resample(v,dt_fine),
                             w -> time_resample(w,dt_coarse),
                             Float32,Float32,name="Time resampling")
    return I
end

function scale!(a::Number,x::judiVector)
    for j=1:x.nsrc
        x.data[j] *= a
    end
end

function scale!(x::judiVector,a::Number)
    for j=1:x.nsrc
        x.data[j] *= a
    end
end

broadcast!(.*, x::judiVector, y::judiVector, a::Number) = scale!(y, a)

function broadcast!(identity, x::judiVector, y::judiVector)
    copy!(x,y)
end

function broadcast!(identity, x::judiVector, a::Number, y::judiVector, z::judiVector)
    scale!(y,a)
    copy!(x, y + z)
end

function copy!(x::judiVector,y::judiVector)
    for j=1:x.nsrc
        x.data[j] = y.data[j]
    end
    x.geometry = deepcopy(y.geometry)
end

#broadcast!(identity, x::judiVector, y::judiVector) = copy!(x,y)

function axpy!(a::Number,X::judiVector,Y::judiVector)
    for j=1:Y.nsrc
        Y.data[j] = a*X.data[j] + Y.data[j]
    end
end

similar(x::judiVector, kwargs...) = judiVector(x.geometry, x.data)*0f0


###########################################################################################################

# Overload base function for SeisIO objects

vec(x::SeisIO.SeisCon) = vec(x[1].data)
norm(x::SeisIO.IBMFloat32; kwargs...) = norm(convert(Float32,x); kwargs...)
dot(x::SeisIO.IBMFloat32, y::SeisIO.IBMFloat32) = dot(convert(Float32,x), convert(Float32,y))
dot(x::SeisIO.IBMFloat32, y::Float32) = dot(convert(Float32,x), y)
dot(x::Float32, y::SeisIO.IBMFloat32) = dot(x, convert(Float32,y))

# binary operations return dense arrays
+(x::SeisIO.SeisCon, y::SeisIO.SeisCon) = +(x[1].data,y[1].data)
+(x::SeisIO.IBMFloat32, y::SeisIO.IBMFloat32) = +(convert(Float32,x),convert(Float32,y))
+(x::SeisIO.IBMFloat32, y::Float32) = +(convert(Float32,x),y)
+(x::Float32, y::SeisIO.IBMFloat32) = +(x,convert(Float32,y))

-(x::SeisIO.SeisCon, y::SeisIO.SeisCon) = -(x[1].data,y[1].data)
-(x::SeisIO.IBMFloat32, y::SeisIO.IBMFloat32) = -(convert(Float32,x),convert(Float32,y))
-(x::SeisIO.IBMFloat32, y::Float32) = -(convert(Float32,x),y)
-(x::Float32, y::SeisIO.IBMFloat32) = -(x,convert(Float32,y))

+(a::Number, x::SeisIO.SeisCon) = +(a,x[1].data)
+(x::SeisIO.SeisCon, a::Number) = +(x[1].data,a)

-(a::Number, x::SeisIO.SeisCon) = -(a,x[1].data)
-(x::SeisIO.SeisCon, a::Number) = -(x[1].data,a)

*(a::Number, x::SeisIO.SeisCon) = *(a,x[1].data)
*(x::SeisIO.SeisCon, a::Number) = *(x[1].data,a)

/(a::Number, x::SeisIO.SeisCon) = /(a,x[1].data)
/(x::SeisIO.SeisCon, a::Number) = /(x[1].data,a)
