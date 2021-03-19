############################################################
# judiRHS ## #################################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: January 2017

export judiRHS, judiRHSexception

############################################################

struct judiRHS{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
    name::String
    m::Integer
    n::Integer
    info::Info
    geometry::Geometry
    data
end

mutable struct judiRHSexception <: Exception
    msg :: String
end


############################################################

## outer constructors

"""
    judiRHS
        name::String
        m::Integer
        n::Integer
        info::Info
        geometry::Geometry
        data

Abstract sparse vector for right-hand-sides of the modeling operators. The `judiRHS` vector has the\\
dimensions of the full time history of the wavefields, but contains only the data defined at the \\
source or receiver positions (i.e. wavelets or shot records).

Constructor
==========

    judiRHS(info, geometry, data)

Examples
========

Assuming `Pr` and `Ps` are projection operators of type `judiProjection` and `dobs` and `q` are\\
seismic vectors of type `judiVector`, then a `judiRHS` vector can be created as follows:

    rhs = Pr'*dobs    # right-hand-side with injected observed data
    rhs = Ps'*q    # right-hand-side with injected wavelet

"""
function judiRHS(info,geometry,data;vDT::DataType=Float32)
    vDT == Float32 || throw(judiRHSexception("Domain type not supported"))
    # length of vector
    m = info.n * sum(info.nt)
    n = 1
    return judiRHS{Float32}("judiRHS",m,n,info,geometry,data)
end

####################################################################
## overloaded Base functions

# conj(jo)
conj(A::judiRHS{vDT}) where vDT =
    judiRHS{vDT}("conj("*A.name*")",A.m,A.n,A.info,A.geometry,A.data)

# transpose(jo)
transpose(A::judiRHS{vDT}) where vDT =
    judiRHS{vDT}(""*A.name*".'",A.n,A.m,A.info,A.geometry,A.data)

# adjoint(jo)
adjoint(A::judiRHS{vDT}) where vDT =
    judiRHS{vDT}(""*A.name*"'",A.n,A.m,A.info,A.geometry,A.data)

####################################################################

# +(judiRHS,judiRHS)
function +(A::judiRHS{avDT}, B::judiRHS{bvDT}) where {avDT, bvDT}

    # Error checking
    size(A) == size(B) || throw(judiRHSexception("shape mismatch"))
    compareInfo(A.info, B.info) == true || throw(judiRHSexception("info mismatch"))
    isequal(A.geometry.nt,B.geometry.nt) == true || throw(judiRHSexception("sample number mismatch"))
    isequal(A.geometry.dt,B.geometry.dt) == true || throw(judiRHSexception("sample interval mismatch"))
    isequal(A.geometry.t,B.geometry.t) == true || throw(judiRHSexception("recording time mismatch"))

    # Size
    m = A.info.n * sum(A.info.nt)
    n = 1

    # merge geometries and data
    xloc = [vcat(A.geometry.xloc[j], B.geometry.xloc[j]) for j=1:A.info.nsrc]
    yloc = [vcat(A.geometry.yloc[j], B.geometry.yloc[j]) for j=1:A.info.nsrc]
    zloc = [vcat(A.geometry.zloc[j], B.geometry.zloc[j]) for j=1:A.info.nsrc]
    dt = vcat(A.geometry.dt, B.geometry.dt)
    nt = vcat(A.geometry.nt, B.geometry.nt)
    t = vcat(A.geometry.t, B.geometry.t)
    data = Array{Array{Float32, 2}}(undef, A.info.nsrc)

    for j=1:A.info.nsrc
        data[j] = [A.data[j] B.data[j]]
    end
    geometry = GeometryIC{Float32}(xloc,yloc,zloc,dt,nt,t)
    nvDT = promote_type(avDT,bvDT)

    return judiRHS{nvDT}("judiRHS",m,n,A.info,geometry,data)
end

# -(judiRHS,judiRHS)
function -(A::judiRHS{avDT}, B::judiRHS{bvDT}) where {avDT, bvDT}

    # Error checking
    size(A) == size(B) || throw(judiRHSexception("shape mismatch"))
    compareInfo(A.info, B.info) == true || throw(judiRHSexception("info mismatch"))
    isequal(A.geometry.nt,B.geometry.nt) == true || throw(judiRHSexception("sample number mismatch"))
    isequal(A.geometry.dt,B.geometry.dt) == true || throw(judiRHSexception("sample interval mismatch"))
    isequal(A.geometry.t,B.geometry.t) == true || throw(judiRHSexception("recording time mismatch"))

    # Size
    m = A.info.n * sum(A.info.nt)
    n = 1

    # merge geometries and data
    xloc = [vcat(A.geometry.xloc[j], B.geometry.xloc[j]) for j=1:A.info.nsrc]
    yloc = [vcat(A.geometry.yloc[j], B.geometry.yloc[j]) for j=1:A.info.nsrc]
    zloc = [vcat(A.geometry.zloc[j], B.geometry.zloc[j]) for j=1:A.info.nsrc]
    dt = vcat(A.geometry.dt, B.geometry.dt)
    nt = vcat(A.geometry.nt, B.geometry.nt)
    t = vcat(A.geometry.t, B.geometry.t)
    data = Array{Array{Float32, 2}}(undef, A.info.nsrc)

    for j=1:A.info.nsrc
        data[j] = [A.data[j] -B.data[j]]
    end
    geometry = GeometryIC{Float32}(xloc,yloc,zloc,dt,nt,t)
    nvDT = promote_type(avDT,bvDT)

    return judiRHS{nvDT}("judiRHS",m,n,A.info,geometry,data)
end

function subsample(a::judiRHS{avDT},srcnum) where avDT
    info = Info(a.info.n, length(srcnum), a.info.nt[srcnum])
    geometry = subsample(a.geometry,srcnum)     # Geometry of subsampled data container
    return judiRHS(info,geometry,a.data[srcnum];vDT=avDT)
end

getindex(x::judiRHS,a) = subsample(x,a)
