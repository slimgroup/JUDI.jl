export DataMute, FrequencyFilter, judiTimeDerivative, judiTimeIntegration
export judiFilter, low_filter, judiDataMute, muteshot

############################################ Data mute ###############################################
"""
    struct DataMute{T, mode} <: DataPreconditioner{T, T}
        srcGeom::Geometry
        recGeom::Geometry
        vp::Vector{T}
        t0::Vector{T}
        taperwidth::Vector{Int64}
    end

Data mute linear operator a where {T, N}sociated with source `srcGeom` and receiver `recGeom` geometries used to compute the distance to the source
for each trace in the data. Data mute preconditionner. Supports two modes (:reflection, :turning) that mute either the turning waves (standard direct wave mute) or mutes the reflections.
A cosine tapr is applied with width `taperwidth` to avoid abrupt change and infinite frequency jumps in the data.

Constructors
============

    judiDataMute(srcGeom, recGeom; vp=1500, t0=.1, mode=:reflection, taperwidth=floor(Int, 2/t0))

Construct the data mute operator from the source `srcGeom` and receiver `recGeom` geometries.

    judiDataMute(q, d; vp=1500, t0=.1, mode=:reflection, taperwidth=floor(Int, 2/t0))

Construct the data mute operator from the judivector source `q` and judivector data `d`.

Parameters
============
The following optional paramet where {T, N}rs control the muting operator

- `vp`: P wave velocity of the direct wave (usually water velocity). Can be a constant or a Vector with one value per source position. Devfaults to `1500m/s`
- `t0`: Time shift in seconds (usually width of the wavelet). Defaults to ``.1 sec``
- `mode`: 
    `:reflection` to keep the reflections and mute above the direct wave (i.e for RTM)
    `:turning` to keep the turning waves and mute below the direct wave (i.e for FWI)
- `taperwidth`: Width of the cosine taper in number of samples. Defaults to `2 / t0`
"""
struct DataMute{T, mode} <: DataPreconditioner{T, T}
    m::Integer
    srcGeom::Geometry
    recGeom::Geometry
    vp::Vector{T}
    t0::Vector{T}
    taperwidth::Vector{Int64}
end

function judiDataMute(srcGeom::Geometry, recGeom::Geometry; vp=1500, t0=.1, mode=:reflection, taperwidth=floor(Int, 2/t0))
    mode ∈ [:reflection, :turning] || throw(ArgumentError("Only reflection (mute turning) and turning (mute refelctions) modes supported"))
    nsrc = get_nsrc(srcGeom)
    get_nsrc(recGeom) == nsrc || throw(ArgumentError("Incompatible geometries with $(nsrc) and $(get_nsrc(recGeom)) number of sources"))
    VP = Vector{Float32}(undef, nsrc); VP .= vp
    T0 = Vector{Float32}(undef, nsrc); T0 .=t0
    TW = Vector{Int64}(undef, nsrc); TW .= taperwidth
    m = n_samples(recGeom)
    return DataMute{Float32, mode}(m, srcGeom, recGeom, VP, T0, TW)
end
    
judiDataMute(q::judiVector, d::judiVector; kw...) = judiDataMute(q.geometry, d.geometry; kw...)

# Implementation
matvec_T(D::DataMute{T, mode} , x::AbstractVector{T}) where {T, mode} = matvec(D, x)  # Basically a mask so symmetric and self adjoint
matvec(D::DataMute{T, mode} , x::AbstractVector{T}) where {T, mode} = muteshot(x, D.srcGeom, D.recGeom; vp=D.vp, t0=D.t0, mode=mode, taperwidth=D.taperwidth)

# Real diagonal operator
conj(I::DataMute{T, mode}) where {T, mode} = I
adjoint(I::DataMute{T, mode}) where {T, mode} = I
transpose(I::DataMute{T, mode}) where {T, mode} = I

# getindex for source subsampling
function getindex(P::DataMute{T, mode}, i) where {T, mode}
    geomi = P.recGeom[i]
    m = n_samples(geomi)
    inds = isa(i, Integer) ? (i:i) : i
    DataMute{T, mode}(m, P.srcGeom[inds], geomi, P.vp[inds], P.t0[inds], P.taperwidth[inds])
end

function _mutetrace!(t::AbstractVector{T}, taper::AbstractVector{T}, i::Integer, taperwidth::Integer, ::Val{:reflection}) where T
    t[1:i-taperwidth] .= 0f0
    t[i-taperwidth+1:i] .*= taper
end

function _mutetrace!(t::AbstractVector{T}, taper::AbstractVector{T}, i::Integer, taperwidth::Integer, ::Val{:turning}) where T
    t[i+taperwidth+1:end] .= 0f0
    t[i+1:i+taperwidth] .*= taper
end

function muteshot!(shot::AbstractMatrix{T}, rGeom::Geometry, srcGeom::Geometry; vp=1500, t0=.1, mode=:reflection, taperwidth=floor(Int, 2/t0)) where T
    sGeom = Geometry(srcGeom)
    rGeom = Geometry(rGeom)
    nt, nrec = size(shot)
    taper = _taper(Val(mode), taperwidth)
    # Loop over traces
    @inbounds for t=1:nrec
        r = radius(rGeom, sGeom, t) 
        tt = 1f3 * (r / vp + t0) / rGeom.dt[1]
        i = min(max(1, floor(Int, tt)), nt)
        if _tapew(i, taperwidth, nt, Val(mode))
            _mutew!(view(shot, :, t), taper, i, nt, Val(mode))
        else
            _mutetrace!(view(shot, :, t), taper, i, taperwidth, Val(mode))
        end
    end
end

function muteshot(shot::Vector{T}, srcGeom::Geometry, recGeom::Geometry;
                   vp=1500, t0=.1, mode=:reflection, taperwidth=floor(Int, 2/t0)) where {T<:Number}
        sr = reshape(shot, recGeom)
        for s=1:get_nsrc(recGeom)
            sri = view(sr, :, :, s)
            muteshot!(sri, recGeom[s], srcGeom[s]; vp=vp[s], t0=t0[s], mode=mode, taperwidth=taperwidth[s])
        end
        return vec(vcat(sr...))
end

function muteshot(shot::judiVector, srcGeom::Geometry; vp=1500, t0=.1, mode=:reflection, taperwidth=20)
    out = deepcopy(get_data(shot))
    for s=1:out.nsrc
        muteshot!(out.data[s], shot.geometry[s], srcGeom[s]; vp=vp[s], t0=t0[s], mode=mode, taperwidth=taperwidth[s])
    end
    out
end

muteshot(shot::judiVector, srcGeom::Geometry, recGeom::Geometry; kw...) = muteshot(shot, srcGeom; kw...)

"""
    struct FrequencyFilter
        recGeom

Bandpass filter linear operator. Filters the input `judiVector` or `Vector`

Constructor
============
    judiFilter(geometry, fmin, fmax) 
    judiFilter(judiVector, fmin, fmax)
"""
struct FrequencyFilter{T, fm, FM} <: DataPreconditioner{T, T}
    m::Integer
    recGeom::Geometry
end

judiFilter(geometry::Geometry, fmin::T, fmax::T) where T = judiFilter(geometry, Float32(fmin), Float32(fmax))
judiFilter(geometry::Geometry, fmin::Float32, fmax::Float32) = FrequencyFilter{Float32, fmin, fmax}(n_samples(geometry), geometry)
judiFilter(v::judiVector, fmin, fmax) = judiFilter(v.geometry, fmin, fmax)

function matvec(D::FrequencyFilter{T, fm, FM}, x::Vector{T}) where {T, fm, FM}
    dr = reshape(x, D.recGeom)
    for j=1:get_nsrc(D.recGeom)
        dri = view(dr, :, :, j)
        filter!(dri, dri, D.recGeom[j].dt[1]; fmin=fm, fmax=FM)
    end
    return reshape(vcat(dr...), size(x))
end

function matvec(::FrequencyFilter{T, fm, FM}, Din::judiVector{T, AT}) where {T, fm, FM, AT}
    Dout = deepcopy(Din)	
    for j=1:Dout.nsrc
        filter!(Dout.data[j], Din.data[j], Din.geometry[j].dt[1]; fmin=fm, fmax=FM)
    end
    return Dout	
end

# getindex for source subsampling
function getindex(P::FrequencyFilter{T, fm, FM}, i) where {T, fm, FM}
    geomi = P.recGeom[i]
    return FrequencyFilter{T, fm, FM}(n_samples(geomi), geomi)
end

# filtering is self-adjoint (diagonal in fourier domain)
matvec_T(D::FrequencyFilter{T, fm, FM}, x) where {T, fm, FM} = matvec(D, x)

# Real diagonal operator
conj(I::FrequencyFilter{T}) where T = I
adjoint(I::FrequencyFilter{T}) where T = I
transpose(I::FrequencyFilter{T}) where T = I

function filter!(dout::AbstractArray{T, N}, din::AbstractArray{T, N}, dt::T; fmin=T(0.01), fmax=T(100)) where {T, N}
    responsetype = Bandpass(fmin, fmax; fs=1e3/dt)	
    designmethod = Butterworth(5)
    tracefilt!(x, y) = filt!(x, digitalfilter(responsetype, designmethod), y)
    map(i-> tracefilt!(selectdim(dout, N, i), selectdim(din, N, i)), 1:size(dout, 2))
end

low_filter(Din::judiVector; fmin=0.01, fmax=100.0) = judiFilter(Din.geometry, fmin, fmax)*Din
low_filter(Din::judiVector, ::Any; fmin=0.01, fmax=100.0) = low_filter(Din; fmin=fmin, fmax=fmax)

"""
    low_filter(Din, dt_in; fmin=0, fmax=25)

Performs a causal band pass filtering [fmin, fmax] on the input data bases on its sampling rate `dt`.
"""
function low_filter(Din::Matrix{T}, dt_in; fmin=0.01, fmax=100.0) where T
    out = similar(Din)
    filter!(out, Din, dt_in; fmin=T(fmin), fmax=T(fmax))
    return out
end

# Legacy top mute is deprecated since only working for marine data
judiMarineTopmute2D(::Integer, geometry::Geometry; params=Array{Any}(undef, 3), flipmask=false) = throw(MethodError(judiMarineTopmute2D, "judiMarineTopmute2D is deprecated due to its limitations and inaccuracy, please use judiDataMute"))

"""
    TimeDifferential{K}
        recGeom

Differential operator of order `K` to be applied along the time dimension. Applies the ilter `w^k` where `k` is the order. For example,
the tinme derivative is `TimeDifferential{1}` and the time integration is `TimeDifferential{-1}`

Constructor
============

    judiTimeIntegration(recGeom, order)
    judiTimeIntegration(judiVector, order)

    judiTimeDerivative(recGeom, order)
    judiTimeDerivative(judiVector, order)
"""

struct TimeDifferential{T, K} <: DataPreconditioner{T, T}
    m::Integer
    recGeom::Geometry
end

TimeDifferential(g::Geometry{T}, order::Number) where T = TimeDifferential{T, order}(n_samples(g), g)

judiTimeDerivative(v::judiVector{T, AT}, order::Number) where {T, AT} = TimeDifferential(v.geometry, order)
judiTimeDerivative(g::Geometry{T}, order::Number) where {T} = TimeDifferential(g, order)

judiTimeIntegration(v::judiVector{T, AT}, order::Number) where {T, AT} = TimeDifferential(v.geometry, -order)
judiTimeIntegration(g::Geometry{T}, order::Number) where {T} = TimeDifferential(g, -order)

# Real diagonal operator
conj(D::TimeDifferential{T, K}) where {T, K} = D
adjoint(D::TimeDifferential{T, K}) where {T, K} = D
transpose(D::TimeDifferential{T, K}) where {T, K} = D
inv(D::TimeDifferential{T, K}) where {T, K} = TimeDifferential{T, -K}(D.m, D.recGeom)

# diagonal in fourier domain so self-adjoint
matvec_T(D::TimeDifferential{T, K}, x) where {T, K} = matvec(D, x)

# getindex for source subsampling
function getindex(P::TimeDifferential{T, K}, i) where {T, K}
    geomi = P.recGeom[i]
    m = n_samples(geomi)
    TimeDifferential{T, K}(m, geomi)
end

function matvec(D::TimeDifferential{T, K}, x::judiVector{T, AT}) where {T, AT, K}
    out = similar(x)
    for s=1:out.nsrc
        # make omega^K
        ω = 2 .* pi .* fftfreq(D.recGeom.nt[s], 1/D.recGeom.dt[s])
        ω[ω.==0] .= 1f0
        ω .= abs.(ω).^K
        out.data[s] .= real.(ifft(ω .* fft(x.data[s], 1), 1))
    end
    return out
end

function matvec(D::TimeDifferential{T, K}, x::Array{T}) where {T, K}
    xr = reshape(x, D.recGeom)
    # make omega^K
    ω = 2 .* pi .* fftfreq(D.recGeom.nt[1], 1/D.recGeom.dt[1])
    ω[ω.==0] .= 1f0
    ω .= abs.(ω).^K
    out = real.(ifft(ω .* fft(xr, 1), 1))
    return reshape(out, size(x))
end
