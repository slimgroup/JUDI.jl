############################################################
# judiWavefield ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2017

export judiWavefield, fft, ifft

############################################################

mutable struct judiWavefield{T} <: judiMultiSourceVector{T}
    nsrc::Integer
    dt::Vector{T}
    data::Vector{<:Union{Array{T, N}, PyArray}} where N
end

############################################################

## outer constructors

"""
judiWavefield
        nsrc::Integer
        dt::Real
        data

Abstract vector for seismic wavefields.

Constructors
============

Construct wavefield vector from an info structure, a cell array of wavefields and the computational \\
time step dt:

    judiWavefield(nsrc, dt, data)


"""
function judiWavefield(nsrc::Integer, dt::Real, data::Array{T, N}) where {T<:Number, N}
	# length of vector
	dataCell = [convert(Array{Float32, N}, data) for j=1:nsrc]
	return judiWavefield{Float32}(nsrc, [Float32(dt) for i=1:nsrc], dataCell)
end

function judiWavefield(dt::Real, data::Vector{Array{T, N}}) where {T, N}
	# length of vector
	nsrc = length(data)
	T != Float32 && (data = tof32.(data))
	return judiWavefield{Float32}(nsrc, [Float32(dt) for i=1:nsrc], data)
end

judiWavefield(dt::Real, data::Vector{Any}) = judiWavefield(dt, tof32.(data))
judiWavefield(dt::Real, data::Array{T, N}) where {T<:Number, N} = judiWavefield(1, dt, data)

conj(w::judiWavefield{T}) where {T<:Complex} = judiWavefield{R}(w.nsrc, w.dt, conj(w.data))

############################################################
## overloaded multi_source functions
time_sampling(jv::judiWavefield) = jv.dt

####################################################################
# JOLI conversion
jo_convert(::Type{T}, jw::judiWavefield{T}, ::Bool) where {T<:Number} = jw
jo_convert(::Type{T}, jw::judiWavefield{vT}, B::Bool) where {T<:Number, vT} = judiWavefield{T}(jw.nsrc, jv.dt, jo_convert.(T, jw.data, B))
zero(::Type{T}, v::judiWavefield{vT}; nsrc::Integer=v.nsrc) where {T, vT} = judiWavefield{T}(nsrc, v.dt, T(0) .* v.data[1:nsrc])
zero(::Type{T}, v::judiWavefield{vT}; nsrc::Integer=v.nsrc) where {T<:Real, vT<:Complex} = judiWavefield{T}(nsrc, v.dt, T(0) .* real(v.data[1:nsrc]))
(w::judiWavefield)(x::Vector{<:Array}) = judiWavefield(w.dt, x)

function copy!(jv::judiWavefield, jv2::judiWavefield)
    v.data .= jv2.data
    jv.dt = jv2.dt
    jv
end

copyto!(jv::judiWavefield, jv2::judiWavefield) = copy!(jv, jv2)
make_input(w::judiWavefield) = w.data[1]
check_compat(ms::Vararg{judiWavefield, N}) where N = all(y -> y.dt == first(ms).dt, ms)

getindex(a::judiWavefield{T}, srcnum::RangeOrVec) where T = judiWavefield{T}(length(srcnum), a.dt[srcnum], a.data[srcnum])
####################################################################

function push!(a::judiWavefield{T}, b::judiWavefield{T}) where T
    append!(a.data, b.data)
    append!(a.dt, b.dt)
    a.nsrc += b.nsrc
end

# DFT operator for wavefields, acts along time dimension
function fft(x_in::judiWavefield{T}) where T
    x = similar(x_in, Complex{Float32})
    for i=1:x_in.nsrc
        x.data[i] = fft(x_in.data[i], 1)/sqrt(size(x_in.data[i], 1))
    end
    return x
end

function ifft(x_in::judiWavefield{T}) where T
    x = similar(x_in, Float32)
    for i=1:x_in.nsrc
        x.data[i] = real(ifft(x_in.data[i], 1)) * sqrt(size(x_in.data[i], 1))
    end
    return x
end

function isapprox(x::judiWavefield, y::judiWavefield; rtol::Real=sqrt(eps()), atol::Real=0)
    isapprox(x.data, y.data; rtol=rtol, atol=atol) && isapprox(x.dt, y.dt;rtol=rtol, atol=atol)
end

