############################################################
# judiWavefield ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2017

export judiWavefield, fft_wavefield

############################################################

mutable struct judiWavefield{T} <: judiMultiSourceVector{T}
    nsrc::Integer
    dt::T
    data::Vector{Array{T, N}} where N
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
function judiWavefield(nsrc::Integer, dt::Real, data::Union{Array{T, N}, PyCall.PyObject, String};  vDT::DataType=Float32) where {T<:Number, N}
	# length of vector
	dataCell = [vDT.(data) for j=1:nsrc]
	return judiWavefield{vDT}(nsrc, Float32(dt), dataCell)
end

function judiWavefield(dt::Real, data::Union{Vector{Any}, Vector{Array{T, N}}};vDT::DataType=Float32) where {T, N}
	# length of vector
	nsrc = length(data)
	T != Float32 && (data = tof32.(data))
	return judiWavefield{vDT}(nsrc, Float32(dt), data)
end

conj(w::judiWavefield{T}) where {T<:Complex} = judiWavefield{R}(w.nsrc, w.dt, conj(w.data))

############################################################
## overloaded multi_source functions
time_sampling(jv::judiWavefield) = [jv.dt for i=1:jv.nsrc]

####################################################################
# JOLI conversion
jo_convert(::Type{T}, jw::judiWavefield{T}, ::Bool) where {T<:Number} = jw
jo_convert(::Type{T}, jw::judiWavefield{vT}, B::Bool) where {T<:Number, vT} = judiWavefield{T}(jw.nsrc, jv.dt, jo_convert.(T, jw.data, B))
zero(::Type{T}, v::judiWavefield{vT}) where {T, vT} = judiWavefield{T}(v.nsrc, v.dt, T(0) .* v.data)
(w::judiWavefield)(x::Vector{<:Array}) = judiWavefield(w.dt, x)

function copy!(jv::judiWavefield, jv2::judiWavefield)
    v.data .= jv2.data
    jv.dt = jv2.dt
    jv
end

copyto!(jv::judiWavefield, jv2::judiWavefield) = copy!(jv, jv2)

make_input(w::judiWavefield, dtComp) = (w.data[1], nothing)

check_compat(ms::Vararg{judiWavefield, N}) where N = all(y -> y.dt == first(ms).dt, ms)

getindex(a::judiWavefield{T}, srcnum::RangeOrVec) where T = judiWeights{T}(length(srcnum), a.dt[srcnum], a.data[srcnum])
####################################################################

function push!(a::judiWavefield{T}, b::judiWavefield{T}) where T
    append!(a.data, b.data)
    a.nsrc += b.nsrc
end

# DFT operator for wavefields, acts along time dimension
function fft_wavefield(x_in::judiWavefield{T}, mode) where T
    nsrc = x_in.nsrc
    nt = size(x_in.data[1], 1)
    if mode==1
        x = similar(x_in, Complex{Float32})
        for i=1:nsrc
            x.data[i] = fft(x_in.data[i], 1)/sqrt(nt)
        end
    elseif mode==-1
        data = Vector{Array{real(T), ndims(x_in.data[1])}}(undef, x_in.nsrc)
        for i=1:nsrc
            data[i] = real(ifft(x_in.data[i], 1)) * sqrt(nt)
        end
        x = judiWavefield{Float32}(x_in.nsrc, x_in.dt, data)
    end
    return x
end

function isapprox(x::judiWavefield, y::judiWavefield; rtol::Real=sqrt(eps()), atol::Real=0)
    isapprox(x.data, y.data; rtol=rtol, atol=atol) && isapprox(x.dt, y.dt;rtol=rtol, atol=atol)
end

