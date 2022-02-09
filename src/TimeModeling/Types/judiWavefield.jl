############################################################
# judiWavefield ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2017

export judiWavefield, judiWavefieldException, judiDFTwavefield, muteWavefield, dump_wavefield, fft_wavefield

############################################################

mutable struct judiWavefield{T} <: judiMultiSourceVector{T}
	nsrc::Integer
    dt::T
	data::Vector{Array{T, N}} where N
end

mutable struct judiWavefieldException <: Exception
	msg :: String
end

############################################################

## outer constructors

"""
judiWavefield
        dt::Real
        data

Abstract vector for seismic wavefields.

Constructors
============

Construct wavefield vector from an info structure, a cell array of wavefields and the computational \\
time step dt:

    judiWavefield(info, nt, data)


"""
function judiWavefield(nsrc::Integer, dt::Real, data::Union{Array, PyCall.PyObject, String};  vDT::DataType=Float32)
	# length of vector
	dataCell = [vDT.(data) for j=1:nsrc]
	return judiWavefield{vDT}(nsrc, Float32(dt), dataCell)
end

function judiWavefield(dt::Real, data::Union{Array{Any,1}, Array{Array{T, N}, 1}};vDT::DataType=Float32) where {T, N}
	# length of vector
	nsrc = length(data)
	T != Float32 && (data = tof32.(data))
	return judiWavefield{vDT}(nsrc, Float32(dt), data)
end

############################################################
## overloaded multi_source functions
time_sampling(jv::judiVector) = (jv.dt for i=1:jv.nsrc)

####################################################################
## overloaded Base functions

conj(A::judiWavefield{vDT}) where vDT = judiWavefield{vDT}(A.nsrc, A.dt, conj(A.data))

jo_convert(::Type{T}, jw::judiWavefield{T}, ::Bool) where {T} = jw
jo_convert(::Type{T}, jw::judiWavefield{vT}, B::Bool) where {T, vT} = judiWavefield{T}(jw.nsrc, jv.dt, jo_convert(T, jw.data, B))
####################################################################

function push!(a::judiWavefield{T}, b::judiWavefield{T}) where T
	append!(a.data, b.data)
	a.info.nsrc += b.info.nsrc
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
		x = similar(x_in, Float32)
		for i=1:nsrc
			x.data[i] = real(ifft(x_in.data[i], 1)) * sqrt(nt)
		end
	end
	return x
end

function isapprox(x::judiWavefield, y::judiWavefield; rtol::Real=sqrt(eps()), atol::Real=0)
    x.info.nsrc == y.info.nsrc || throw(judiVectorException("Incompatible number of sources"))
    isapprox(x.data, y.data; rtol=rtol, atol=atol)
end

