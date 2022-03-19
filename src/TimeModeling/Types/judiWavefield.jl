############################################################
# judiWavefield ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2017

export judiWavefield, judiWavefieldException, judiDFTwavefield, muteWavefield, dump_wavefield, fft_wavefield

############################################################

mutable struct judiWavefield{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
	name::String
	m::Integer
	n::Integer
	info::Info
    dt
	data
end

mutable struct judiWavefieldException <: Exception
	msg :: String
end

############################################################

## outer constructors

"""
judiWavefield
        name::String
        m::Integer
        n::Integer
        info::Info
        dt::Real
        data

Abstract vector for seismic wavefields.

Constructors
============

Construct wavefield vector from an info structure, a cell array of wavefields and the computational \\
time step dt:

    judiWavefield(info, nt, data)


"""
function judiWavefield(info,dt::Real,data::Union{AbstractArray, String};  vDT::DataType=Float32)

	# length of vector
	m = info.n * sum(info.nt)
	n = 1
	dataCell = [vDT.(data) for j=1:info.nsrc]

	return judiWavefield{vDT}("judiWavefield",m,n,info, Float32(dt), dataCell)
end

function judiWavefield(info, dt::Real, data::Union{Array{Any,1}, Array{Array{T, N}, 1}};vDT::DataType=Float32) where {T, N}
	# length of vector
	nsrc = length(data)
	nsrc != info.nsrc && throw("Different number of sources in info ($(info.nsrc)) and data array ($nsrc)")
	m = info.n * sum(info.nt)
	n = 1
	T != Float32 && (data = tof32.(data))
	return judiWavefield{vDT}("judiWavefield",m,n,info, Float32(dt), data)
end


####################################################################
## overloaded Base functions

# conj(jo)
conj(A::judiWavefield{vDT}) where vDT =
	judiWavefield{vDT}("conj("*A.name*")",A.m,A.n,A.info,A.dt,A.data)

# transpose(jo)
transpose(A::judiWavefield{vDT}) where vDT =
	judiWavefield{vDT}(""*A.name*".'",A.n,A.m,A.info,A.dt,A.data)

# adjoint(jo)
adjoint(A::judiWavefield{vDT}) where vDT = conj(transpose(A))

####################################################################

function vcat(ai::Vararg{judiWavefield{avDT}, N}) where {avDT, N}
	N == 1 && (return ai[1])
	N > 2 && (return vcat(ai[1], vcat(ai[2:end]...)))
	a, b = ai
	m = a.m + b.m
	n = 1
	nsrc = a.info.nsrc + b.info.nsrc
	data = Array{Array{avDT, 3}, 1}(undef, nsrc)
	nt = Array{Integer}(undef, nsrc)
	for j=1:a.info.nsrc
		data[j] = a.data[j]
		nt[j] = a.info.nt[j]
	end
	for j=a.info.nsrc+1:nsrc
		data[j] = b.data[j-a.info.nsrc]
		nt[j] = b.info.nt[j-a.info.nsrc]
	end
	info = Info(a.info.n, nsrc, nt)
	return judiWavefield{avDT}("judiWavefield",info.n * sum(info.nt), 1, info, a.dt ,data)
end

function push!(a::judiWavefield{T}, b::judiWavefield{T}) where T
	append!(a.data, b.data)
	a.m += b.m
	a.info.nsrc += b.info.nsrc
end

# DFT operator for wavefields, acts along time dimension
function fft_wavefield(x_in,mode)
	nsrc = x_in.info.nsrc
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

# Sampling mask to extract wavefields from full vector
subsample(u::judiWavefield,srcnum) = judiWavefield(u.info,u.data[srcnum];vDT=eltype(u))
similar(u::judiWavefield) = 0f0 * u
similar(u::judiWavefield, vDT::DataType) = judiWavefield{vDT}("judiWavefield",u.m, u.n, u.info, u.dt, Array{vDT}.(0f0.*u.data))

# norm
function norm(a::judiWavefield{avDT}, p::Real=2) where avDT
    if p == Inf
        return max([maximum(abs.(a.data[i])) for i=1:a.info.nsrc]...)
    end
    x = 0.f0
    for j=1:a.info.nsrc
        x += Float32(a.dt) * sum(abs.(vec(a.data[j])).^p)
    end
    return x^(1.f0/p)
end

# inner product
function dot(a::judiWavefield{avDT}, b::judiWavefield{bvDT}) where {avDT, bvDT}
	# Dot product for data containers
	size(a) == size(b) || throw(judiWavefieldException("dimension mismatch"))
	dotprod = 0f0
	for j=1:a.info.nsrc
		dotprod += Float32(a.dt) * dot(vec(a.data[j]),vec(b.data[j]))
	end
	return dotprod
end

# abs
function abs(a::judiWavefield{avDT}) where avDT
	b = deepcopy(a)
	for j=1:a.info.nsrc
		b.data[j] = abs.(a.data[j])
	end
	return b
end

function isapprox(x::judiWavefield, y::judiWavefield; rtol::Real=sqrt(eps()), atol::Real=0)
    x.info.nsrc == y.info.nsrc || throw(judiVectorException("Incompatible number of sources"))
    isapprox(x.data, y.data; rtol=rtol, atol=atol)
end

####################################################################################################

isfinite(x::judiWavefield) = all(all(isfinite.(x.data[i])) for i=1:length(x.data))

