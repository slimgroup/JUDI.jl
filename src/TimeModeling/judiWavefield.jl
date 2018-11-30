############################################################
# judiWavefield ##############################################
############################################################

# Authors: Philipp Witte (pwitte@eos.ubc.ca), Henryk Modzelewski (hmodzelewski@eos.ubc.ca)
# Date: June 2017

export judiWavefield, judiWavefieldException, judiDFTwavefield, muteWavefield, dump_wavefield

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

function judiWavefield(info,dt::Real,data::Union{Array, PyCall.PyObject, String}; vDT::DataType=Float32)

	# length of vector
	m = info.n * sum(info.nt)
	n = 1
	dataCell = Array{Any}(undef, info.nsrc)
	for j=1:info.nsrc
		dataCell[j] = data
	end
	return judiWavefield{vDT}("judiWavefield",m,n,info,dt,dataCell)
end

function judiWavefield(info,dt::Real,data::Array{Any,1};vDT::DataType=Float32)
	# length of vector
	m = info.n * sum(info.nt)
	n = 1
	return judiWavefield{vDT}("judiWavefield",m,n,info,dt,data)
end


####################################################################
## overloaded Base functions

# conj(jo)
conj(A::judiWavefield{vDT}) where vDT =
	judiWavefield{vDT}("conj("*A.name*")",A.m,A.n,A.info,A.dt,A.data)

# transpose(jo)
transpose(A::judiWavefield{vDT}) where vDT =
	judiWavefield{vDT}(""*A.name*".'",A.n,A.m,A.info,A.dt,A.data)

# ctranspose(jo)
ctranspose(A::judiWavefield{vDT}) where vDT =
	judiWavefield{vDT}(""*A.name*"'",A.n,A.m,A.info,A.dt,A.data)

####################################################################

function vcat(a::judiWavefield{avDT},b::judiWavefield{bvDT}) where {avDT, bvDT}
	m = a.m + b.m
	n = 1
	nsrc = a.info.nsrc + b.info.nsrc
	data = Array{Any}(undef, nsrc)
	nt = Array{Any}(undef, nsrc)
	for j=1:a.info.nsrc
		data[j] = a.data[j]
		nt[j] = a.info.nt[j]
	end
	for j=a.info.nsrc+1:nsrc
		data[j] = b.data[j-a.info.nsrc]
		nt[j] = b.info.nt[j-a.info.nsrc]
	end
	info = Info(a.info.n,nsrc,nt)
	return judiWavefield(info,a.dt,data)
end

# add and subtract, mulitply and divide, norms, dot ...


# DFT operator for wavefields, acts along time dimension
function fft_wavefield(x_in,mode)
	nsrc = x_in.info.nsrc
	if mode==1
		x = judiWavefield(x_in.info,deepcopy(x_in.data); vDT=Complex{Float32})
		for i=1:nsrc
			x.data[i] = convert(Array{Complex{Float32}},x.data[i])
			nx = size(x.data[i],2)
			nz = size(x.data[i],3)
			for j=1:nx
				for k=1:nz
					x.data[i][:,j,k] = fft(x.data[i][:,j,k])
				end
			end
		end
	elseif mode==-1
		x = judiWavefield(x_in.info,deepcopy(x_in.data); vDT=Float32)
		for i=1:nsrc
			nx = size(x.data[i],2)
			nz = size(x.data[i],3)
			for j=1:nx
				for k=1:nz
					x.data[i][:,j,k] = real(ifft(x.data[i][:,j,k]))
				end
			end
			x.data[i] = convert(Array{Float32},real(x.data[i]))
		end
	end
	return x
end

# Sampling mask to extract wavefields from full vector
subsample(u::judiWavefield,srcnum) = judiWavefield(u.info,u.data[srcnum];vDT=eltype(u))

function muteWavefield(u_in::judiWavefield,ts_keep)
	u = deepcopy(u_in)
	for j=1:u.info.nsrc
		idx = ones(size(u.data[j],1));
		idx[ts_keep] = 0
		zero_idx = findall(idx)
		u.data[j][zero_idx,:,:] *= 0.f0
	end
	return u
end

# norm
function norm(a::judiWavefield{avDT}, p::Real=2) where avDT
	np = load_numpy()
    x = 0.f0
    for j=1:a.info.nsrc
        if typeof(a.data[j]) == String
            x += a.dt * sum(np[:abs](np[:load](a.data[j])).^p)
        else
            x += a.dt * sum(np[:abs](a.data[j]["data"]).^p)
        end
    end
    return x^(1.f0/p)
end

# Save wavefield to disk
function dump_wavefield(u::PyObject)
    name = join(["wavefield_", randstring(8), ".dat"])
    u["data"][:dump](name)
    return name
end

function dump_wavefield(u::judiWavefield)
    name = Array{Any}(undef, u.info.nsrc)
    for j=1:u.info.nsrc
        name[j] = join(["wavefield_", randstring(8), ".dat"])
        u.data[j]["data"][:dump](name[j])
    end
    return name
end
