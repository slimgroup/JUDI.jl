# Data topmute
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: December 2017

# Updated by Ziyi Yin, November 2020, ziyi.yin@gatech.edu
#
export marineTopmute2D, judiMarineTopmute2D
export model_topmute, judiTopmute, find_water_bottom, depth_scaling, judiDepthScaling, low_filter, judiFilter

############################################ Data space preconditioners ################################################

function judiFilter(geometry, fmin, fmax)
    N = n_samples(geometry)
    D = joLinearFunctionFwd_T(N, N,
                              v -> low_filter(v,geometry.dt[1];fmin=fmin, fmax=fmax),
                              w -> low_filter(w,geometry.dt[1];fmin=fmin, fmax=fmax),
                              Float32,Float32,name="Data filter")
    return D
end

function low_filter(Din::Array{Float32, 1}, dt_in; fmin=0.0, fmax=25.0)	
    Dout = deepcopy(Din)	
    responsetype = Bandpass(fmin, fmax; fs=1e3/dt_in)	
    designmethod = Butterworth(5)	
    return filt(digitalfilter(responsetype, designmethod), Float32.(Dout))	
end	

function low_filter(Din::Array{Float32, 2}, dt_in; fmin=0.0, fmax=25.0)	
    Dout = deepcopy(Din)
    responsetype = Bandpass(fmin, fmax; fs=1e3/dt_in)	
    designmethod = Butterworth(5)	
    for i=1:size(Dout,2)	
        Dout[:, i] = filt(digitalfilter(responsetype, designmethod), Float32.(Dout[:, i]))	
    end	
    return Dout	
end

function low_filter(Din::judiVector, dt_in; fmin=0.0, fmax=25.0)	
    Dout = deepcopy(Din)	
    for j=1:Dout.nsrc
		if size(Din.data[j], 2) == 1
			Dout.data[j] = low_filter(Dout.data[j], dt_in; fmin=fmin, fmax=fmax)
		else
	        for i=1:size(Din.data[j], 2)	
	            Dout.data[j][:, i] = low_filter(Dout.data[j][:, i], dt_in; fmin=fmin, fmax=fmax)
	        end	
		end
    end
    return Dout	
end

# This brrute force and should be improved into a lazy evaluation that only does the mute at read time.
marineTopmute2D(Dobs::judiVector{T, SeisCon}, muteStart::Integer; mute=Array{Any}(undef, 3), flipmask=false) where T<:Number =
    marineTopmute2D(get_data(Dobs), muteStart; mute=mute, flipmask=flipmask)

function marineTopmute2D(Dobs::judiVector{T, Matrix{T}}, muteStart::Integer; mute=Array{Any}(undef, 3), flipmask=false) where T<:Number
    # Data topmute for end-on spread marine streamer data
    Din = deepcopy(Dobs)

    for j=1:Din.nsrc

        # Design mute window
        x0 = 1f0
        xend = length(Din[j].geometry.xloc[1])
        nt = Din[j].geometry.nt[1]
        nrec = length(Din[j].geometry.xloc[1])
        drec = abs(Din[j].geometry.xloc[1][1] - Din[j].geometry.xloc[1][2])
        offsetDirectWave = 1.5f0*Din[j].geometry.t[1]
        idxOffset = Int(round(offsetDirectWave/drec))
        dx = round(idxOffset - idxOffset/10f0)

        if j==1 && ~isassigned(mute)
            z0 = muteStart - Int(round(muteStart/10))
            slope = 1.05f0*(nt - z0)/dx
            mute[1] = x0
            mute[2] = z0
            mute[3] = slope
        else#if j==1 && isassigned(mute)
            x0 = Int64(mute[1])
            z0 = Int64(mute[2])
            slope = mute[3]
        end

        mask = ones(Float32,nt,nrec)
        mask[1:z0,:] .= 0f0

        # Linear mute
        if (nrec-x0 < dx)
            x = nrec
            zIntercept = Int(round(z0+slope*(x-x0)))
            zax = z0+1:1:zIntercept
        else
            x = x0+dx
            zax = z0+1:1:nt
        end
        if length(zax) > 1
            xax = Array{Int}(round.(range(x0,stop=x,length=length(zax))))
        else
            xax = Int(round(x0))
        end
        for k=1:length(zax)
            mask[min(zax[k], nt),xax[k]:end] .= 0f0
        end
        flipmask == true && (mask = reverse(mask, dims=2))
        Din.data[j] = Din.data[j].*mask
    end
    return Din
end

function judiMarineTopmute2D(muteStart,geometry;params=Array{Any}(undef, 3),flipmask=false)
# JOLI wrapper for the linear depth scaling function
    N = n_samples(geometry)
    D = joLinearFunctionFwd_T(N,N,
                             v -> marineTopmute2D(v,muteStart;mute=params, flipmask=flipmask),
                             w -> marineTopmute2D(w,muteStart;mute=params, flipmask=flipmask),
                             Float32,Float32,name="Data mute")
    return D
end


############################################ Model space preconditioners ###############################################


function model_topmute(n::Tuple{Int64,Int64}, mute_end::Array{Integer,1}, length::Int64, x_orig)
# Model domain topmute for a velocity model of dimensions n = [nx, nz].
    x = deepcopy(reshape(x_orig,n))
    for j=1:n[1]
        mute_start = mute_end[j] - length
        filter = zeros(Float32, n[2])
        filter[1:mute_start-1] .= 0f0
        filter[mute_end[j]+1:end] .= 1f0
        taper_length = mute_end[j] - mute_start + 1
        taper = (1f0 .+ sin.((Float32(pi)*(0:taper_length-1))/(taper_length - 1) .- Float32(pi)/2f0))/2f0
        filter[mute_start:mute_end[j]] = taper
        global x[j,:] = x[j,:].*filter
    end
    return vec(x)
end

function model_topmute(n::Tuple{Int64,Int64}, mute_end::Int64, length::Int64, x_orig)
# Model domain topmute for a velocity model of dimensions n = [nx, nz].
    x = deepcopy(reshape(x_orig,n))
    mute_start = mute_end - length
    filter = zeros(Float32, n[2])
    filter[1:mute_start-1] .= 0f0
    filter[mute_end+1:end] .= 1f0
    taper_length = mute_end - mute_start + 1
    taper = (1f0 .+ sin.((Float32(pi)*(0:taper_length-1))/(taper_length - 1) .- Float32(pi)/2f0))/2f0
    filter[mute_start:mute_end] = taper
    for j=1:n[1]
        global x[j,:] = x[j,:].*filter
    end
    return vec(x)
end

model_topmute(n::Tuple{Int64,Int64}, mute_end::Array{Float32, 2}, length, x) = vec(mute_end) .* vec(x)

model_topmute(n::Tuple{Int64,Int64,Int64}, mute_end::Int64, length::Int64, x_orig) = model_topmute((n[1]*n[2], n[3]), mute_end, length, x_orig)

model_topmute(n::Tuple{Int64,Int64,Int64}, mute_end::Array{Integer,2}, length::Int64, x_orig) = model_topmute((n[1]*n[2], n[3]), vec(mute_end), length, x_orig)
    
model_topmute(n::Tuple{Int64,Int64,Int64}, mute_end::Array{Float32, 3}, length, x) = vec(mute_end) .* vec(x)

function judiTopmute(n, mute_end, length)
    # JOLI wrapper for model domain topmute
    N = prod(n)
    T = joLinearFunctionFwd_T(N,N,
                             v -> model_topmute(n, mute_end, length, v),
                             w -> model_topmute(n, mute_end, length, w),
                             Float32,Float32,name="Model topmute")
    return T
end

function find_water_bottom(m::AbstractArray{avDT,2};eps = 1e-4) where {avDT}
    #return the indices of the water bottom of a seismic image
    n = size(m)
    idx = zeros(Integer, n[1])
    for j=1:n[1]
        k=1
        while true
            if abs(m[j,k]) > eps
                idx[j] = k
                break
            end
            k += 1
        end
    end
    return idx
end

find_water_bottom(m::AbstractArray{avDT,3};eps = 1e-4) where {avDT} = reshape(find_water_bottom(reshape(m, :, size(m)[end]);eps=eps), size(m,1), size(m,2))
find_water_bottom(m::PhysicalParameter;eps = 1e-4) = find_water_bottom(m.data)

function depth_scaling(m, model)
# Linear depth scaling function for seismic images
    m_out = deepcopy(reshape(m,Int(prod(model.n)/model.n[end]),model.n[end]))
    filter = sqrt.(0f0:Float32(model.d[end]):Float32(model.d[end])*(model.n[end]-1))
    return vec(m_out.*filter')
end

function judiDepthScaling(model)
# JOLI wrapper for the linear depth scaling function
    N = prod(model.n)
    return joLinearFunctionFwd_T(N, N,
                    v -> depth_scaling(v,model),
                    w -> depth_scaling(w,model),
                    Float32,Float32,name="Depth scaling")
end
