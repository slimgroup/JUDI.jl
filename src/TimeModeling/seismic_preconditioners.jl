# Data topmute
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: December 2017
#

export marineTopmute2D, judiMarineTopmute2D
export model_topmute, judiTopmute, find_water_bottom, depth_scaling, judiDepthScaling, laplace


############################################ Data space preconditioners ################################################


function marineTopmute2D(Dobs::judiVector, muteStart::Integer; mute=Array{Any}(undef, 3), flipmask=false)
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
            x0 = mute[1]
            z0 = mute[2]
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
            mask[zax[k],xax[k]:end] .= 0f0
        end
        flipmask == true && (mask = reverse(mask, dims=2))
        Din.data[j] = Din.data[j].*mask
    end
    return Din
end

function judiMarineTopmute2D(muteStart,geometry;params=Array{Any}(undef, 3),flipmask=false)
# JOLI wrapper for the linear depth scaling function
    nsrc = length(geometry.xloc)
    N = 0
    for j=1:nsrc
        N += geometry.nt[j]*length(geometry.xloc[j])
    end
    D = joLinearFunctionFwd_T(N,N,
                             v -> marineTopmute2D(v,muteStart;mute=params, flipmask=flipmask),
                             w -> marineTopmute2D(w,muteStart;mute=params, flipmask=flipmask),
                             Float32,Float32,name="Data mute")
    return D
end


############################################ Model space preconditioners ###############################################


function model_topmute(n::Tuple{Int64,Int64}, mute_end::Array{Integer,1}, length::Int64, x_orig)
# Model domain topmute for a velocity model of dimensions n = [nx, nz].
    x = copy(reshape(x_orig,n))
    for j=1:n[1]
        mute_start = mute_end[j] - length
        filter = zeros(Float32, n[2])
        filter[1:mute_start-1] .= 0f0
        filter[mute_end[j]+1:end] .= 1f0
        taper_length = mute_end[j] - mute_start + 1
        taper = (1f0 .+ sin.((Float32(pi)*(0:taper_length-1))/(taper_length - 1) .- Float32(pi)/2f0))/2f0
        filter[mute_start:mute_end[j]] = taper
        M = diagm(0=>filter)
        global x[j,:] = x[j,:].*filter
    end
    return vec(x)
end

function model_topmute(n::Tuple{Int64,Int64}, mute_end::Int64, length::Int64, x_orig)
# Model domain topmute for a velocity model of dimensions n = [nx, nz].
    x = copy(reshape(x_orig,n))
    mute_start = mute_end - length
    filter = zeros(Float32, n[2])
    filter[1:mute_start-1] .= 0f0
    filter[mute_end+1:end] .= 1f0
    taper_length = mute_end - mute_start + 1
    taper = (1f0 .+ sin.((Float32(pi)*(0:taper_length-1))/(taper_length - 1) .- Float32(pi)/2f0))/2f0
    filter[mute_start:mute_end] = taper
    M = diagm(0=>filter)
    for j=1:n[1]
        global x[j,:] = x[j,:].*filter
    end
    return vec(x)
end

model_topmute(n::Tuple{Int64,Int64}, mute_end::Array{Float32, 2}, length, x) = vec(mute_end) .* vec(x)

function judiTopmute(n, mute_start, length)
    # JOLI wrapper for model domain topmute
    N = prod(n)
    T = joLinearFunctionFwd_T(N,N,
                             v -> model_topmute(n, mute_start, length, v),
                             w -> model_topmute(n, mute_start, length, w),
                             Float32,Float32,name="Model topmute")
    return T
end

function find_water_bottom(m)
    #return the indices of the water bottom of a seismic image
    n = size(m)
    idx = zeros(Integer,n[1])
    eps = 1e-4
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

function depth_scaling(m,model)
# Linear depth scaling function for seismic images
    m = reshape(m,model.n)
    filter = sqrt.(0f0:model.d[2]:model.d[2]*(model.n[2]-1))
    F = diagm(0=>filter)
    for j=1:model.n[1]
        m[j,:] = F*m[j,:]
    end
    return vec(m)
end

function judiDepthScaling(model)
# JOLI wrapper for the linear depth scaling function
    N = prod(model.n)
    D = joLinearFunctionFwd_T(N, N,
                             v -> depth_scaling(v,model),
                             w -> depth_scaling(w,model),
                             Float32,Float32,name="Depth scaling")
end

# function laplace(model::TimeModeling.Model)
# # 2D Laplace operator
#
#     # 2nd derivative in x direction
#     d1 = ones(Float32,model.n[1]-1)*1f0/(model.d[1]^2)
#     d2 = ones(Float32,model.n[1])*-2f0/(model.d[1]^2)
#     d3 = d1
#     d = (d1, d2, d3)
#     position = (-1, 0, 1)
#
#     Dx = spdiagm(d,position,model.n[1],model.n[1])
#     Ix = speye(Float32,model.n[1])
#
#     # 2nd derivative in z direction
#     d1 = ones(Float32,model.n[2]-1)*1f0/(model.d[2]^2)
#     d2 = ones(Float32,model.n[2])*-2f0/(model.d[2]^2)
#     d3 = d1
#     d = (d1, d2, d3)
#     position = (-1, 0, 1)
#
#     Dz = spdiagm(d,position,model.n[2],model.n[2])
#     Iz = speye(Float32,model.n[2])
#
#     # 2D Laplace operator
#     D = kron(Dz,Ix) + kron(Iz,Dx)
# end
