# Taper function. Used for data and model muting
_taper(::Val{:reflection}, n::Integer=20) = convert(Vector{Float32}, (cos.(range(pi, stop=2*pi, length=n)) .+ 1) ./ 2)
_taper(::Val{:turning}, n::Integer=20) = convert(Vector{Float32}, (cos.(range(0, stop=pi, length=n)) .+ 1) ./ 2)


# water bottom
"""
    find_water_bottom(v; eps=1e-4)

Fund water bottom based on (x, y) or (x, y, z) input array by finding the first value for each vertical trace that
is not close to the top value (first value such that m[x,y,z] > m[x,y,1] for each x, y)
"""
function find_water_bottom(m::AbstractArray{avDT, N};eps = 1e-4) where {avDT, N}
    #return the indices of the water bottom of a seismic image
    n = size(m)
    idx = zeros(Integer, n[1:end-1])
    wbfunc(x) = abs(x - x[1]) > eps
    @inbounds @simd for i in CartesianIndices(idx)
        idx[i] = findfirst(wbfuncm[i, :])
    end
    return idx
end

find_water_bottom(m::PhysicalParameter;eps=1e-4) = find_water_bottom(m.data; eps=eps)
