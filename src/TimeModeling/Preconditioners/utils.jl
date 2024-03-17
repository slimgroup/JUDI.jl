export find_water_bottom

# Taper function. Used for data and model muting
_taper(::Val{:reflection}, n::Integer=20) = convert(Vector{Float32}, (cos.(range(pi, stop=2*pi, length=n)) .+ 1) ./ 2)
_taper(::Val{:turning}, n::Integer=20) = convert(Vector{Float32}, (cos.(range(0, stop=pi, length=n)) .+ 1) ./ 2)

# Muting utils
_yloc(y::Vector{T}, t::Integer) where T = length(y) > 1 ?  y[t] : y[1]

radius(G1::Geometry, G2::Geometry, t::Integer) = sqrt.((G1.xloc[1][t] .- G2.xloc[1][1]).^2 .+ (_yloc(G1.yloc[1], t) .- G2.yloc[1][1]).^2 .+ (G1.zloc[1][t] .- G2.zloc[1][1]).^2)

_tapew(i::Integer, taperwidth::Integer, ::Integer, ::Val{:reflection}) = i < taperwidth
_tapew(i::Integer, taperwidth::Integer, nt::Integer, ::Val{:turning}) = i > (nt - taperwidth)

_mutew!(t::AbstractVector{T}, taper::AbstractVector{T}, i::Integer, ::Integer, ::Val{:reflection}) where T = broadcast!(*, t[1:i], t[1:i], taper[end-i+1:end])
_mutew!(t::AbstractVector{T}, taper::AbstractVector{T}, i::Integer, nt::Integer, ::Val{:turning}) where T = broadcast!(*, t[i:nt], t[i:nt], taper[1:(nt-i+1)])

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
    wbfunc(x, x1) = abs(x - x1) > eps
    for i in CartesianIndices(idx)
        idx[i] = findfirst(x->wbfunc(x, m[i, 1]), m[i, :])
    end
    return idx
end

function find_water_bottom(m::AbstractArray{avDT, N}, wbval::Number; inv=true) where {avDT, N}
    #return the indices of the water bottom of a seismic image
    n = size(m)
    idx = zeros(Integer, n[1:end-1])
    wbfunc(x) = inv ? x < wbval : x > wbval

    for i in CartesianIndices(idx)
        idx[i] = findfirst(wbfunc, m[i, :])
    end
    return idx
end


find_water_bottom(m::PhysicalParameter, wbval::Number; inv=true) = find_water_bottom(m.data, wbval; inv=inv)
find_water_bottom(m::PhysicalParameter;eps=1e-4) = find_water_bottom(m.data; eps=eps)
