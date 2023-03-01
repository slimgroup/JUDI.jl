export AbstractSize, time_space, time_space_src, time_src, rec_space, space, space_src

struct AbstractSize
    dims::Dict
end

AbstractSize(n::NTuple{N, Symbol}, sizes) where N = AbstractSize(Dict(zip(n, sizes)))

# Base
function getindex(S::AbstractSize, sinds::Union{AbstractRange, Vector{<:Integer}})
    k = keys(S.dims)
    dim_map = copy(S.dims)
    for (k, v) in S.dims
        new_v = typeof(v) <: Integer ? v : v[sinds]
        dim_map[k] = new_v
    end
    :src ∈ keys(dim_map) && (dim_map[:src] = length(sinds))
    AbstractSize(dim_map)
end
getindex(S::AbstractSize, i::Integer) = getindex(S, i:i)

getindex(S::AbstractSize, I...) = AbstractSize(keys(S.dims[I...]), values(S.dims[I...]))
getindex(S::AbstractSize, I::Symbol) = S.dims[I]
setindex!(S::AbstractSize, v, I::Symbol) = setindex!(S.dims, v, I)
iterate(S::AbstractSize) = iterate(S.dims)
iterate(S::AbstractSize, state) = iterate(S.dims, state)

Base.isless(i::Int64, a::AbstractSize) = isless(i, nsamples(a))

convert(::Type{T}, S::AbstractSize) where T<:Number = convert(T, nsamples(S))
(::Type{T})(S::AbstractSize) where T<:Union{Integer, AbstractFloat} = convert(T, nsamples(S))
(c::Colon)(i::T, S::AbstractSize) where T = c(i, T(S))

Base.keys(S::AbstractSize) = keys(S.dims)
Base.values(S::AbstractSize) = values(S.dims)
Base.merge!(S1::AbstractSize, S2::AbstractSize) = merge!(S1.dims, S2.dims)

==(S1::AbstractSize, S2::AbstractSize) = S1.dims == S2.dims
==(S1::Integer, S2::AbstractSize) = nsamples(S2) == S1
==(S1::AbstractSize, S2::Integer) = nsamples(S1) == S2

Base.repr(S::AbstractSize) = "($(join(keys(S.dims), " * ")))"

similar(x::judiMultiSourceVector, ::Type{ET}, dims::AbstractSize) where ET = similar(x, ET)
similar(x::AbstractVector, ::Type{ET}, dims::AbstractSize) where ET = similar(x, ET, Int(dims))

# Update size
set_space_size!(S::AbstractSize, vals::NTuple{2, Integer}) = begin pop!(S.dims, :y, 0); merge!(S.dims, Dict(zip((:x, :z), vals))) end
set_space_size!(S::AbstractSize, vals::NTuple{3, Integer}) = merge!(S.dims, Dict(zip((:x, :y, :z), vals)))

# Actual integer size
nsamples(dims::Dict{Symbol, Any}) = sum(.*(values(dims)...)) ÷ get(dims, :src, 1)
nsamples(dims::Dict{N, <:Integer}) where N = prod(values(dims))
nsamples(D::AbstractSize) = nsamples(D.dims)

# Constructors
space(N::NTuple{2, Integer}) = AbstractSize((:x, :z), (N[1], N[2]))
space(N::NTuple{3, Integer}) = AbstractSize((:x, :y, :z), (N[1], N[2], N[3]))

time_space(N::NTuple{2, Integer}) = AbstractSize((:time, :x, :z), ([0], N[1], N[2]))
time_space(N::NTuple{3, Integer}) = AbstractSize((:time, :x, :y, :z), ([0], N[1], N[2], N[3]))

time_space_src(nsrc::Integer, nt, N::Integer) = AbstractSize((:src, :time, :x, :y, :z), (nsrc, nt, zeros(Int, N)...))
time_space_src(nsrc::Integer, nt, N::NTuple{2, Integer}) = AbstractSize((:src, :time, :x, :z), (nsrc, nt, N...))
time_space_src(nsrc::Integer, nt, N::NTuple{3, Integer}) = AbstractSize((:src, :time, :x, :y, :z), (nsrc, nt, N...))
time_space_src(nsrc::Integer, nt) = AbstractSize((:src, :time, :x, :y, :z), (nsrc, nt, 0, 0, 0))

space_src(nsrc::Integer) = AbstractSize((:src, :x, :y, :z), (nsrc, 0, 0, 0))

time_src(nsrc::Integer, nt) = AbstractSize((:src, :time), (nsrc, nt))

rec_space(G::Geometry) = AbstractSize((:src, :time, :rec), (get_nsrc(G), G.nt, G.nrec))
