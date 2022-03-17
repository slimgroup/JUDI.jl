export AbstractDim, AbstractSize, make_id, jAdjoint, time_space_size, space_size

struct AbstractDim <: Integer
    name::Symbol
end

struct AbstractSize{N}
    dims::NTuple{N, AbstractDim}
end

getindex(S::AbstractSize, I...) = AbstractSize(S.dims[I...])
==(S1::AbstractSize, S2::AbstractSize) = S1.dims == S2.dims
==(d1::AbstractDim, d2::AbstractDim) = d1.name == d2.name

Base.repr(D::AbstractDim) = D.name
Base.repr(D::AbstractSize) = "($(join(repr.(D.dims), " x ")))"

const _time_space = AbstractSize((AbstractDim(:s), AbstractDim(:time), AbstractDim(:n_x), AbstractDim(:n_y), AbstractDim(:n_z)))
const _space = AbstractSize((AbstractDim(:n_x), AbstractDim(:n_y), AbstractDim(:n_z)))
const _rec_space = AbstractSize((AbstractDim(:s), AbstractDim(:time), AbstractDim(:r)))

time_space_size(N::Integer) = _time_space[1:N+2]
space_size(N::Integer) = _space[1:N]

const _used_id = []

function make_id()::Symbol
    new_id = Symbol(randstring(10))
    (new_id ∈ _used_id) && (return make_id())
    push!(_used_id, new_id)
    return new_id
end