export AbstractDim, AbstractSize, make_id, jAdjoint


struct AbstractDim <: Integer
    name::Symbol
end

struct AbstractSize{N}
    dims::NTuple{N, AbstractDim}
end


const _time_space = (AbstractDim(:time), AbstractDim(:n_x), AbstractDim(:n_y), AbstractDim(:n_z))
const _rec_space = AbstractSize((AbstractDim(:time), AbstractDim(:n_rec)))
time_space_size(N::Integer) = AbstractSize(_time_space[1:N+1])



const _used_id = []

function make_id()::Symbol
    new_id = Symbol(randstring(10))
    (new_id ∈ _used_id) && (return make_id())
    push!(_used_id, new_id)
    return new_id
end


struct jAdjoint{T}
    op::T
end