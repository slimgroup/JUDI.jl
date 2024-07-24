"""
    single_reduce!(x, y) 

Inplace reduction of y into x.
"""
@inline single_reduce!(x::T, y::T) where T = x .+= y

for T in [judiVector, judiWeights, judiWavefield]
    @eval @inline single_reduce!(x::$T, y::$T) = begin push!(x, y); x end
end

@inline single_reduce!(x::Ref, y::Ref) = begin x[] += y[]; x end

@inline function single_reduce!(x::Tuple, y::Tuple)
    nx = length(x)
    ny = length(y)
    (nx == ny) || throw(ArgumentError("Incompatible lengths ($nx, $ny)"))
    for (xi, yi) in zip(x, y)
        single_reduce!(xi, yi)
    end
    x
end


"""
    local_reduce!(future, other)

Reduce `other` future into local `future`. This is perform remotely on the julia worker
`future.where`.
"""
function local_reduce!(my_future::_TFuture, other_future::_TFuture)
    y = remotecall_fetch(fetch, other_future.where, other_future)
    x = fetch(my_future)
    single_reduce!(x, y)
    y = []
    nothing
end

"""
    reduce_level!(futures, nleaf)

Reduce one level of the reduction tree consisting of `nleaf` futures. Each `leaf`
reduces into itself `futures[i]`.
"""
function reduce_level!(futures::Vector{_TFuture}, nleaf::Int)
    nleaf == 0 && return
    @sync for i = 1:2:2*nleaf
        @async remotecall_wait(local_reduce!, futures[i].where, futures[i], futures[i+1])
    end
    deleteat!(futures, 2:2:2*nleaf)
end

"""
    reduce!(x)
binary-tree reduction of a Vector of Futures `x`

Adapted from `DistributedOperations.jl` (MIT license). Striped from custom types in it and extended to multiple outputs
with different reduction functions.
"""
function reduce!(futures::Vector{_TFuture})
    isnothing(_worker_pool()) && return reduce_all_workers!(futures)
    # Number of parallel workers
    nwork = nworkers(_worker_pool())
    nf = length(futures)
    # Reduction batch. We want to avoid finished task to hang waiting for the
    # binary tree reduction to reach their index holding memory.
    bsize = min(nwork, nf)
    # First batch
    res = reduce_all_workers!(futures[1:bsize])
    # Loop until all reduced
    for i = bsize+1:bsize:nf
        last = min(nf, i + bsize - 1)
        single_reduce!(res, reduce_all_workers!(futures[i:last]))
    end
    return res
end


function reduce_all_workers!(futures::Vector{_TFuture})
    # Get length a next power of two for binary reduction
    M = length(futures)
    L = round(Int, log2(prevpow(2,M)))
    m = 2^L
    # remainder
    R = M - m
    
    # Reduce the remainder
    reduce_level!(futures, R)

    # Binary tree reduction
    for l = L:-1:1
        reduce_level!(futures, 2^(l-1))
    end
    fetch(futures[myid()])
end
