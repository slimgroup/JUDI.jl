# Get number of source from list of args
function get_nsrc(args::Vararg{Any, N}) where N
    nsrcs = 0
    for a ∈ args
        try
            nsrc = a.nsrc
        catch e
            nsrc = 0
        end
        (nsrc == nsrcs || nsrcs == 0) || throw(ArgumentError("Incompatible number of sources ($nsrc, $nsrcs)"))
    end
end

#### Parallelization

function judipmap(func, iter, reduceop!::Vararg{Function, N}; kw...) where N
    length(iter) == 1 && (return func(iter[1]))
    # worker pool
    p = default_worker_pool()
    # Make it retyr if fails
    rfunc = retry(func, delays=ExponentialBackOff(n=3))
    # Future results
    results = Array{Future}(undef, length(iter))
    # Submit all tasks, asynchronously
    @sync begin
        for i ∈ iter
            @async results[i] = remotecall(rfunc, p, i)
        end
    end
    # Reduce futures
    out = reduce!(results, reduceop!...)
    return out
end

function judipmap(func, models::Array{Model, 1}, args::Vararg{Array, N}) where N
    # Future results
    nmodel = length(models)
    argout = Array{Any}(undef, nmodel)
    @sync begin
        for (mi, ai) in enumerate(zip(models, args...))
            @async argout[mi] = func(ai...)
        end
    end
    return argout
end

"""
binary-tree reduction for parallel map.
Modified from `DistributedOperations.jl` (Striped from custom types in it) and extended to multiple outputs
with different reduction functions.

Copyright (c) 2020: Chevron U.S.A Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

@inline sum!(x::Tuple, y::Tuple) = begin for i=1:length(x) x[i] .+= y[i] end; nothing end
@inline sum!(x, y) = begin x .+= y; nothing end
@inline vcat!(x, y) = begin push!(x, y); nothing end

@inline apply_op!(x, y, op!::Function) = op!(x, y)

@inline function apply_op!(x::Tuple, y::Tuple, op!::Vararg{Function, N}) where N
    N == 1 && (op![1](x, y); return)
    nx = length(x)
    ny = length(y)
    (nx == ny) || throw(ArgumentError("Incompatible lengths ($nx, $ny)"))
    for (opi!, xi, yi) in zip(op!, x, y)
        opi!(xi, yi)
    end
    nothing
end

# Local reducttion function
function local_reduce!(my_future::Future, other_future::Future,
                       op!::Vararg{Function, N}) where N
    y = remotecall_fetch(fetch, other_future.where, other_future)
    x = fetch(my_future)
    apply_op!(x, y, op!...)
    nothing
end

function reduce_level!(red_func!::Function, futures::Vector{Future},
                       nleaf::Int, red_op!::Vararg{Function, N}) where N
    nleaf == 0 && return
    @sync for i = 1:nleaf
        @async remotecall_fetch(red_func!, futures[i].where, futures[i], futures[nleaf+i], red_op!...)
    end
end

function reduce!(futures::Vector{Future}, op!::Vararg{Function, N}) where N
    # Get length a next power of two for binary reduction
    M = length(futures)
    L = round(Int,log2(prevpow(2,M)))
    m = 2^L
    # reminder
    R = M - m

    # Reduce the reminder
    reduce_level!(local_reduce!, futures, R, op!...)

    # Binary tree reduction
    for l = L:-1:1
        m = 2^(l-1)
        reduce_level!(local_reduce!, futures, m, op!...)
    end
    fetch(futures[myid()])
end
