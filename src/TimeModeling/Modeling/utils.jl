function judipmap(func, iter::UnitRange; on_error = nothing)
    # Ignore map and workers if single source
    length(iter) == 1 && (return [func(iter[1])])
    # Switch to asyncmap if serial julia
    length(default_worker_pool()) < 2 && (return asyncmap(func, iter))
    # Standard pmap if parallel and multiple sources
    return pmap(func, iter; on_error=on_error)
end

function judipmap(func, model::Array{Model, 1}, args::Vararg{Array, N}) where N
    # overload for multiple models
    argout = Array{Any, 1}(undef, length(model))
    @sync for (i, ai) in enumerate(zip(model, args...))
        @async argout[i] = func(ai...)
    end
    return argout
end

"""
binary-tree reduction for parallel map. Modified from `DistributedOperations.jl` (Striped from custom types in it)

Copyright (c) 2020: Chevron U.S.A Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
function reduce!(futures::Vector{Future}, op!::Function=sum!)

    # Local reducttion function
    function local_reduce!(my_future, other_future, op!) 
        y = remotecall_fetch(fetch, other_future.where, other_future)
        x = fetch(my_future)
        op!(x, y)
        nothing
    end

    # Get length a next power of two for binary reduction
    M = length(futures)
    L = round(Int,log2(prevpow(2,M)))
    m = 2^L
    # reminder
    R = M - m

    # Reduce the reminder
    reduce_level!(local_reduce!, op!, futures, R)

    # Binary tree reduction
    for l = L:-1:1
        m = 2^(l-1)
        reduce_level!(local_reduce!, op!, futures, m)
    end
    fetch(futures[myid()])
end


function reduce_level!(red_func!::Function, red_op!::Function, futures::Vector{Future}, nleaf::Int=0)
    @sync for i = 1:nleaf
        @async remotecall_fetch(red_func!, futures[i].where, futures[i], futures[m+i], red_op!)
    end
end