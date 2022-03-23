"""
    single_reduce!(x, y) 

Inplace reduction of y into x.
"""
@inline single_reduce!(x, y) = begin x .+= y; x end

for T in [judiVector, judiWeights, judiWavefield]
    @eval @inline single_reduce!(x::$T, y) = begin push!(x, y); x end
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
    safe_gc()

Generic GC, compatible with different julia versions of it.
"""
safe_gc() = try Base.GC.gc(); catch; gc() end

"""
    local_reduce!(future, other)

Reduce `other` future into local `future`. This is perform remotely on the julia worker
`future.where`.
"""
function local_reduce!(my_future::_TFuture, other_future::_TFuture)
    y = fetch(other_future)
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

Modified from `DistributedOperations.jl` (Striped from custom types in it) and extended to multiple outputs
with different reduction functions.

Copyright (c) 2020: Chevron U.S.A Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
function reduce!(futures::Vector{_TFuture})
    # Get length a next power of two for binary reduction
    M = length(futures)
    L = round(Int,log2(prevpow(2,M)))
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
