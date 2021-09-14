using Distributed, LinearAlgebra
@everywhere using JUDI

@everywhere @inline sum!(x, y) = begin x .+= y; nothing end

@everywhere function reduce_level!(red_func!::Function, red_op!::Function, futures::Vector{Future}, nleaf::Int=0)
    nleaf == 0 && return
    @sync for i = 1:nleaf
        @async remotecall_fetch(red_func!, futures[i].where, futures[i], futures[nleaf+i], red_op!)
    end
end

    # Local reducttion function
@everywhere function local_reduce!(my_future::Future, other_future::Future, op!::Function) 
    y = remotecall_fetch(fetch, other_future.where, other_future)
    x = fetch(my_future)
    op!(x, y)
    nothing
end

function reduce!(futures::Vector{Future}, op!::Function=sum!)

    # Get length a next power of two for binary reduction
    M = length(futures)
    L = round(Int,log2(prevpow(2,M)))
    m = 2^L
    # reminder
    R = M - m

    # Reduce the reminder
    reduce_level!(local_reduce!, op!, futures, R)

    # Now do binary reduction on power of two tree
    for l = L:-1:1
        m = 2^(l-1)
        reduce_level!(local_reduce!, op!, futures, m)
    end
    fetch(futures[myid()])
end


@everywhere myfunc(i) = begin sleep(1); return PhysicalParameter(i*ones(1000, 100), (10, 10), (0, 0)) end

p = length(default_worker_pool())
results = Vector{Future}(undef, p)

@sync for pi=1:p
    @async results[pi] = @spawn myfunc(pi)
end
res = reduce!(results)
res2 = reduce(+, pmap(myfunc, 1:p))

# Time it
@time begin
    @sync for pi=1:p
        @async results[pi] = @spawn myfunc(pi)
    end
    res = reduce!(results)
end

@time res2 =  reduce(+, pmap(myfunc, 1:p))

@show typeof(res), typeof(res2), size(res), size(res2)
@show norm(res - res2)