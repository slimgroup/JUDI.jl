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
