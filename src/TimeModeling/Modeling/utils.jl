

function judipmap(func, iter::UnitRange; )
    # Ignore map and workers if single source
    length(iter) == 1 && (return [func(iter[1])])
    # Switch to asyncmap if serial julia
    length(default_worker_pool()) < 2 && (return asyncmap(func, iter))
    # Standard pmap if parallel and multiple sources
    return pmap(func, iter)
end
