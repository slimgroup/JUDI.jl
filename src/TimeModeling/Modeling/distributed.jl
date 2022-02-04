_parallel_functions = [:time_modeling, :fwi_objective, :twri_objective, :extended_source_modeling, :lsrtm_objective]
# Shot and experiment parallelism
for func in _parallel_functions
    @eval $func(args...; kwargs...) = task_distributed($func, _worker_pool(), args...; kwargs...)
end

# Find number of experiments
"""
    _get_nexp(x)

Get number of experiments given a JUDI type. By default we have only one experiment unless we input
a Vector of judiType such as [model, model] to compute gradient for different cases at once.
"""
_get_nexp(x) = 1
for T in [judiVector, Model, judiWeights, judiWavefield, PhysicalParameter, Vector{Float32}]
    @eval _get_nexp(v::Vector{<:$T}) = length(v)
end   

"""
    get_nexp(args...)

Get number of experiments given a list of arguments. By default we have only one experiment unless we input
a Vector of judiType such as [model, model] to compute gradient for different cases at once.
All arguments can only be either a Vector of size `n` or a basic type. Basic type are conisdered common 
to all `n` experiments. For Example

    VALID: get_nexp(([model1, model2], dobs)...) = 2 (compute a function for common data and two different models)
    INVALID:  get_nexp(([model1, model2], [dobs, dobs, dobs])...) 
"""
function get_nexp(args...)
    n_exp = setdiff([_get_nexp(a) for a ∈ args], 1)
    length(n_exp) < 2 || throw(ArgumentError("Incompatible number of experiments in arguments"))
    return max(n_exp..., 1)
end

# Filter arguments for given task
"""
    _get_exp(x, i)

Filter input `x`` for experiment number `i`. Returns `x` is a constant not depending on experiment.
"""
_get_exp(x, i) = x
for T in [judiVector, Model, judiWeights, judiWavefield, Vector{Float32}, PhysicalParameter]
    @eval _get_exp(v::Vector{<:$T}, i) = v[i]
end

"""
    get_exp(i, args...; kwargs...)

Filter input arguments and keyword arguments for experiment number `i`.
"""
get_exp(i, args...; kwargs...) = (_get_exp(a, i) for a in (args..., kwargs.data...))

# Find task iterator (number of sources and indices)
"""
    _get_iter(x)

Get source number iterator from `x`
"""
_get_iter(x::UnitRange) = x
_get_iter(x::judiVector) = 1:x.nsrc
_get_iter(x) = nothing

"""
    get_iter(args...)

Get source number iterator input argument list (number of sources in the data or provided UnitRange)
"""
function get_iter(args...)
    iter = try
        filter(x -> !isnothing(x), [_get_iter(a) for a ∈ args])[1]
    catch BoundsError
        throw(ArgumentError("No iterator found in argument list"))
    end
    return iter, filter(x -> x != iter, args)
end


###
subsample(x, i) = x

"""
    submit_task!(rfunc, pool, i, args...) 

Submits a new task running the function `rfunc` on an available worker in the `pool`. The function `rfunc` taskes
for input each element of `args` at index `i`.
"""
submit_task!(rfunc, pool, i, args...) = remotecall(rfunc, pool, (subsample(a, i) for a ∈ args)...)

"""
    filter_exp(x, ::Val)

Filter output based on the number of experiments. Returns `x[1]` for a single experiment or 
(sum(x[i][1] for i in 1:length(x)), [x[i][2] for i in 1:length(x)]) (sum of objective values and list of gradient)
for multiple experiments.
"""
filter_exp(x, ::Val{1}) = x[1]
filter_exp(x, ::Val) = gather_nexp(x)

gather_nexp(x) = (sum(x[i][1] for i in 1:length(x)), [x[i][k] for i in 1:length(x)] for k in 2:length(x[1]))

"""
    as_vec(x, ::Val{Bool})

Vectorizes output when `return_array` is set to `true`.
"""
as_vec(x, ::Val) = x
as_vec(x::Tuple, v::Val) = tuple((as_vec(xi, v) for xi in x)...)
as_vec(x::Ref, ::Val) = x[]
as_vec(x::PhysicalParameter, ::Val{true}) = vec(x.data)
for T in [judiVector, judiWeights, judiWavefield]
    @eval as_vec(x::$T, ::Val{true}) = vcat([vec(x.data[i]) for i=1:length(x.data)]...)
end
####

Base.retry(f, pool; kwargs...) = retry(f; kwargs...)

"""
    task_distributed(func, args...; kwargs...)

Distribute the function `func` over the JUDI pool and reduce the result.
"""
function task_distributed(func, pool, args...; kwargs...)
    # Get number of experiments
    nexp = get_nexp(args...)
    judilog("Running $(func) for $(nexp) experiments")
    # Make it retry if fails
    rfunc = retry(func, pool, delays=ExponentialBackOff(n=3))
    # Allocate results
    out = Vector{Any}(undef, nexp)
    # Submit all tasks, asynchronously
    @sync begin
        for e ∈ 1:nexp
            # get local args
            args_e = get_exp(e, args...; kwargs...)
            iter, args_e = get_iter(args_e...)
            # Get options
            opt = findfirst(x -> typeof(x) <: Options, args_e)
            opt = args_e[opt]
            # Allocate Future results
            res_e = Vector{_TFuture}(undef, length(iter))
            judilog("Running experiment $e out of $(nexp) with $(length(iter)) sources")
            flush(stdout)
            for i ∈ iter
                res_e[i] = submit_task!(rfunc, pool, i, args_e...)
            end
            @async out[e] = as_vec(reduce!(res_e), Val(opt.return_array))
        end
    end
 
    out = process_illum(out, get_models(args...), Val(nexp))
    out = filter_exp(out, Val(nexp))
    return out
end


"""
    single_reduce!(x, y) 

Inplace reduction of y into x.
"""
@inline single_reduce!(x, y) = begin x .+= y; nothing end
@inline single_reduce!(x::Illum, y::Illum) = single_reduce!(x.p, y.p)

for T in [judiVector, judiWeights, judiWavefield]
    @eval @inline single_reduce!(x::$T, y) = begin push!(x, y); nothing end
end

@inline single_reduce!(x::Ref, y::Ref) where {T<:Number} = begin x[] += y[]; nothing end

@inline function single_reduce!(x::Tuple, y::Tuple)
    nx = length(x)
    ny = length(y)
    (nx == ny) || throw(ArgumentError("Incompatible lengths ($nx, $ny)"))
    for (xi, yi) in zip(x, y)
        single_reduce!(xi, yi)
    end
    nothing
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
    y = remotecall_fetch(fetch, other_future.where, other_future)
    x = fetch(my_future)
    single_reduce!(x, y)
    # Force GC since julia parallel still has issues with it
    y=1;safe_gc(); remotecall(safe_gc, other_future.where);
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
        @async remotecall_fetch(local_reduce!, futures[i].where, futures[i], futures[i+1])
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
