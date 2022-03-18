"""
    propagate(F::judiPropagator{T, mode}, q)

Base propagation interfaces that calls the devito `mode` propagator (forward/adjoint/..)
with `q` as a source. The return type is infered from `F`.
"""
function propagate(F::judiPropagator{T, O}, q::AbstractVector{T}) where {T, O}
    srcGeometry, srcData, recGeometry, recData, dm = make_input(F, q)
    return time_modeling(F.model, srcGeometry, srcData, recGeometry, recData, dm, O, F.options)
end

function propagate(F::judiPropagator{T, :adjoint}, q::AbstractVector{T}) where {T}
    srcGeometry, srcData, recGeometry, recData, dm = make_input(F, q)
    return time_modeling(F.model, recGeometry, recData, srcGeometry, srcData, dm, :adjoint, F.options)
end

propagate(t::Tuple{judiPropagator, AbstractVector}) = propagate(t[1], t[2])

"""
    run_and_reduce(func, pool, nsrc, arg_func)

Runs the function `func` for indices `1:nsrc` within arguments `func(arg_func(i))`. If the 
the pool is empty, a standard loop and accumulation is ran. If the pool is a julia WorkerPool or
any custom Distributed pool, he loop is distributed via `remotecall` followed by are binary tree emote reduction.
"""
function run_and_reduce(func, ::Nothing, nsrc, arg_func::Function)
    out = func(arg_func(1))
    for i = 2:nsrc
        args_loc = arg_func(i)
        tmp = func(args_loc)
        single_reduce!(out, tmp)
    end
    return out
end

function run_and_reduce(func, pool, nsrc, arg_func::Function)
    res = Vector{Future}(undef, nsrc)
    @sync for i = 1:nsrc
        args_loc = arg_func(i)
        res[i] = remotecall(func, pool, args_loc)
    end
    res = reduce!(res)
    return res
end

src_i(::judiJacobian{T, :born, FT}, q, ::Integer) where {T, FT} = q
src_i(::judiPropagator{T, O}, q::judiMultiSourceVector{T}, i::Integer) where {T, O} = q[i]
src_i(::judiPropagator{T, O}, q::Vector{<:Array{T}}, i::Integer) where {T, O} = q[i]

get_nsrc(::judiPropagator, q::judiMultiSourceVector) = q.nsrc
get_nsrc(::judiPropagator, q::Vector{<:Array}) = length(q)
get_nsrc(J::judiJacobian, ::dmType) = J.q.nsrc

"""
    multi_src_propagate(F::judiPropagator{T, O}, q::AbstractVector)

Propagates the source `q` with the `F` propagator. The return type is infered from `F` and the 
propagation kernel is defined by `O` (forward, adjoint, born or adjoint_born).
"""
function multi_src_propagate(F::judiPropagator{T, O}, q::AbstractVector{T}) where {T<:Number, O}
    q = process_input_data(F, q)
    # Number of sources and init result
    nsrc = get_nsrc(F, q)
    pool = _worker_pool()
    arg_func = i -> (F[i], src_i(F, q, i))
    # Distribute source
    res = run_and_reduce(propagate, pool, nsrc, arg_func)
    return as_vec(res,  Val(F.options.return_array))
end

"""
    multi_src_fg!(G, model, q, dobs, dm; options=Options(), nlind=false, lin=false)

This is the main multi-source wrapper function for `fwi_objective` and `lsrtm_objective`.
Computes the misifit and gradient (LSRTM if `lin` else FWI) for the given `q` source and `dobs` and
perturbation `dm`.
"""
function multi_src_fg!(G, model, q, dobs, dm; options=Options(), nlind=false, lin=false)
    # Number of sources and init result
    nsrc = try q.nsrc catch; dobs.nsrc end
    pool = _worker_pool()
    # Distribute source
    arg_func = i -> (model, q[i], dobs[i], dm, options, nlind, lin)
    # Distribute source
    res = run_and_reduce(multi_src_fg, pool, nsrc, arg_func)
    f, g = as_vec(res,  Val(options.return_array))
    G .= g
    return f
end
