"""
    propagate(F::judiPropagator{T, mode}, q)

Base propagation interfaces that calls the devito `mode` propagator (forward/adjoint/..)
with `q` as a source. The return type is infered from `F`.
"""
function propagate(F::judiPropagator{T, O}, q::AbstractArray{T}, illum::Bool) where {T, O}
    srcGeometry, srcData, recGeometry, recData, dm = make_input(F, q)
    return time_modeling(F.model, srcGeometry, srcData, recGeometry, recData, dm, O, F.options, _prop_fw(F), illum)
end

propagate(t::Tuple{judiPropagator, AbstractArray}) = propagate(t[1], t[2], compute_illum(t[1].model, t[1].mode))
propagate(F::judiPropagator{T, O}, q::AbstractArray{T}) where {T, O} = propagate(F, q, compute_illum(F.model, O))


"""
    run_and_reduce(func, pool, nsrc, arg_func)

Runs the function `func` for indices `1:nsrc` within arguments `func(arg_func(i))`. If the 
the pool is empty, a standard loop and accumulation is ran. If the pool is a julia WorkerPool or
any custom Distributed pool, the loop is distributed via `remotecall` followed by are binary tree remote reduction.
"""
function run_and_reduce(func, pool, nsrc, arg_func::Function; kw=nothing)
    # Allocate devices
    _set_devices!()
    # Run distributed loop
    res = Vector{_TFuture}(undef, nsrc)
    for i = 1:nsrc
        args_loc = arg_func(i)
        kw_loc = isnothing(kw) ? Dict() : kw(i)
        res[i] = remotecall(func, pool, args_loc...; kw_loc...)
    end
    res = reduce!(res)
    return res
end

function run_and_reduce(func, ::Nothing, nsrc, arg_func::Function; kw=nothing)
    @juditime "Running $(func) for first src" begin
        kw_loc = isnothing(kw) ? Dict() : kw(1)
        out = func(arg_func(1)...; kw_loc...)
    end
    for i=2:nsrc
        @juditime "Running $(func) for src $(i)" begin
            kw_loc = isnothing(kw) ? Dict() : kw(i)
            next = func(arg_func(i)...; kw_loc...)
        end
        single_reduce!(out, next)
    end
    out
end

function _set_devices!()
    ndevices = length(_devices)
    if ndevices < 2
        return
    end
    asyncmap(enumerate(workers())) do (pi, p)
        remotecall_wait(p) do
            pyut.set_device_ids(_devices[pi % ndevices + 1])
        end
    end
end

_prop_fw(::judiPropagator{T, O}) where {T, O} = true 
_prop_fw(::judiPropagator{T, :adjoint}) where T = false
_prop_fw(J::judiJacobian) = _prop_fw(J.F)


src_i(::judiAbstractJacobian{T, :born, FT}, q::dmType{T}, ::Integer) where {T<:Number, FT} = q
src_i(::judiPropagator{T, O}, q::judiMultiSourceVector{T}, i::Integer) where {T, O} = q[i]
src_i(::judiPropagator{T, O}, q::Vector{<:Array{T}}, i::Integer) where {T, O} = q[i]

get_nsrc(::judiPropagator, q::judiMultiSourceVector) = q.nsrc
get_nsrc(::judiPropagator, q::Vector{<:Array}) = length(q)
get_nsrc(J::judiAbstractJacobian, ::dmType{T}) where T<:Number = J.q.nsrc

"""
    multi_src_propagate(F::judiPropagator{T, O}, q::AbstractVector)

Propagates the source `q` with the `F` propagator. The return type is infered from `F` and the 
propagation kernel is defined by `O` (forward, adjoint, born or adjoint_born).
"""
function multi_src_propagate(F::judiPropagator{T, O}, q::AbstractArray{T}) where {T<:Number, O}
    q = process_input_data(F, q)
    # Number of sources and init result
    nsrc = get_nsrc(F, q)
    pool = _worker_pool()
    illum = compute_illum(F.model, O)
    arg_func = i -> (F[i], src_i(F, q, i), illum)
    # Distribute source
    res = run_and_reduce(propagate, pool, nsrc, arg_func)
    res = update_illum(res, F)
    res = _project_to_physical_domain(res, F.model)
    res = as_vec(res, Val(F.options.return_array))
    return res
end

"""
    multi_src_fg!(G, model, q, dobs, dm; options=Options(), nlind=false, lin=false)

This is the main multi-source wrapper function for `fwi_objective` and `lsrtm_objective`.
Computes the misifit and gradient (LSRTM if `lin` else FWI) for the given `q` source and `dobs` and
perturbation `dm`.
"""
function multi_src_fg!(G, model, q, dobs, dm; options=Options(), kw...)
    # Number of sources and init result
    nsrc = try q.nsrc catch; dobs.nsrc end
    pool = _worker_pool()
    illum = compute_illum(model, :adjoint_born)
    # Distribute source
    arg_func = i -> (model, q[i], dobs[i], dm, options[i])
    kw_func = i -> Dict(:illum=> illum, Dict(k => kw_i(v, i) for (k, v) in kw)...)
    # Distribute source
    res = run_and_reduce(multi_src_fg, pool, nsrc, arg_func; kw=kw_func)
    f, g = update_illum(res, model, :adjoint_born)
    f, g = as_vec(res, Val(options.return_array))
    G .+= g
    return f
end

kw_i(b::Bool, ::Integer) = b
kw_i(f::Function, ::Integer) = f
kw_i(msv::judiMultiSourceVector, i::Integer) = msv[i]
kw_i(P::DataPreconditioner, i::Integer) = P[i]
kw_i(P::ModelPreconditioner, ::Integer) = P
kw_i(P::MultiPreconditioner{TP, T}, i::Integer) where {TP, T} = MultiPreconditioner{TP, T}([kw_i(Pi, i) for Pi in P.precs])
kw_i(t::Tuple, i::Integer) = tuple(kw_i(ti, i) for ti in t)
kw_i(d::Vector{<:Preconditioner}, i::Integer) = foldr(*, [kw_i(di, i) for di in d])
kw_i(d::Tuple{Vararg{<:Preconditioner}}, i::Integer) = foldr(*, [kw_i(di, i) for di in d])
