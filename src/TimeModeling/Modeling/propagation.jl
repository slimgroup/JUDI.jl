"""
    propagate(F::judiPropagator{T, mode}, q)

Base propagation interfaces that calls the devito `mode` propagator (forward/adjoint/..)
with `q` as a source. The return type is infered from `F`.
"""
function propagate(F::judiPropagator{T, O}, q::AbstractVector{T}) where {T, O}
    srcGeometry, srcData, recGeometry, recData, dm = make_input(F, q)
    return time_modeling(F.model, srcGeometry, srcData, recGeometry, recData, dm, O, F.options)
end

src_i(::judiJacobian{T, :born, FT}, q, ::Integer) where {T, FT} = q
src_i(::judiPropagator{T, O}, q::judiMultiSourceVector{T}, i::Integer) where {T, O} = q[i]

nsrc(::judiPropagator, q::judiMultiSourceVector) = q.nsrc
nsrc(::judiPropagator, q::Vector{<:Array}) = length(q)
nsrc(J::judiJacobian, ::dmType) = J.q.nsrc

*(F::judiPropagator{T, O}, q::SourceType{T}) where {T<:Number, O} = multi_src_propagate(F, q)
*(F::judiJacobian{T, O, FT}, q::PhysicalParameter{T}) where {T<:Number, O, FT} = multi_src_propagate(F, q)

function multi_src_propagate(F::judiPropagator{T, O}, q::AbstractVector{T}) where {T<:Number, O}
    q = process_input_data(F, q)
    # Number of sources and init result
    nsrc = nsrc(F, q)
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    # Distribute source
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, F[i], src_i(F, q, i))
    end
    # Reduce result
    res = reduce!(res)
    return as_vec(res,  Val(F.options.return_array))
end

function multi_src_fg!(G, model, q, dobs, dm; options=Options(), nlind=false, lin=false)
    # Number of sources and init result
    nsrc = try q.nsrc catch; dobs.nsrc end
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    # Distribute source
    multi_src_fg(model, q[1], dobs[1], dm, options, nlind, lin)
    @sync for i=1:nsrc
        @async res[i] = remotecall(multi_src_fg, pool, model, q[i], dobs[i], dm, options, nlind, lin)
    end
    # Reduce result
    res = reduce!(res)
    f, g = as_vec(res,  Val(options.return_array))
    G .= g
    return f
end
