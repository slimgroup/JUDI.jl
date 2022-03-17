const SourceType{T} = Union{Vector{Array{T}}, judiMultiSourceVector{T}, PhysicalParameter{T}}

"""
    propagate(F::judiPropagator{T, mode}, q)

Base propagation interfaces that calls the devito `mode` propagator (forward/adjoint/..)
with `q` as a source. The return type is infered from `F`.
"""
function propagate(F::judiPropagator{T, O}, q::SourceType{T}) where {T, O}
    srcGeometry, srcData, recGeometry, recData, dm = make_input(F, q)
    return time_modeling(F.model, srcGeometry, srcData, recGeometry, recData, dm, O, F.options)
end

src_i(::judiJacobian{T, :born, FT}, q, ::Integer) where {T, FT} = q
src_i(::judiPropagator, q, i) = q[i]

function *(F::judiPropagator{T, O}, q::SourceType{T}) where {T<:Number, O}
    # Number of sources and init result
    nsrc = try q.nsrc catch; F.q.nsrc end
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    # Distribute source
    propagate(F[1], src_i(F, q, 1))
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, F[i], src_i(F, q, i))
    end
    # Reduce result
    res = reduce!(res)
    return as_vec(res,  Val(F.options.return_array))
end

*(F::judiPropagator{T, O}, q::Vector{T}) where {T, O} = F*process_input_data(q, F.model, nsrc(F))