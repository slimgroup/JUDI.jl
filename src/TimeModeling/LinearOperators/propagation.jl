
const SourceType{T} = Union{Vector{T}, judiMultiSourceVector{T}, PhysicalParameter{T}}

"""
    propagate(F::judiPropagator{T, mode}, q)

Base propagation interfaces that calls the devito `mode` propagator (forward/adjoint/..)
with `q` as a source. The return type is infered from `F`.
"""
function propagate(F::judiPropagator{T, O}, q::SourceType{T}) where {T, O}
    pysolver = getfield(JUDI, solver(F))
    op = eval(:($pysolver.$O))
    # Out type
    Tout = out_type(F, pysolver."model".dim)
    # Make Options
    prop_kw = make_input(F, q, pysolver)
    # Propagate
    dout = pycall(op, Tout; prop_kw...)
    # create out
    dout = process_out(F, dout, pysolver.dt)
    return dout
end

src_i(J::judiJacobian{T, :born, FT}, q, i) where {T, FT} = J.q[i]
src_i(J, q, i) = q[i]

function *(F::judiPropagator{T, O}, q::SourceType{T}) where {T<:Number, O}
    # Number of sources and init result
    nsrc = try q.nsrc catch; F.q.nsrc end
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    # Make sure the model has correct values
    set_dm!(F, q)
    # Distribute source
    propagate(F[1], src_i(F, q, 1))
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, F[i], src_i(F, q, i))
    end
    # Reduce result
    res = reduce!(res)
    return as_vec(res,  Val(F.options.return_array))
end
