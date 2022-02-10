function make_input(rI::judiProjection{T}, qI::judiProjection{T}, q::judiVector{T, AT}, pysolver::PyObject) where {T, AT}
    qIn = get_source(q, convert(Float32, pysolver.dt))
    src_coords = get_coords(qI)
    rec_coords = get_coords(rI)
    Dict(:save=>false, :wavelet=>qIn, :src_coords=>src_coords, :rec_coords=>rec_coords)
end 

function propagate(solver::Symbol, op::String, rI::AnyProjection{T}, qI::AnyProjection{T}, q::judiMultiSourceVector{T}) where T
    pysolver = getfield(JUDI, solver)
    op = eval(:($pysolver.$(op)))
    # Out type
    Tout = out_type(rI, pysolver."model".dim)
    # Make Options
    prop_kw = make_input(rI, qI, q, pysolver)
    # Propagate
    dout = pycall(op, Tout; prop_kw...)
    # create out
    dout = process_out(dout, rI, pysolver.dt)
    return dout
end

function propagate(solver::Symbol, op::String, q::judiVector, d::judiVector)
    pysolver = getfield(JUDI, solver)
    op = eval(:($pysolver.$(op)))
    # Interpolate input data to computational grid
    dtComp = convert(Float32, pysolver.dt)
    qIn = time_resample(q.data[1], q.geometry.dt[1], dtComp)
    dIn = time_resample(d.data[1], d.geometry.dt[1], dtComp)
    src_coords = hcat(q.geometry.xloc[1], q.geometry.zloc[1])
    rec_coords = hcat(d.geometry.xloc[1], d.geometry.zloc[1])
    # Propagate
    grad = pycall(op, Array{Float32, 2}, wavelet=qIn, src_coords=src_coords, rec_coords=rec_coords, rec_data=dIn)
    # create out
    pymodel = pysolver."model"
    grad = remove_padding(grad, pymodel.padsizes; true_adjoint=pysolver.options["sum_padding"])
    return PhysicalParameter(grad, pymodel.spacing, pymodel.origin)
end


function *(F::judiDataSourceModeling, q::judiMultiSourceVector)
    # Number of sources and init result
    nsrc = q.nsrc
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    # Make sure the model has correct values
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, solver(F), operator(F), F.rInterpolation[i], F.qInjection[i], q[i])
    end
    res = reduce!(res)
    return res
end

function *(J::judiJacobian{D, :born, FT}, dm::AbstractVector{D}) where {D, FT}
    # Number of sources and init result
    nsrc = J.q.nsrc
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    # Make sure the model has correct values
    set_dm!(J.model, J.options, solver(J), dm)
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, solver(J), operator(J), J.F.rInterpolation[i], J.F.qInjection[i], J.q[i])
    end
    res = reduce!(res)
    return res
end

function *(J::judiJacobian{D, :adjoint_born, FT}, δd::judiVector) where {D, FT}
    # Number of sources and init result
    nsrc = J.q.nsrc
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, solver(J), operator(J), J.q[i], δd)
    end
    res = reduce!(res)
    return res
end