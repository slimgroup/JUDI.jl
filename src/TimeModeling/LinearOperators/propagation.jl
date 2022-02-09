

function propagate(solver::Symbol, op::String, q::judiVector, recgeom::Geometry) where {T, N}
    pysolver = getfield(JUDI, solver)
    op = eval(:($pysolver.$(op)))
    # Interpolate input data to computational grid
    dtComp = convert(Float32, pysolver.dt)
    qIn = time_resample(q.data[1], q.geometry.dt[1], dtComp)
    src_coords = hcat(q.geometry.xloc[1], q.geometry.zloc[1])
    rec_coords = hcat(recgeom.xloc[1], recgeom.zloc[1])
    # Propagate
    rec_data = pycall(op, Array{Float32, 2}, save=false,
                      wavelet=qIn, src_coords=src_coords, rec_coords=rec_coords)
    # create out
    rec_data = time_resample(rec_data, dtComp, q.geometry.dt[1])
    return judiVector{Float32, Array{Float32, 2}}("F*q", prod(size(rec_data)), 1, 1, recgeom, [rec_data])
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


function *(F::judiDataPointSourceModeling, q::judiVector)
    # Number of sources and init result
    nsrc = q.nsrc
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    # Make sure the model has correct values
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, solver(F), operator(F), q[i], subsample(F.rInterpolation.geometry, i))
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
        @async res[i] = remotecall(propagate, pool, solver(J), operator(J), J.q[i], subsample(J.F.rInterpolation.geometry, i))
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