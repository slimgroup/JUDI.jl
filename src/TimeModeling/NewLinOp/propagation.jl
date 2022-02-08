

function propagate(solver::Symbol, op::String, q::judiVector, recgeom)
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


function *(F::judiDataPointSourceModeling, q::judiVector)
    nsrc = q.nsrc
    pool = default_worker_pool()
    res = Vector{Future}(undef, nsrc)
    @sync for i=1:nsrc
        @async res[i] = remotecall(propagate, pool, solver(F), operator(F), q[i], subsample(F.rInterpolation.geometry, i))
    end
    res = reduce!(res)
    return res
end