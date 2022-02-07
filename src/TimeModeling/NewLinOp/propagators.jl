export judiModeling
# Base abstract type
abstract type judiPropagator{D} <: joAbstractLinearOperator{D, D} end


# Base PDE type

struct judiModeling{D} <: judiPropagator{D}
    name::String
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    solver::Symbol
end

struct judiPointSourceModeling{D} <: judiPropagator{D}
    F::judiModeling{D}
    qInjection::jAdjoint{<:judiProjection{D}}
end

struct judiDataPointSourceModeling{D} <: judiPropagator{D}
    rInterpolation::judiProjection{D}
    F::judiModeling{D}
    qInjection::jAdjoint{<:judiProjection{D}}
end

struct judiDataModeling{D} <: judiPropagator{D}
    rInterpolation::judiProjection{D}
    F::judiModeling{D}
end


function judiModeling(model; options=Options)
    D = eltype(model.m)
    m = time_space_size(ndims(model.m.data))
    solver = init_solver(model, options)
    return judiModeling{D}("judiModeling", m, m, model, options, solver)
end


function init_solver(model, options)
    pymodel = devito_model(model, options)
    opts = Dict(s => getfield(options, s) for s in fieldnames(Options))
    solver = make_id()
    @sync for p in workers()
        @async remotecall_wait(()->Core.eval(Main, :($solver = ac."WaveSolver"($pymodel, $(opts)...))), p)
    end
    return solver
end

function forward_propagate(solver::Symbol, q::judiVector, rec_coords)
    pysolver = getfield(Main, sovber)
    # Interpolate input data to computational grid
    dtComp = pysolver.dt
    qIn = time_resample(q.data[1], q.geometry.dt[1], dtComp)[1]
    rec_data = pysolver."forward"(save=False, wavelet=qIn, src_coords=src_coords, rec_coords=rec_coords)
    # create out
    recgeom = Geomtery(rec_coords; dt=q.geometry.dt[1], t=q.geometry.t[1])
    rec_data = time_resample(rec_data, dtComp, q.geometry.dt[1])
    return judiVector{Float32, Array{Float32, 2}}("F*q", prod(size(rec_data)), 1, 1, recgeom, [rec_data])
end

function *(F::judiDataPointSourceModeling, q::judiVector)
    nsrc = q.nsrc
    res = Vector{Future}(undef, nsrc)
    @sync for i=1:nsrc
        coords_i = hcat(F.rInterpolation.geometry.xloc[i], F.rInterpolation.geometry.zloc[i])
        @async res[i] = forward_propagate(F.solver, q[i], coords_i)
    end
    reduce!(res)
    return res
end


*(F::judiModeling{D}, P::jAdjoint{judiProjection{D}}) where D = judiPointSourceModeling{D}(F, P)
*(P::judiProjection{D}, F::judiModeling{D}) where D = judiDataModeling{D}(P, F)

*(P::judiProjection{D}, F::judiPointSourceModeling{D}) where D = judiDataPointSourceModeling{D}(P, F.F, F.qInjection)
*(F::judiPointSourceModeling{D}, P::jAdjoint{judiProjection{D}}) where D = judiDataPointSourceModeling{D}(F.rInjection, F.F, P)

