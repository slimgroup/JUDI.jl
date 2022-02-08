export judiModeling
# Base abstract type
abstract type judiPropagator{D, O} <: joAbstractLinearOperator{D, D} end

display(P::judiPropagator{D, O}) where {D, O} = println("JUDI $(operator(P)) propagator $(repr(P.n)) -> $(repr(P.m))")

const adjoint_map = Dict(:forward => :adjoint, :adjoint => :forward, :born => :gradient, :gradient => :born)
adjoint(s::Symbol) = adjoint_map[s]
# Base PDE type
struct judiModeling{D, O} <: judiPropagator{D, O}
    name::String
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    solver::Symbol
end

struct judiPointSourceModeling{D, O} <: judiPropagator{D, O}
    F::judiModeling{D, O}
    qInjection::judiProjection{D}
end

struct judiDataPointSourceModeling{D, O} <: judiPropagator{D, O}
    rInterpolation::judiProjection{D}
    F::judiModeling{D, O}
    qInjection::judiProjection{D}
end

struct judiDataModeling{D, O} <: judiPropagator{D, O}
    rInterpolation::judiProjection{D}
    F::judiModeling{D, O}
end

# Adjoints
adjoint(F::judiModeling{D, O}) where {D, O} = judiModeling{D, adjoint(O)}("judiModeling", F.m, F.n, F.model, F.options, F.solver)
adjoint(F::judiDataModeling{D, O}) where {D, O} = judiPointSourceModeling{D, adjoint(O)}(adjoint(F.F), F.rInterpolation)
adjoint(F::judiPointSourceModeling{D, O}) where {D, O}= judiDataModeling{D, adjoint(O)}(F.qInjection, adjoint(F.F))
adjoint(F::judiDataPointSourceModeling{D, O}) where {D, O} = judiDataPointSourceModeling{D, adjoint(O)}(F.qInjection, adjoint(F.F), F.rInterpolation)

solver(F::judiModeling) = F.solver
solver(F::judiPropagator) = F.F.solver

operator(F::judiPropagator{D, O}) where {D, O} = String(O)

# Constructor
function judiModeling(model; options=Options)
    D = eltype(model.m)
    m = time_space_size(ndims(model.m.data))
    solver = init_solver(model, options)
    return judiModeling{D, :forward}("judiModeling", m, m, model, options, solver)
end

function init_local(name::Symbol, model::Model, options::Options)
    pymodel = devito_model(model, options)
    opts = Dict(Symbol(s) => getfield(options, s) for s in fieldnames(JUDI.Options))
    Core.eval(JUDI, :($name = we."WaveSolver"($pymodel; $(opts)...)))
    nothing
end

function init_solver(model::Model, options::Options)
    solver = make_id()
    @sync for p in workers()
        @async remotecall_wait(init_local, p, solver, model, options)
    end
    return solver
end

*(F::judiModeling{D, O}, P::jAdjoint{judiProjection{D}}) where {D, O} = judiPointSourceModeling{D, O}(F, P.op)
*(P::judiProjection{D}, F::judiModeling{D, O}) where {D, O} = judiDataModeling{D, O}(P, F)

*(P::judiProjection{D}, F::judiPointSourceModeling{D, O}) where {D, O} = judiDataPointSourceModeling{D, O}(P, F.F, F.qInjection)
*(F::judiDataModeling{D, O}, P::jAdjoint{judiProjection{D}}) where {D, O} = judiDataPointSourceModeling{D, O}(F.rInterpolation, F.F, P.op)