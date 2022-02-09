export judiModeling, judiJacobian

# Base abstract type
abstract type judiPropagator{D, O} <: joAbstractLinearOperator{D, D} end

display(P::judiPropagator{D, O}) where {D, O} = println("JUDI $(operator(P)) propagator $(repr(P.n)) -> $(repr(P.m))")

const adjoint_map = Dict(:forward => :adjoint, :adjoint => :forward, :born => :adjoint_born, :adjoint_born => :born)

adjoint(s::Symbol) = adjoint_map[s]
transpose(F::judiPropagator) = adjoint(F)

# Base PDE type
struct judiModeling{D, O} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    solver::Symbol
end

struct judiPointSourceModeling{D, O} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    F::judiModeling{D, O}
    qInjection::judiProjection{D}
    judiPointSourceModeling{D, O}(F::judiModeling{D, O}, qInjection::judiProjection{D}) where {D, O} = new(F.m, qInjection.m, F.model, F.options, F, qInjection)
end

struct judiDataPointSourceModeling{D, O} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    rInterpolation::judiProjection{D}
    F::judiModeling{D, O}
    qInjection::judiProjection{D}
    judiDataPointSourceModeling{D, O}(rInterpolation::judiProjection{D}, F::judiModeling{D, O}, qInjection::judiProjection{D}) where {D, O} =
        new(rInterpolation.m, qInjection.m, F.model, F.options, rInterpolation, F, qInjection)
end

struct judiDataModeling{D, O} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    rInterpolation::judiProjection{D}
    F::judiModeling{D, O}
    judiDataModeling{D, O}(rInterpolation::judiProjection{D}, F::judiModeling{D, O}) where {D, O} = new(rInterpolation.m, F.n, F.model, F.options, rInterpolation, F)
end

# Jacobian
struct judiJacobian{D, O, FT} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    F::FT
    q::judiVector
end

# Adjoints
adjoint(F::judiModeling{D, O}) where {D, O} = judiModeling{D, adjoint(O)}(F.n, F.m, F.model, F.options, F.solver)
adjoint(F::judiDataModeling{D, O}) where {D, O} = judiPointSourceModeling{D, adjoint(O)}(adjoint(F.F), F.rInterpolation)
adjoint(F::judiPointSourceModeling{D, O}) where {D, O}= judiDataModeling{D, adjoint(O)}(F.qInjection, adjoint(F.F))
adjoint(F::judiDataPointSourceModeling{D, O}) where {D, O} = judiDataPointSourceModeling{D, adjoint(O)}(F.qInjection, adjoint(F.F), F.rInterpolation)

solver(F::judiModeling) = F.solver
solver(F::judiPropagator) = solver(F.F)

operator(::judiPropagator{D, O}) where {D, O} = String(O)
# Constructor
function judiModeling(model; options=Options)
    D = eltype(model.m)
    m = time_space_size(ndims(model))
    solver = init_solver(model, options)
    return judiModeling{D, :forward}(m, m, model, options, solver)
end

judiJacobian(F::judiPropagator{D, O}, q) where {D, O} = judiJacobian{D, :born, typeof(F)}(F.m, space_size(ndims(F.model)), F.model, F.options, F, q)
adjoint(J::judiJacobian{D, O, FT}) where {D, O, FT} = judiJacobian{D, adjoint(O), typeof(adjoint(J.F))}(J.n, J.m, J.model, J.options, adjoint(J.F), J.q)

*(F::judiModeling{D, O}, P::jAdjoint{judiProjection{D}}) where {D, O} = judiPointSourceModeling{D, O}(F, P.op)
*(P::judiProjection{D}, F::judiModeling{D, O}) where {D, O} = judiDataModeling{D, O}(P, F)

*(P::judiProjection{D}, F::judiPointSourceModeling{D, O}) where {D, O} = judiDataPointSourceModeling{D, O}(P, F.F, F.qInjection)
*(F::judiDataModeling{D, O}, P::jAdjoint{judiProjection{D}}) where {D, O} = judiDataPointSourceModeling{D, O}(F.rInterpolation, F.F, P.op)