export judiModeling, judiJacobian

# Base abstract type
abstract type judiPropagator{D, O} <: joAbstractLinearOperator{D, D} end

isequal(P1::judiPropagator, P2::judiPropagator) = P1 == P2

abstract type judiComposedPropagator{D, O} <: judiPropagator{D, O} end

function getproperty(C::judiComposedPropagator, s::Symbol)
    s == :model && (return C.F.model)
    s == :options && (return C.F.options)
    return getfield(C, s)
end

display(P::judiPropagator{D, O}) where {D, O} = println("JUDI $(operator(P)) propagator $(repr(P.n)) -> $(repr(P.m))")

const adjoint_map = Dict(:forward => :adjoint, :adjoint => :forward, :born => :adjoint_born, :adjoint_born => :born)

adjoint(s::Symbol) = adjoint_map[s]

# Base PDE type
struct judiModeling{D, O} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    solver::Symbol
end

==(F1::judiModeling{D, O1}, F2::judiModeling{D, O2}) where {D, O1, O2} =
    (O1 == O2 && F1.model == F2.model && F1.options == F2.options && F1.solver == F2.solver)

struct judiPointSourceModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::judiModeling{D, O}
    qInjection::AnyProjection{D}
    judiPointSourceModeling{D, O}(F::judiModeling{D, O}, qInjection::AnyProjection{D}) where {D, O} = new(F.m, qInjection.m, F, qInjection)
end

==(F1::judiPointSourceModeling, F2::judiPointSourceModeling) = (F1.F == F2.F && F1.qInjection == F2.qInjection)

struct judiDataSourceModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    rInterpolation::AnyProjection{D}
    F::judiModeling{D, O}
    qInjection::AnyProjection{D}
    judiDataSourceModeling{D, O}(rInterpolation::AnyProjection{D}, F::judiModeling{D, O}, qInjection::AnyProjection{D}) where {D, O} =
        new(rInterpolation.m, qInjection.m, rInterpolation, F, qInjection)
end

==(F1::judiDataSourceModeling, F2::judiDataSourceModeling) = (F1.F == F2.F && F1.qInjection == F2.qInjection && F1.rInterpolation == F2.rInterpolation)

struct judiDataModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    rInterpolation::AnyProjection{D}
    F::judiModeling{D, O}
    judiDataModeling{D, O}(rInterpolation::AnyProjection{D}, F::judiModeling{D, O}) where {D, O} = new(rInterpolation.m, F.n, rInterpolation, F)
end

==(F1::judiDataModeling, F2::judiDataModeling) = (F1.F == F2.F && F1.rInterpolation == F2.rInterpolation)

# Jacobian
struct judiJacobian{D, O, FT} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::FT
    q::judiMultiSourceVector
end

==(F1::judiJacobian{D, O1, FT1}, F2::judiJacobian{D, O2, FT2}) where {D, O1, O2, FT1, FT2} = (O1 == O2 && FT1 == FT2 && F1.F == F2.F && F1.q == F2.q)

solver(F::judiModeling) = F.solver
solver(F::judiPropagator) = solver(F.F)

operator(::judiPropagator{D, O}) where {D, O} = String(O)

# Constructor
function judiModeling(model::Model; options=Options())
    D = eltype(model.m)
    m = time_space_size(ndims(model))
    solver = init_solver(model, options)
    return judiModeling{D, :forward}(m, m, model, options, solver)
end

judiModeling(model::Model, src_geom::Geometry, rec_geom::Geometry; options=Options()) =
    judiProjection(rec_geom) * judiModeling(model; options=options) * adjoint(judiProjection(src_geom))

judiJacobian(F::judiPropagator{D, O}, q) where {D, O} = judiJacobian{D, :born, typeof(F)}(F.m, space_size(ndims(F.model)), F, q)

# Adjoints
conj(F::judiPropagator) = F
transpose(F::judiPropagator) = adjoint(F)

adjoint(F::judiModeling{D, O}) where {D, O} = judiModeling{D, adjoint(O)}(F.n, F.m, F.model, F.options, F.solver)
adjoint(F::judiDataModeling{D, O}) where {D, O} = judiPointSourceModeling{D, adjoint(O)}(adjoint(F.F), F.rInterpolation)
adjoint(F::judiPointSourceModeling{D, O}) where {D, O}= judiDataModeling{D, adjoint(O)}(F.qInjection, adjoint(F.F))
adjoint(F::judiDataSourceModeling{D, O}) where {D, O} = judiDataSourceModeling{D, adjoint(O)}(F.qInjection, adjoint(F.F), F.rInterpolation)
adjoint(J::judiJacobian{D, O, FT}) where {D, O, FT} = judiJacobian{D, adjoint(O), typeof(adjoint(J.F))}(J.n, J.m, adjoint(J.F), J.q)

# Composition
*(F::judiModeling{D, O}, P::jAdjoint{<:AnyProjection{D}}) where {D, O} = judiPointSourceModeling{D, O}(F, P.op)
*(P::AnyProjection{D}, F::judiModeling{D, O}) where {D, O} = judiDataModeling{D, O}(P, F)

*(P::AnyProjection{D}, F::judiPointSourceModeling{D, O}) where {D, O} = judiDataSourceModeling{D, O}(P, F.F, F.qInjection)
*(F::judiDataModeling{D, O}, P::jAdjoint{<:AnyProjection{D}}) where {D, O} = judiDataSourceModeling{D, O}(F.rInterpolation, F.F, P.op)

# indexing
getindex(F::judiModeling{D, O}, i) where {D, O} = judiModeling{D, O}(F.m, F.n, F.model, F.options[i], F.solver)
getindex(F::judiDataModeling{D, O}, i) where {D, O} = judiPointSourceModeling{D, O}(F.rInterpolation[i], F.F[i])
getindex(F::judiPointSourceModeling{D, O}, i) where {D, O}= judiDataModeling{D, O}(F.F[i], F.qInjection[i])
getindex(F::judiDataSourceModeling{D, O}, i) where {D, O} = judiDataSourceModeling{D, O}(F.rInterpolation[i], F.F[i], F.qInjection[i])
getindex(J::judiJacobian{D, O, FT}, i) where {D, O, FT} = judiJacobian{D, O, FT}(J.m, J.n, J.F[i], J.q[i])