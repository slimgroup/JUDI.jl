export judiProjection
# Base abstract type
abstract type judiNoopOperator{D} <: joAbstractLinearOperator{D, D} end

struct judiProjection{D} <: judiNoopOperator{D}
    name::String
    m::AbstractSize
    n::AbstractSize
    geometry::Geometry
end

judiProjection(G::Geometry) = judiProjection{Float32}("Projection", _rec_space, time_space_size(2), G)
adjoint(P::judiNoopOperator{D}) where D = jAdjoint(P)

display(P::judiProjection{D}) where {D, O} = println("JUDI projection operator $(repr(P.n)) -> $(repr(P.m))")