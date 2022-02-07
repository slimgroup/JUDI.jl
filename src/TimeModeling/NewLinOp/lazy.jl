export judiProjection
# Base abstract type
abstract type judiNoopOperator{D, R} <: joAbstractLinearOperator{D, R} end

struct judiProjection{D} <: judiNoopOperator{D, D}
    name::String
    m::AbstractSize
    n::AbstractSize
    geometry::Geometry
end

judiProjection(G::Geometry) = judiProjection{Float32}("Projection", _rec_space, time_space_size(2), G)