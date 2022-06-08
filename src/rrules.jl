############################ AD rules ############################################
function rrule(F::judiPropagator{T, O}, m::AbstractArray{T}, q::AbstractArray{T}) where {T, O}
    y = F(m, q)
    dims = length(q)
    function pullback(Δy)
        dm = @thunk(∇m(F, m, q, Δy))
        dq = @thunk(∇source(F, m, Δy, dims))
        return (NoTangent(), dm, dq)
    end
    y = F.options.return_array ? reshape(y, F.rInterpolation, F.model; with_batch=true) : y
    return y, pullback
end

function ∇source(F::judiPropagator{T, O}, m::AbstractArray{T}, Δy, dims::Integer) where {T, O}
    ra = F.options.return_array
    dq = F'(m, Δy)
    # Reshape if vector
    dq = ra ? reshape(dq, dims) : dq
    return dq
end


function ∇m(F::judiPropagator{T, O}, m::AbstractArray{T}, q::AbstractArray{T}, Δy) where {T, O}
    ra = F.options.return_array
    dm = judiJacobian(F, q)'(m, Δy)
    # Reshape if vector and a-dimensionalize
    dm = ra ? reshape(dm, size(m)) : dm
    return dm
end

# projection
(project::ProjectTo{AbstractArray})(dx::PhysicalParameter) = project(reshape(dx.data, project.axes))

############################# mul ##########################################
# Array with additional channel and batch dim
*(F::judiPropagator, q::Array{T, 4}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 5}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 6}) where T = F*vec(q)
