############################ AD rules ############################################
function rrule(F::judiPropagator{T, O}, m::AbstractArray{T}, q::AbstractArray{T}) where {T, O}
    y = F(m, q)
    function pullback(Δy)
        dq = @thunk(reshape(F'(m, Δy), length(q)))
        dm = @thunk(reshape(judiJacobian(F, q)'(m, Δy), size(m)))
        return (NoTangent(), dm, dq)
    end
    return y, pullback
end

# projection
(project::ProjectTo{AbstractArray})(dx::PhysicalParameter) = project(reshape(dx.data, project.axes))

############################# mul ##########################################
# Array with additional channel and batch dim
*(F::judiPropagator, q::Array{T, 4}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 5}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 6}) where T = F*vec(q)
