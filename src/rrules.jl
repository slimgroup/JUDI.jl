############################ Two params rules ############################################
function rrule(F::judiPropagator{T, O}, m::AbstractArray{T}, q::AbstractArray{T}) where {T, O}
    y = F(m, q)
    function pullback(Δy)
        dm = @thunk(∇m(F, m, q, Δy))
        dq = @thunk(∇source(F, m, Δy))
        return (NoTangent(), dm, dq)
    end
    y = F.options.return_array ? reshape(y, F.rInterpolation, F.model; with_batch=true) : y
    return y, pullback
end

function ∇source(F::judiPropagator{T, O}, m::AbstractArray{T}, Δy) where {T, O}
    ra = F.options.return_array
    dq = F'(m, Δy)
    # Reshape if vector
    dq = ra ? reshape(dq, adjoint(F.qInjection), F.model; with_batch=true) : dq
    return dq
end

function ∇m(F::judiPropagator{T, :forward}, m::AbstractArray{T}, q::AbstractArray{T}, Δy) where {T}
    ra = F.options.return_array
    dm = judiJacobian(F, q)'(m, Δy)
    # Reshape if vector and a-dimensionalize
    dm = ra ? reshape(dm, size(m)) : dm
    return dm
end

∇m(F::judiPropagator{T, :adjoint}, m, q, Δy) where {T} = ∇m(adjoint(F), m, Δy, q)

############################ Single param rules ############################################
function rrule(::typeof(*), F::judiPropagator, x::AbstractArray{T}) where T
    y = F*x
    function Fback(Δy)
        dx = @thunk(F' * Δy)
        dF = @thunk(∇prop(F, x, Δy))
        return NoTangent(), dF, dx
    end
    return y, Fback
end

function rrule(F::judiPropagator, x)
    Fx = F(x)
    function backx(ΔF)
        dx = @thunk(ΔF.F * ΔF.dir)
        return NoTangent(), dx
    end
    return Fx, backx
end

∇prop(F::judiPropagator, q::AbstractArray, dy::AbstractArray) = (F=judiJacobian(F, q)', dir=dy)
∇prop(J::judiJacobian{D, :born, FT}, dm::AbstractArray, δd::AbstractArray) where {D, FT} = (F=judiJacobian(adjoint(J.F), δd), dir=dm)
∇prop(J::judiJacobian{D, :adjoint_born, FT}, δd::AbstractArray, dm::AbstractArray) where {D, FT} = (F=judiJacobian(adjoint(J.F), δd), dir=dm)

# projection
(project::ProjectTo{AbstractArray})(dx::PhysicalParameter) = project(reshape(dx.data, project.axes))

# Adjoint to avoid odd corner cases in some Zygote version
function rrule(::typeof(adjoint), F::judiPropagator)
    Fa = adjoint(F)
    _LinOp_pullback(y) = (NoTangent(), adjoint(y))
    return Fa, _LinOp_pullback
end


# Preconditioners
function rrule(::typeof(*), P::Preconditioner, x)
    back(y) = NoTangent(), NoTangent(), matvec_T(P, y)
    return matvec(P, x), back
end