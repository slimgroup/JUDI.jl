"""
    LazyPropagation
        post::Function
        F::judiPropagator
        q

A structure to encapsulate a JUDI propagation that may not need to be computed. This allows to
to bypass ChainRule's thunks that do not play well with Zygote (may be fixed). Only a few arithmetic operations
are supported. By default, any attempt to perform an arithmectic (+, -, *, ...) operation on a LazyPropagation will evaluate the propagation.

**Note**: Long term, this should be extended in order to allow writing FWI/LSRTM with linear operators
rather than the `fwi_objective/lsrtm_objective` wrapper and infer what needs to be computed from these unevaluated propagations.

Parameters
===========
* `RT`: the return type of F*q for dispatch
* `post`: Function (default to identity). Function to be applied to the result `post(F*q)` after propagation
* `F`: the JUDI propgator
* `q`: The source to compute F*q
"""
struct LazyPropagation
    post::Function
    F::judiPropagator
    q
end

eval_prop(F::LazyPropagation) = F.post(F.F * F.q)
Base.collect(F::LazyPropagation) = eval_prop(F)
LazyPropagation(F::judiPropagator, q) = LazyPropagation(identity, F, q)

# Only a few arithmetic operation are supported

for op in [:+, :-, :*, :/]
    @eval begin
        $(op)(F::LazyPropagation, y::AbstractArray{T}) where T = $(op)(eval_prop(F), y)
        $(op)(y::AbstractArray{T}, F::LazyPropagation) where T = $(op)(y, eval_prop(F))
        $(op)(y::LazyPropagation, F::LazyPropagation) = $(op)(eval_prop(y), eval_prop(F))
        broadcasted(::typeof($op), y::LazyPropagation, F::LazyPropagation) = broadcasted($(op), eval_prop(y), eval_prop(F))
    end
    for YT ∈ [AbstractArray, Broadcast.Broadcasted]
        @eval begin
            broadcasted(::typeof($op), F::LazyPropagation, y::$(YT)) = broadcasted($(op), eval_prop(F), y)
            broadcasted(::typeof($op), y::$(YT), F::LazyPropagation) = broadcasted($(op), y, eval_prop(F))
        end
    end
end

broadcasted(::typeof(^), y::LazyPropagation, p::Real) = eval_prop(y).^(p)
*(F::judiPropagator, q::LazyPropagation) = F*eval_prop(q)
*(M::Preconditioner, q::LazyPropagation) = M*eval_prop(q)
matvec(M::Preconditioner, q::LazyPropagation) = matvec(M, eval_prop(q))

reshape(F::LazyPropagation, dims...) = LazyPropagation(x->reshape(x, dims...), F.F, F.q)
copyto!(x::AbstractArray, F::LazyPropagation) = copyto!(x, eval_prop(F))
dot(x::AbstractArray, F::LazyPropagation) = dot(x, eval_prop(F))
dot(F::LazyPropagation, x::AbstractArray) = dot(x, F)
norm(F::LazyPropagation, p::Real=2) = norm(eval_prop(F), p)
adjoint(F::JUDI.LazyPropagation) = F

############################ Two params rules ############################################
function rrule(F::judiPropagator{T, O}, m::AbstractArray{T}, q::AbstractArray{T}) where {T, O}
    y = F(m, q)
    function pullback(Δy)
        dm = ∇m(F, m, q, Δy)
        dq = ∇source(F, m, Δy)
        return (NoTangent(), dm, dq)
    end
    y = F.options.return_array ? reshape(y, F.rInterpolation, F.model; with_batch=true) : y
    return y, pullback
end

function ∇source(F::judiPropagator{T, O}, m::AbstractArray{T}, Δy) where {T, O}
    ra = F.options.return_array
    # Reshape if vector
    post::Function = ra ? (dq -> reshape_array(dq, F')) : identity
    return LazyPropagation(post, F(m)', Δy)
end

function ∇m(F::judiPropagator{T, :forward}, m::AbstractArray{T}, q::AbstractArray{T}, Δy) where {T}
    ra = F.options.return_array
    # Reshape if vector and a-dimensionalize
    post::Function = ra ? (dm -> reshape(dm, F.model.n..., 1, 1)) : identity
    q = _as_src(F.qInjection.op, F.model, q)
    return LazyPropagation(post, judiJacobian(F(m), q)', Δy)
end

∇m(F::judiPropagator{T, :adjoint}, m, q, Δy) where {T} = ∇m(adjoint(F), m, Δy, q)

############################ Single param rules ############################################
# We derive the following rule based on linear algebra calculus for parametric matrices
# The derivation of the dF where F = F(m) is derived below for F a wave-equation
# propagator. The derivation is not valid for J the Jacobian that is not currently supported
# for this case but is ony supported for J(q) where q is the source of the underlying wave equation.
#
# d (f(F*x))/dm = tr( (d (f(F*x))/dF)^T dF/dm)
# dF/dm = J(m) 
# d (f(F*x))/dF = dy x^T
# d (f(F*x))/dm  = tr(  (dy x^T)^T J ) = tr( x dy^T dF/dm ) = tr( x dy^T dF/dm )
#             = tr( - Pr Ai da/dm Ai Ps' x dy^T)
#             = tr( - Pr Ai (u.dt2) dy^T)
#             = tr( - dy^T  Pr Ai (u.dt2) )
#             = tr(- (u.dt2)^T  Ai^T Pr^ dy) = J^T dy
# F = Pr Ai(m) Ps'
# dF/dm = Pr dAi/dm Ps' = - Pr Ai da/dm Ai Ps'
#
#
# Finally, for simplicity, we directly return `dm` rather than a Tangent to avoid
# complications with the fields.

function rrule(::typeof(*), F::judiPropagator, x::AbstractArray{T}) where T
    ra = F.options.return_array
    y = F*x
    postx = ra ? (dx -> reshape(dx, size(x))) : identity
    function Fback(Δy)
        dx = LazyPropagation(postx, F', Δy)
        # F is m parametric
        dF = ∇prop(F, x, Δy)
        return NoTangent(), dF, dx
    end
    y = F.options.return_array ? reshape_array(y, F) : y
    return y, Fback
end

# Rule for x->F(x). Currently supported are
# m -> F(m) where m is the squared slowness and F is a wave-equation propagator
# q -> J(a) where q is a source and J is the Jacobian w.r.t m of a wave-equation propagator
# This rules expect the input of the pullback to come from F*q and to be a LazyPropagation
function rrule(F::judiPropagator, x)
    Fx = F(x)
    postx = F.options.return_array ? (dx -> reshape(dx, size(x))) : identity
    function backx(ΔF)
        dx = LazyPropagation(postx, ΔF.F, ΔF.q)
        return NoTangent(), dx
    end
    return Fx, backx
end

∇prop(F::judiPropagator, q::AbstractArray, dy::AbstractArray) = LazyPropagation(judiJacobian(F, q)', dy)
∇prop(J::judiJacobian{D, :born, FT}, dm::AbstractArray, δd::AbstractArray) where {D, FT} = LazyPropagation(judiJacobian(adjoint(J.F), δd), dm)
∇prop(J::judiJacobian{D, :adjoint_born, FT}, δd::AbstractArray, dm::AbstractArray) where {D, FT} = LazyPropagation(judiJacobian(adjoint(J.F), δd), dm)

# projection
(project::ProjectTo{AbstractArray})(dx::PhysicalParameter) = project(reshape(dx.data, project.axes))
(project::ProjectTo{AbstractArray})(dx::LazyPropagation) = project(reshape(eval_prop(dx), project.axes))

# Reshaping
reshape_array(u, F::judiPropagator) = reshape(u, F.rInterpolation, F.model; with_batch=true)
reshape_array(u, F::judiAbstractJacobian{T, :adjoint_born, FT}) where {T, FT} = reshape(u, F.model.n..., 1, 1)

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
