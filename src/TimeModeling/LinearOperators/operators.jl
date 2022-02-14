export judiModeling, judiJacobian

const adjoint_map = Dict(:forward => :adjoint, :adjoint => :forward, :born => :adjoint_born, :adjoint_born => :born)

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
adjoint(s::Symbol) = adjoint_map[s]
set_dm!(::judiPropagator, dm) = nothing

# Base PDE type
struct judiModeling{D, O} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::Options
    solver::Symbol
end

# A base propagator returns a wavefield
out_type(::judiPropagator, ndim) = Array{Float32, ndim+1}
make_input(::judiModeling, q::judiMultiSourceVector, pysolver::PyObject) = Dict(:save=>true, :q=>q.data[1])
process_out(::judiModeling{T, O}, dout, dt) where {T, O} = judiWavefield{T}(1, dt, [dout])

==(F1::judiModeling{D, O1}, F2::judiModeling{D, O2}) where {D, O1, O2} =
    (O1 == O2 && F1.model == F2.model && F1.options == F2.options && F1.solver == F2.solver)

# Propagator with source
struct judiPointSourceModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::judiModeling{D, O}
    qInjection::AdjointProjection{D}
    judiPointSourceModeling{D, O}(F::judiModeling{D, O}, qInjection::AdjointProjection{D}) where {D, O} = new(F.m, qInjection.m, F, qInjection)
end

function make_input(F::judiPointSourceModeling{T}, q::judiMultiSourceVector{T}, pysolver::PyObject) where {T, AT}
    dt = convert(Float32, pysolver.dt)
    qI_kw = make_input(F.qInjection, dt)
    q_kw = make_input(q, dt)
    Dict(:save=>true, qI_kw..., q_kw...)
end 

==(F1::judiPointSourceModeling, F2::judiPointSourceModeling) = (F1.F == F2.F && F1.qInjection == F2.qInjection)

# Propagator with measurments and source
struct judiDataSourceModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    rInterpolation::Projection{D}
    F::judiModeling{D, O}
    qInjection::AdjointProjection{D}
    judiDataSourceModeling{D, O}(rInterpolation::Projection{D}, F::judiModeling{D, O}, qInjection::AdjointProjection{D}) where {D, O} =
        new(rInterpolation.m, qInjection.m, rInterpolation, F, qInjection)
end

function make_input(F::judiDataSourceModeling{T, O}, q::judiMultiSourceVector{T}, pysolver::PyObject) where {T, O, AT}
    dt = convert(Float32, pysolver.dt)
    qI_kw = make_input(F.qInjection, dt)
    rI_kw = make_input(F.rInterpolation, dt)
    q_kw = make_input(q, dt)
    Dict(rI_kw..., qI_kw..., q_kw...)
end

# A propagator with measurment returns an array based on the projection
out_type(F::judiDataSourceModeling, ndim) = out_type(F.rInterpolation, ndim)
process_out(F::judiDataSourceModeling, dout, dt) = process_out(F.rInterpolation, dout, dt, solver(F))

==(F1::judiDataSourceModeling, F2::judiDataSourceModeling) = (F1.F == F2.F && F1.qInjection == F2.qInjection && F1.rInterpolation == F2.rInterpolation)

# Propagator with measurments
struct judiDataModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    rInterpolation::Projection{D}
    F::judiModeling{D, O}
    judiDataModeling{D, O}(rInterpolation::Projection{D}, F::judiModeling{D, O}) where {D, O} = new(rInterpolation.m, F.n, rInterpolation, F)
end

function make_input(F::judiDataModeling{T, O}, q::judiMultiSourceVector{T}, pysolver::PyObject) where {T, O, AT}
    dt = convert(Float32, pysolver.dt)
    rI_kw = make_input(F.rInterpolation, dt)
    q_kw = make_input(q, dt)
    Dict(rI_kw..., q_kw...)
end

# A propagator with measurment returns an array based on the projection
out_type(F::judiDataModeling, ndim) = out_type(F.rInterpolation, ndim)
process_out(F::judiDataModeling, dout, dt) = process_out(F.rInterpolation, dout, dt, solver(F))

==(F1::judiDataModeling, F2::judiDataModeling) = (F1.F == F2.F && F1.rInterpolation == F2.rInterpolation)

# Jacobian
struct judiJacobian{D, O, FT} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::FT
    q::judiMultiSourceVector
end

function make_input(J::judiJacobian{D, :born, FT}, q::judiMultiSourceVector, pysolver::PyObject) where {D, FT, AT}
    dt = convert(Float32, pysolver.dt)
    qI_kw = make_input(J.F.qInjection, dt)
    rI_kw = make_input(J.F.rInterpolation, dt)
    q_kw = make_input(q, dt)
    Dict(rI_kw..., qI_kw..., q_kw...)
end

function make_input(J::judiJacobian{D, :adjoint_born, FT}, q::judiMultiSourceVector, pysolver::PyObject) where {D, FT, AT}
    dt = convert(Float32, pysolver.dt)
    rI_kw = make_input(J.F.rInterpolation, dt)
    qI_kw = make_input(J.F.qInjection, dt)
    q_kw = make_input(J.q, dt)
    rec_d = get_source(q, dt)
    Dict(rI_kw..., qI_kw..., q_kw..., :rec_data=>rec_d)
end

set_dm!(J::judiJacobian{D, :born, FT}, dm) where {D, FT} = set_dm!(J.model, J.options, solver(J), dm)

# A propagator with measurment returns an array based on the projection
out_type(J::judiJacobian{D, :born, FT}, ndim) where {D, FT} = out_type(J.F.rInterpolation, ndim)
out_type(::judiJacobian{D, :adjoint_born, FT}, ndim) where {D, FT} = Array{Float32, ndim}
process_out(J::judiJacobian{D, :born, FT}, dout, dt) where {D, FT} = process_out(J.F.rInterpolation, dout, dt, solver(J))

function process_out(J::judiJacobian{D, :adjoint_born, FT}, dout, dt) where {D, FT}
    dout = remove_padding(dout, getfield(JUDI, solver(J))."model".padsizes;
                          true_adjoint=J.options.sum_padding)
    PhysicalParameter(dout, J.model.d, J.model.o)
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

judiJacobian(F::judiPropagator{D, O}, q::judiMultiSourceVector) where {D, O} = judiJacobian{D, :born, typeof(F)}(F.m, space_size(ndims(F.model)), F, q)

# Backward compat with giving weights as array. Not recommened
function judiJacobian(F::judiComposedPropagator{D, O}, q::Array{D, N}) where {D, O, N}
    @warn "judiWeights is recommned for judiJacobian(F, weights)"
    nsrc = try length(F.qInjection.data) catch; length(F.rInterpolation.data) end
    return judiJacobian(F, judiWeights(reshape(q, F.model.n); nsrc=nsrc))
end

# Adjoints
conj(F::judiPropagator) = F
transpose(F::judiPropagator) = adjoint(F)

adjoint(F::judiModeling{D, O}) where {D, O} = judiModeling{D, adjoint(O)}(F.n, F.m, F.model, F.options, F.solver)
adjoint(F::judiDataModeling{D, O}) where {D, O} = judiPointSourceModeling{D, adjoint(O)}(adjoint(F.F), adjoint(F.rInterpolation))
adjoint(F::judiPointSourceModeling{D, O}) where {D, O}= judiDataModeling{D, adjoint(O)}(adjoint(F.qInjection), adjoint(F.F))
adjoint(F::judiDataSourceModeling{D, O}) where {D, O} = judiDataSourceModeling{D, adjoint(O)}(adjoint(F.qInjection), adjoint(F.F), adjoint(F.rInterpolation))
adjoint(J::judiJacobian{D, O, FT}) where {D, O, FT} = judiJacobian{D, adjoint(O), FT}(J.n, J.m, J.F, J.q)

# Composition
*(F::judiModeling{D, O}, P::AdjointProjection{D}) where {D, O} = judiPointSourceModeling{D, O}(F, P)
*(P::Projection{D}, F::judiModeling{D, O}) where {D, O} = judiDataModeling{D, O}(P, F)

*(P::Projection{D}, F::judiPointSourceModeling{D, O}) where {D, O} = judiDataSourceModeling{D, O}(P, F.F, F.qInjection)
*(F::judiDataModeling{D, O}, P::AdjointProjection{D}) where {D, O} = judiDataSourceModeling{D, O}(F.rInterpolation, F.F, P)

# indexing
getindex(F::judiModeling{D, O}, i) where {D, O} = judiModeling{D, O}(F.m, F.n, F.model, F.options[i], F.solver)
getindex(F::judiDataModeling{D, O}, i) where {D, O} = judiDataModeling{D, O}(F.rInterpolation[i], F.F[i])
getindex(F::judiPointSourceModeling{D, O}, i) where {D, O}= judiPointSourceModeling{D, O}(F.F[i], F.qInjection[i])
getindex(F::judiDataSourceModeling{D, O}, i) where {D, O} = judiDataSourceModeling{D, O}(F.rInterpolation[i], F.F[i], F.qInjection[i])
getindex(J::judiJacobian{D, O, FT}, i) where {D, O, FT} = judiJacobian{D, O, FT}(J.m, J.n, J.F[i], J.q[i])

##### Lazy scaling

struct LazyScal
    s::Number
    P::judiPropagator
end

*(s::Number, P::judiPropagator) = LazyScal(s, P)
\(P::judiPropagator, s::Number) = LazyScal(1/s, P)
adjoint(L::LazyScal) = LazyScal(L.s, adjoint(L.P))

*(L::LazyScal, x) = L.s * (L.P * x)