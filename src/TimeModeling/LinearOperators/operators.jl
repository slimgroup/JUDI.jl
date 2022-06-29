export judiModeling, judiDataModeling, judiDataSourceModeling, judiPointSourceModeling, judiJacobian
export judiSimSrcWeights, judiSimSrcWeightsException

const adjoint_map = Dict(:forward => :adjoint, :adjoint => :forward, :born => :adjoint_born, :adjoint_born => :born)

# Base abstract types
abstract type judiPropagator{D, O} <: joAbstractLinearOperator{D, D} end
abstract type judiComposedPropagator{D, O} <: judiPropagator{D, O} end
abstract type judiAbstractJacobian{D, O, FT} <: judiComposedPropagator{D, O} end

############################################################################################################################
# Concrete types
struct judiModeling{D, O} <: judiPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    model::Model
    options::JUDIOptions
end

struct judiPointSourceModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    F::judiModeling{D, O}
    qInjection::AdjointProjection{D}
    function judiPointSourceModeling{D, O}(F::judiModeling{D, O}, qInjection::AdjointProjection{D}) where {D, O}
        # instantiate un-initialized sizes
        ts = time_space_src(get_nsrc(qInjection), get_nt(qInjection), F.model.n)
        merge!(F.n, ts)
        merge!(F.m, ts)
        update_size(qInjection, F)
        new(F.m, qInjection.n, F, qInjection)
    end
end

struct judiDataModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    rInterpolation::Projection{D}
    F::judiModeling{D, O}
    function judiDataModeling{D, O}(rInterpolation::Projection{D}, F::judiModeling{D, O}) where {D, O}
        ts = time_space_src(get_nsrc(rInterpolation), get_nt(rInterpolation), F.model.n)
        merge!(F.n, ts)
        merge!(F.m, ts)
        update_size(rInterpolation, F)
        new(rInterpolation.m, F.n, rInterpolation, F)
    end
end

struct judiDataSourceModeling{D, O} <: judiComposedPropagator{D, O}
    m::AbstractSize
    n::AbstractSize
    rInterpolation::Projection{D}
    F::judiModeling{D, O}
    qInjection::AdjointProjection{D}
    function judiDataSourceModeling{D, O}(rInterpolation::Projection{D}, F::judiModeling{D, O}, qInjection::AdjointProjection{D}) where {D, O}
        ts = time_space_src(get_nsrc(rInterpolation), get_nt(rInterpolation), F.model.n)
        merge!(F.n, ts)
        merge!(F.m, ts)
        update_size(rInterpolation, F)
        update_size(qInjection, F)
        new(rInterpolation.m, qInjection.n, rInterpolation, F, qInjection)
    end
end

struct judiJacobian{D, O, FT} <: judiAbstractJacobian{D, O, FT}
    m::AbstractSize
    n::AbstractSize
    F::FT
    q::judiMultiSourceVector
end

struct LazyScal
    s::Number
    P::judiPropagator
end

############################################################################################################################
# Base and JOLI compat
_get_property(J::judiPropagator, ::Val{:fop}) = x -> J*x
_get_property(J::judiPropagator, ::Val{:fop_T}) = x -> J'*x
_get_property(J::judiPropagator, ::Val{:name}) = "$(typeof(J))"
_get_property(J::judiPropagator, ::Val{s}) where {s} = getfield(J, s)

_get_property(C::judiComposedPropagator, ::Val{:model}) = C.F.model
_get_property(C::judiComposedPropagator, ::Val{:options}) = C.F.options

getproperty(J::judiPropagator, s::Symbol) = _get_property(J, Val{s}())

display(P::judiPropagator{D, O}) where {D, O} = println("JUDI $(String(O)){$D} propagator $(repr(P.n)) -> $(repr(P.m))")
show(io::Union{IOBuffer, IOContext}, P::judiPropagator{D, O}) where {D, O} = print(io, "JUDI $(String(O)){$D} propagator $(repr(P.n)) -> $(repr(P.m))")

############################################################################################################################
# Constructors
"""
    judiModeling(model; options=Options())
    judiModeling(model, src_geometry, rec_geometry; options=Options())

Create seismic modeling operator for a velocity model given as a `Model` structure.
The function also takes the source and receiver geometries
as additional input arguments, which creates a combined operator `judiProjection*judiModeling*judiProjection'`.

Example
=======
`Pr` and `Ps` are projection operators of type `judiProjection` and
`q` is a data vector of type `judiVector`:
    F = judiModeling(model)
    dobs = Pr*F*Ps'*q
    F = judiModeling(model, q.geometry, rec_geometry)
    dobs = F*q
"""
function judiModeling(model::Model; options=Options())
    D = eltype(model.m)
    m = time_space(model.n)
    return judiModeling{D, :forward}(m, m, model, options)
end

judiModeling(model::Model, src_geom::Geometry, rec_geom::Geometry; options=Options()) =
    judiProjection(rec_geom) * judiModeling(model; options=options) * adjoint(judiProjection(src_geom))

"""
    judiJacobian(F,q)

Create a linearized modeling operator from the non-linear modeling operator `F` and
the source `q`. `q` can be either a judiVector (point source Jacobian) or a `judiWeight` for
extended source modeling.
`F` is a full modeling operator including source/receiver projections.

Examples
========
1) `F` is a modeling operator without source/receiver projections:
    J = judiJacobian(Pr*F*Ps',q)
2) `F` is the combined operator `Pr*F*Ps'`:
    J = judiJacobian(F,q)
"""
function judiJacobian(F::judiComposedPropagator{D, O}, q::judiMultiSourceVector; options=nothing) where {D, O}
    update!(F.F.options, options)
    return judiJacobian{D, :born, typeof(F)}(F.m, space(F.model.n), F, q)
end

# Backward compat with giving weights as array. Not recommened
function judiJacobian(F::judiComposedPropagator{D, O}, q::Array{D, N}; options=nothing) where {D, O, N}
    @warn "judiWeights is recommended for judiJacobian(F, weights)"
    nsrc = try length(F.qInjection.data) catch; length(F.rInterpolation.data) end
    update!(F.F.options, options)
    return judiJacobian(F, judiWeights(reshape(q, F.model.n); nsrc=nsrc))
end

############################################################################################################################
# Linear algebra (conj/adj/...)
# Adjoints
conj(F::judiPropagator) = F
transpose(F::judiPropagator) = adjoint(F)

adjoint(s::Symbol) = adjoint_map[s]
adjoint(F::judiModeling{D, O}) where {D, O} = judiModeling{D, adjoint(O)}(F.n, F.m, F.model, F.options)
adjoint(F::judiDataModeling{D, O}) where {D, O} = judiPointSourceModeling{D, adjoint(O)}(adjoint(F.F), adjoint(F.rInterpolation))
adjoint(F::judiPointSourceModeling{D, O}) where {D, O}= judiDataModeling{D, adjoint(O)}(adjoint(F.qInjection), adjoint(F.F))
adjoint(F::judiDataSourceModeling{D, O}) where {D, O} = judiDataSourceModeling{D, adjoint(O)}(adjoint(F.qInjection), adjoint(F.F), adjoint(F.rInterpolation))
adjoint(J::judiJacobian{D, O, FT}) where {D, O, FT} = judiJacobian{D, adjoint(O), FT}(J.n, J.m, J.F, J.q)
adjoint(L::LazyScal) = LazyScal(L.s, adjoint(L.P))

# Composition
*(F::judiModeling{D, O}, P::AdjointProjection{D}) where {D, O} = judiPointSourceModeling{D, O}(F, P)
*(P::Projection{D}, F::judiModeling{D, O}) where {D, O} = judiDataModeling{D, O}(P, F)
*(P::Projection{D}, F::judiPointSourceModeling{D, O}) where {D, O} = judiDataSourceModeling{D, O}(P, F.F, F.qInjection)
*(F::judiDataModeling{D, O}, P::AdjointProjection{D}) where {D, O} = judiDataSourceModeling{D, O}(F.rInterpolation, F.F, P)
*(s::Number, P::judiPropagator) = LazyScal(s, P)
\(P::judiPropagator, s::Number) = LazyScal(1/s, P)
*(L::LazyScal, x::SourceType{T}) where {T<:Number} = L.s * (L.P * x)
*(L::LazyScal, x::Array{T, N}) where {T<:Number, N} = L.s * (L.P * x)

# Propagation via linear algebra `*`
*(F::judiPropagator{T, O}, q::judiMultiSourceVector{T}) where {T<:Number, O} = multi_src_propagate(F, q)
*(F::judiPropagator{T, O}, q::AbstractVector{T}) where {T<:Number, O} = multi_src_propagate(F, q)
*(F::judiAbstractJacobian{T, O, FT}, q::dmType{T}) where {T<:Number, O, FT} = multi_src_propagate(F, q)
# Some cases have high dimensional arrays as input such as ML applications.
*(F::judiPropagator, q::Array{T, 4}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 5}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 6}) where T = F*vec(q)

mul!(out::SourceType{T}, F::judiPropagator{T, O}, q::SourceType{T}) where {T<:Number, O} = begin y = F*q; copyto!(out, y) end
mul!(out::SourceType{T}, F::joLinearFunction{T, T}, q::SourceType{T}) where {T<:Number} = begin y = F*q; copyto!(out, y) end
mul!(out::SourceType{T}, F::judiAbstractJacobian{T, :born, FT}, q::Vector{T}) where {T<:Number, FT} = begin y = F*q[:]; copyto!(out, y) end
mul!(out::SourceType{T}, F::judiAbstractJacobian{T, :born, FT}, q::Array{T, 2}) where {T<:Number, FT} = begin y = F*q[:]; copyto!(out, y) end
mul!(out::SourceType{T}, F::judiAbstractJacobian{T, :born, FT}, q::Array{T, 3}) where {T<:Number, FT} = begin y = F*q[:]; copyto!(out, y) end
mul!(out::Array{T, 2}, F::judiAbstractJacobian{T, :adjoint_born, FT}, q::SourceType{T}) where {T<:Number, FT} =  begin y = F*q; copyto!(out, y) end
mul!(out::Array{T, 3}, F::judiAbstractJacobian{T, :adjoint_born, FT}, q::SourceType{T}) where {T<:Number, FT} =  begin y = F*q; copyto!(out, y) end

############################################################################################################################
# Propagation input
process_input_data(::judiPropagator, data::judiMultiSourceVector) = data
process_input_data(F::judiModeling, q::Vector) = process_input_data(q, F.model)
process_input_data(F::judiPointSourceModeling, q::Vector) = process_input_data(q, F.qInjection.geometry)
process_input_data(F::judiDataSourceModeling, q::Vector) = process_input_data(q, F.qInjection.data)
process_input_data(F::judiDataModeling, q::Vector) = process_input_data(q, F.model)
process_input_data(J::judiJacobian{D, :adjoint_born, FT}, q::Vector) where {D, FT} = process_input_data(q, J.F.rInterpolation.data)
process_input_data(::judiJacobian{D, :born, FT}, q::dmType{D}) where {D<:Number, FT} = q

make_input(::judiModeling, q::SourceType) = (nothing, make_input(q), nothing, nothing, nothing)
make_input(::judiModeling, rhs::judiRHS) = (make_src(rhs)..., nothing, nothing, nothing)
make_input(F::judiPointSourceModeling, q::SourceType{T}) where {T} = (make_src(q, F.qInjection)..., nothing, nothing, nothing)
make_input(F::judiPointSourceModeling, q::Matrix{T}) where {T} = (F.qInjection.data[1], q, nothing, nothing, nothing)
make_input(F::judiDataModeling, q::SourceType{T}) where {T} = (nothing, make_input(q), F.rInterpolation.geometry, nothing, nothing)
make_input(F::judiDataModeling{T, O}, q::LazyAdd{T}) where {T, O} = (make_src(q)..., F.rInterpolation.geometry, nothing, nothing)
make_input(F::judiDataModeling, rhs::judiRHS) = (make_src(rhs)..., F.rInterpolation.geometry, nothing, nothing)

make_input(F::judiDataSourceModeling, q::SourceType{T}) where {T} = (make_src(q, F.qInjection)..., F.rInterpolation.data[1], nothing, nothing)
make_input(F::judiDataSourceModeling, q::Matrix{T}) where {T} = (F.qInjection.data[1], q, F.rInterpolation.data[1], nothing, nothing)

function make_input(J::judiJacobian{D, :born, FT}, q::dmType{D}) where {D<:Number, FT}
    srcGeom, srcData = make_src(J.q, J.F.qInjection)
    return srcGeom, srcData, J.F.rInterpolation.data[1], nothing, reshape(q, J.model.n)
end 

function make_input(J::judiJacobian{D, :adjoint_born, FT}, q::SourceType{D}) where {D, FT}
    srcGeom, srcData = make_src(J.q, J.F.qInjection)
    recGeom, recData = make_src(q, J.F.rInterpolation)
    return srcGeom, srcData, recGeom, recData, nothing
end


############################################################################################################################
# Size update based on linear operator

update_size(w::judiProjection, F::judiPropagator) = set_space_size!(w.n, F.model.n)
update_size(w::jAdjoint{<:judiProjection}, F::judiPropagator) = set_space_size!(w.op.n, F.model.n)

update_size(w::judiWavelet, F::judiPropagator) = set_space_size!(w.m, F.model.n)
update_size(w::jAdjoint{<:judiWavelet}, F::judiPropagator) = set_space_size!(w.op.m, F.model.n)
############################################################################################################################
# indexing
getindex(F::judiModeling{D, O}, i) where {D, O} = judiModeling{D, O}(F.m[i], F.n[i], F.model, F.options[i])
getindex(F::judiDataModeling{D, O}, i) where {D, O} = judiDataModeling{D, O}(F.rInterpolation[i], F.F[i])
getindex(F::judiPointSourceModeling{D, O}, i) where {D, O}= judiPointSourceModeling{D, O}(F.F[i], F.qInjection[i])
getindex(F::judiDataSourceModeling{D, O}, i) where {D, O} = judiDataSourceModeling{D, O}(F.rInterpolation[i], F.F[i], F.qInjection[i])
getindex(J::judiJacobian{D, O, FT}, i) where {D, O, FT} = judiJacobian{D, O, FT}(J.m[i], J.n[i], J.F[i], J.q[i])

############################################################################################################################
# Comparisons
isequal(P1::judiPropagator, P2::judiPropagator) = P1 == P2
==(F1::judiModeling{D, O1}, F2::judiModeling{D, O2}) where {D, O1, O2} = (O1 == O2 && F1.model == F2.model && F1.options == F2.options)
==(F1::judiDataModeling, F2::judiDataModeling) = (F1.F == F2.F && F1.rInterpolation == F2.rInterpolation)
==(F1::judiPointSourceModeling, F2::judiPointSourceModeling) = (F1.F == F2.F && F1.qInjection == F2.qInjection)
==(F1::judiDataSourceModeling, F2::judiDataSourceModeling) = (F1.F == F2.F && F1.qInjection == F2.qInjection && F1.rInterpolation == F2.rInterpolation)
==(F1::judiJacobian{D, O1, FT1}, F2::judiJacobian{D, O2, FT2}) where {D, O1, O2, FT1, FT2} = (O1 == O2 && FT1 == FT2 && F1.F == F2.F && F1.q == F2.q)

############################################################
# judiSimSrcWeights ## #################################################
############################################################

# Authors: Ziyi Yin, ziyi.yin@gatech.edu
# Date: Feb 2022

############################################################

struct judiSimSrcWeights{vDT<:Number} <: joAbstractLinearOperator{vDT,vDT}
    name::String
    m::Integer
    n::Integer
    w::Matrix{vDT}  # size: number of simultaneous shots * number of sequential shots
end

mutable struct judiSimSrcWeightsException <: Exception
    msg :: String
end

function judiSimSrcWeights(w::Matrix)
    return judiSimSrcWeights("judiSimSrcWeights", size(w,1), size(w,2), w)
end


############################################################
## overloaded Base functions

# conj(judiSimSrcWeights)
conj(A::judiSimSrcWeights{vDT}) where vDT =
    judiSimSrcWeights("conj("*A.name*")",m,n,conj(w))

# adjoint(judiSimSrcWeights)
adjoint(A::judiSimSrcWeights{vDT}) where vDT =
    judiSimSrcWeights("adjoint("*A.name*")",n,m,adjoint(w))

# transpose(judiSimSrcWeights)
transpose(A::judiSimSrcWeights{vDT}) where vDT =
    judiSimSrcWeights("transpose("*A.name*")",n,m,transpose(w))

############################################################
## overloaded Base *(...judiSimSrcWeights...)

# *(judiSimSrcWeights,judiVector)
function *(A::judiSimSrcWeights{T}, v::judiVector{T, Array{T, N}}) where {T, N}

    A.n == v.nsrc || throw(judiSimSrcWeightsException("Shape mismatch: A:$(size(A)), v: $(size(v))"))
    V = merge(v, A.w)
    jo_check_type_match(T,eltype(V),join(["RDT from *(judiSimSrcWeights,judiVector):",A.name,typeof(A),eltype(V)]," / "))
    return V

end

# *(judiSimSrcWeights, judiJacobian)
function *(A::judiSimSrcWeights{D}, J::judiJacobian{D, O, FT}) where {D, O, FT}

    A.n == J.m[:src] || throw(judiSimSrcWeightsException("Shape mismatch: W:$(size(A)), J: $(size(J))"))

    # set up src/rec geometry
    Ps_sim = judiProjection(getSimGeometry(J.F.qInjection.op.geometry, A.m))
    Pr_sim = judiProjection(getSimGeometry(J.F.rInterpolation.geometry, A.m))

    # set up source
    q_sim = merge(J.q, A.w)

    # set up modeling
    F_sim = judiModeling(J.F.model; options=J.options)

    # set up linearized modeling operator
    return judiJacobian(Pr_sim*F_sim*adjoint(Ps_sim), q_sim)
end

############################################################
## utility function

# turn a source/receiver geometry to a sim src geometry 
function getSimGeometry(geometry::Geometry, nsimsrc::Int=1)

    loc = []
    for i = 1:get_nsrc(geometry)
        for j = 1:length(geometry.xloc[i])
            xloc = geometry.xloc[i][j]
            if length(geometry.yloc[i]) == 1
                yloc = geometry.yloc[i][1]
            else
                yloc = geometry.yloc[i][j]
            end
            zloc = geometry.zloc[i][j]
            union!(loc, [(xloc, yloc, zloc)])
        end
    end

    key = sort(vcat(loc...), by=x->getindex(x,1))

    xloc = @. getindex(key, 1)
    yloc = @. getindex(key, 2)
    zloc = @. getindex(key, 3)

    # set geometry
    if length(geometry.yloc[1]) == 1
        yloc = geometry.yloc[1] # don't change it if 2D data
    end

    return Geometry(xloc,yloc,zloc; dt=geometry.dt[1], t=geometry.t[1], nsrc=nsimsrc)

end