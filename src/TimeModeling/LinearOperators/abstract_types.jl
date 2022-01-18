export judiAbstractLinearOperator, jAdjoint, judlLinearException

"""
    JUDI base linear type for linear operators (F, J, P, ....)

We rely on JOLi for matrix-free linear algebra and therfore require the following fields
    - name
    - m, n (size)

The standard JOLI `fop` and `fop_T` are `judi_forward` and `judi_adjoint` respectively for all operators and are to
be called as `judi_forward(J, vec)`. The `judi_adjoint` is the transpose of `judi_forward`. We provide the property to JOLI via
`J.fop` = x - > judi_forward(J, x)
"""
abstract type judiAbstractLinearOperator{D, R} <: joAbstractLinearOperator{D, R} end

get_name(J::judiAbstractLinearOperator) = split(string(typeof(J)), "{")[1]

function getproperty(J::judiAbstractLinearOperator{D, R}, s::Symbol) where {D, R}
     s == :fop && (return x-> judi_forward(J, x))
     s == :fop_T && (return x-> judi_adjoint(J, x))
     s == :fop_A && (return x-> judi_adjoint(J, x))
     s == :name && (return get_name(J))
     getfield(J, s)
end

function check_types(A::judiAbstractLinearOperator{D, R}, v::vT, dim::Int) where {D, R, vT}
	size(A, dim) == size(v,1) || throw(judiLinearException("Shape[$(dim)] mismatch: A:$(size(A, dim)), v: $(size(v, 1))"))
	jo_check_type_match(D, eltype(v), join(["DDT for *(judiModeling,$(vT)):",typeof(A), eltype(v)]," / "))
end

function mul(J::judiAbstractLinearOperator{D, R}, v) where {D, R}
    check_types(J, v, 2)
    out = judi_forward(J, v)
    check_types(J, out, 1)
    out
end

"""
    Lazy adjoint similar to Julia's LinearAlgebra
"""
struct jAdjoint{T, D, R} <: judiAbstractLinearOperator{D, R}
    J::T
    m::Integer
    n::Integer
    jAdjoint{T, D, R}(x) where {T, D, R} = new(x, x.n, x.m)
end

function getproperty(J::jAdjoint{T, D, R}, s::Symbol) where {T, D, R}
    s == :fop && (return x-> judi_adjoint(J, x))
    s == :fop_T && (return x-> judi_forward(J, x))
    s == :fop_A && (return x-> judi_forward(J, x))
    s == :name && (return "Adjoint($(get_name(J.J)))")
    s ∈ [:J, :m, :n] && (return getfield(J, s))
    getfield(J.J, s)
end

check_types(J::jAdjoint, v::vT, dim::Int) where vT = check_types(J.J, v, dim)

function mul(J::jAdjoint{T, D, R}, v) where {T, D, R}
    check_types(J, v, 1)
    out = judi_adjoint(J.J, v)
    check_types(J, out, 2)
    out
end

getindex(J::jAdjoint, a) = subsample(J, a)
subsample(J::jAdjoint{T, D, R}, inds) where {T, D, R} = jAdjoint{T, D, R}(subsample(J.J, inds))

# Real valued so conj is no-op
conj(J::judiAbstractLinearOperator{D, R}) where {D, R} =  J
# Real operator so adjoint and transpose are the same
for func in [:adjoint, :transpose]
    @eval begin
        $(Symbol("$(func)"))(J::judiAbstractLinearOperator{D, R}) where {D, R} = jAdjoint{typeof(J), R, D}(J)
        $(Symbol("$(func)"))(J::jAdjoint) = J.J
    end
end

for JT in [judiAbstractLinearOperator, jAdjoint]
    for T in [AbstractVector, judiWeights, judiVector, judiWavefield]
        @eval *(J::$(JT), v::vT) where {vT<:$(T)} = mul(J, v)
    end
    @eval *(J::$(JT), v::Array{T, 2}) where {T} = J*vec(v)
    @eval *(J::$(JT), v::Array{T, 3}) where {T} = J*vec(v)
end

"""
    Linear operator error handling
"""
mutable struct judiLinearException <: Exception
    msg :: String
end