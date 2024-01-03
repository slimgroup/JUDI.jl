BroadcastStyle(::Type{<:judiMultiSourceVector}) = ArrayStyle{judiMultiSourceVector}()

function similar(bc::Broadcast.Broadcasted{ArrayStyle{judiMultiSourceVector}}, ::Type{ElType}) where ElType
    # Scan the inputs
    A = find_bc(bc, judiMultiSourceVector)
    return similar(A, ElType)
end


"`A = find_pm(As)` returns the first PhysicalParameter among the arguments."
find_bc(bc::Base.Broadcast.Broadcasted, ::Type{T}) where T = find_bc(bc.args, T)
find_bc(args::Tuple, ::Type{T}) where T = find_bc(find_bc(args[1], T), Base.tail(args), T)
find_bc(x, ::Type{T}) where T = x
find_bc(::Tuple{}, ::Type{T}) where T = nothing
find_bc(a::T, rest, ::Type{T}) where T = a
find_bc(::Any, rest, ::Type{T}) where T = find_bc(rest, T)

extrude(x::judiMultiSourceVector) = extrude(x.data)

# Add broadcasting by hand due to the per source indexing
for func ∈ [:lmul!, :rmul!, :rdiv!, :ldiv!]
    @eval begin
        $func(ms::judiMultiSourceVector, x::Number) = $func(ms.data, x)
        $func(x::Number, ms::judiMultiSourceVector) = $func(x, ms.data)
    end
end

# Broadcasted custom type
check_compat() = true
get_src(ms::judiMultiSourceVector, j) = make_input(ms[j])
get_src(v::Vector{<:Array}, j) = v[j]
get_src(v::Array{T, N}, j) where {T<:Number, N} = v
get_src(n::Number, j) = n

struct MultiSource <: Base.AbstractBroadcasted
    m1
    m2
    op
end

function materialize(bc::MultiSource)
    m1, m2 = materialize(bc.m1), materialize(bc.m2)
    ms = similar(isa(m1, judiMultiSourceVector) ? m1 : m2)
    check_compat(m1, m2)
    for i=1:ms.nsrc
        ms.data[i] .= materialize(broadcasted(bc.op, get_src(m1, i), get_src(m2, i)))
    end
    ms
end

function materialize!(ms::judiMultiSourceVector, bc::MultiSource)
    m1, m2 = materialize(bc.m1), materialize(bc.m2)
    check_compat(ms, m1)
    check_compat(ms, m2)
    for i=1:ms.nsrc
        broadcast!(bc.op, ms.data[i], get_src(m1, i), get_src(m2, i))
    end
    nothing
end

for op ∈ [:+, :-, :*, :/]
    # Two multi source vectors
    @eval begin
        $(op)(ms1::judiMultiSourceVector, ms2::judiMultiSourceVector) = materialize(broadcasted($op, ms1, ms2))
    end
    # External types and broadcasting
    for LT in [Number, Real, Vector{<:Array}, Array{<:Number, <:Integer}]
        # +/*/... julia type vs multi source
        @eval $(op)(ms1::$(LT), ms2::judiMultiSourceVector) = materialize(broadcasted($op, ms1, ms2))
        @eval $(op)(ms1::judiMultiSourceVector, ms2::$(LT)) = materialize(broadcasted($op, ms1, ms2))
        # broadcasted julia type vs multi source vector or broadcast
        for MS in [judiMultiSourceVector, MultiSource]
            @eval broadcasted(::typeof($op), ms1::$(LT), ms2::$(MS)) = MultiSource(ms1, ms2, $op)
            @eval broadcasted(::typeof($op), ms1::$(MS), ms2::$(LT)) = MultiSource(ms1, ms2, $op)
        end
    end
    # multi source with multi source
    @eval broadcasted(::typeof($op), ms1::judiMultiSourceVector, ms2::MultiSource) = MultiSource(ms1, ms2, $op)
    @eval broadcasted(::typeof($op), ms1::MultiSource, ms2::judiMultiSourceVector) = MultiSource(ms1, ms2, $op)
    @eval broadcasted(::typeof($op), ms1::judiMultiSourceVector, ms2::judiMultiSourceVector) = MultiSource(ms1, ms2, $op)
    @eval broadcasted(::typeof($op), ms1::MultiSource, ms2::MultiSource) = MultiSource(ms1, ms2, $op)
end
