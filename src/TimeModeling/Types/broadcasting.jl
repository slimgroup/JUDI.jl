import Base.Broadcast: ArrayStyle, extrude
####################################################################################################

BroadcastStyle(::Type{<:judiMultiSourceVector}) = ArrayStyle{judiMultiSourceVector}()

function similar(bc::Broadcast.Broadcasted{ArrayStyle{judiMultiSourceVector}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ArrayAndChar:
    A = find_msv(bc)
    return similar(A, ElType)
end

"`A = find_aac(As)` returns the first PhysicalParameter among the arguments."
find_msv(bc::Base.Broadcast.Broadcasted) = find_msv(bc.args)
find_msv(args::Tuple) = find_msv(find_msv(args[1]), Base.tail(args))
find_msv(x) = x
find_msv(::Tuple{}) = nothing
find_msv(a::judiMultiSourceVector, rest) = a
find_msv(::Any, rest) = find_msv(rest)

extrude(x::judiMultiSourceVector) = extrude(x.data)

# Add broadcasting by hand due to the per source indexing
for func ∈ [:lmul!, :rmul!, :rdiv!, :ldiv!]
    @eval begin
        $func(ms::judiMultiSourceVector, x::Number) = $func(ms.data, x)
        $func(x::Number, ms::judiMultiSourceVector) = $func(x, ms.data)
    end
end

# Broadcasted custom type
get_src(ms::judiMultiSourceVector, j) = ms.data[j]
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
    ms = similar(find_msv((m1, m2)))
    for i=1:ms.nsrc
        ms.data[i] = materialize(broadcasted(bc.op, get_src(m1, i), get_src(m2, i)))
    end
    ms
end

function materialize!(ms::judiMultiSourceVector, bc::MultiSource)
    m1, m2 = materialize(bc.m1), materialize(bc.m2)
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
    for LT in [Number, Vector{<:Array}, Array{<:Number, <:Integer}]
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