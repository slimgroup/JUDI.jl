####################################################################################################

BroadcastStyle(::Type{<:judiMultiSourceVector}) = Broadcast.ArrayStyle{judiMultiSourceVector}()


function similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{judiMultiSourceVector}}, ::Type{ElType}) where ElType
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

Broadcast.extrude(x::judiMultiSourceVector) = Broadcast.extrude(x.data)


# Add broadcasting by hand due to the per source indexing
for func ∈ [:lmul!, :rmul!, :rdiv!, :ldiv!]
    @eval begin
        $func(ms::judiMultiSourceVector, x::Number) = $func(ms.data, x)
        $func(x::Number, ms::judiMultiSourceVector) = $func(x, ms.data)
    end
end

struct MultiSourceBroadcasted <: Base.AbstractBroadcasted
    bval
    data
    op
end

function materialize(bc::MultiSourceBroadcasted)
    ms = similar(bc.data)
    for i=1:ms.nsrc
        ms.data[i] = materialize(broadcasted(bc.op, bc.bval, bc.data.data[i]))
    end
    ms
end

function materialize!(ms::judiMultiSourceVector, bc::MultiSourceBroadcasted)
    for i=1:ms.nsrc
        broadcast!(bc.op, ms.data[i], bc.bval, bc.data.data[i])
    end
    nothing
end

for op ∈ [:+, :-, :*, :/]
    @eval begin
        broadcasted(::typeof($op), x, ms::judiMultiSourceVector) = MultiSourceBroadcasted(x, ms, $op)
        broadcasted(::typeof($op), ms::judiMultiSourceVector, x) = MultiSourceBroadcasted(x, ms, $op)
    end 
end