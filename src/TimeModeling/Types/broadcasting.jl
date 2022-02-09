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

for op ∈ [:+, :-, :*, :/]
    @eval begin
        broadcast($op, x, ms::judiMultiSourceVector) = broadcast($op, x, ms.data)
        broadcast($op, ms::judiMultiSourceVector, x) = broadcast($op, ms.data, x)
        broadcast!($op, out::judiMultiSourceVector, x, ms::judiMultiSourceVector) = broadcast!($op, out.data, x, ms.data)
        broadcast!($op, out::judiMultiSourceVector, ms::judiMultiSourceVector, x) = broadcast!($op, out.data, ms.data, x)
    end
end