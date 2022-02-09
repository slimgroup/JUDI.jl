


####################################################################################################

BroadcastStyle(::Type{<:judiMultiSourceVector}) = Broadcast.ArrayStyle{judiMultiSourceVector}()


function similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{judiMultiSourceVector}}, ::Type{ElType}) where ElType
    # Scan the inputs for the ArrayAndChar:
    A = find_aac(bc)
    return similar(A, ElType)
end

"`A = find_aac(As)` returns the first PhysicalParameter among the arguments."
find_aac(bc::Base.Broadcast.Broadcasted) = find_aac(bc.args)
find_aac(args::Tuple) = find_aac(find_aac(args[1]), Base.tail(args))
find_aac(x) = x
find_aac(::Tuple{}) = nothing
find_aac(a::judiMultiSourceVector, rest) = a
find_aac(::Any, rest) = find_aac(rest)

function copy!(x::judiMultiSourceVector, y::judiMultiSourceVector)
    for j=1:x.nsrc
        x.data[j] .= y.data[j]
    end
    for f in fieldnames(tyepof(x))
        getfield(x, f) = deepcopy(getfield(y, f) )
    end
end

copy(x::judiMultiSourceVector) = 1f0 * x
