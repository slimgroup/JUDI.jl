

function eval_op(a, b, op)
    c = typeof(a) <: joAbstractLinearOperator ? deepcopy(a) : deepcopy(b)
    for j=1:c.nsrc
        try
            c.data[j] = op(getattri(a, :data, j), getattri(b, :data, j))
        catch e
            broadcast!(op, c.data[j], getattri(a, :data, j), getattri(b, :data, j))
        end
    end
    return c
end

function eval_op_ip(a, b, op)
    # Needed because julia 1.1 has a different definition of these two
    op == ldiv! && (return lmul!(1f0/a, b))
    op == rdiv! && (return rmul!(a, 1f0/b))
    #apply in place op
    for j=1:getattr(a, :nsrc, getattr(b, :nsrc))
        op(getattri(a, :data, j), getattri(b, :data, j))
    end
    a
end

function matmulT(a::AbstractArray{T, 2}, b) where T
    return a*vec(vcat(b.data...))
end

function getattr(o, attr::Symbol, default=o)
    try
        return getfield(o, attr)
    catch e
        return default
    end
end

function getattri(o, attr::Symbol, ind::Integer, default=o)
    try
        return getproperty(o, attr)[ind]
    catch e
        return default
    end
end

tof32(x::Number) = [Float32(x)]
tof32(x::Array{T, N}) where {N, T<:Real} = T==Float32 ? x : Float32.(x)
tof32(x::Array{Array{T, N}, 1}) where {N, T<:Real} = T==Float32 ? x : tof32.(x)
tof32(x::Array{Any, 1}) = try Float32.(x) catch e tof32.(x) end
tof32(x::StepRangeLen) = Float32.(x)
tof32(x::Array{StepRangeLen}) = tof32.(x)

# Bypass mismatch in naming and fields
Base.getproperty(obj::judiWeights, sym::Symbol) = sym == :data ? getfield(obj, :weights) : getfield(obj, sym)
Base.getproperty(W::judiWavefield, sym::Symbol) = sym == :nsrc ? length(W.data) : getfield(W, sym)

# This whole part is basically taking adavantage of julia metaprogramming to define
# arithmetic operations on our types all at once. This makes sure that all is defined
# for all types properly.

for JT in [judiVector, judiWeights, judiWavefield]
    for opo=[:+, :-, :*, :/]
        @eval begin
            $opo(a::$JT, b::T) where {T<:Number} = eval_op(a, b, $opo)
            $opo(a::T, b::$JT) where {T<:Number} = eval_op(a, b, $opo)
            $opo(a::$JT, b::$JT) = eval_op(a, b, $opo)
        end
    end

    @eval -(a::$JT) = -1*a

    for ipop=[:lmul!, :rmul!, :rdiv!, :ldiv!]
        @eval begin
            $ipop(a::$JT, b::T) where {avDT, AT, T<:Number} = eval_op_ip(a, b, $ipop)
            $ipop(a::T, b::$JT) where {avDT, AT, T<:Number} = eval_op_ip(a, b, $ipop)
            $ipop(a::$JT, b::$JT) where {avDT, AT} = eval_op_ip(a, b, $ipop)
        end
    end

    @eval begin
        @doc  """
            vcat(a::Array{$($JT)})
        Concatenate an array of $($JT) into one
        """ vcat(a::Array{$JT, 1})

        @doc  """
            vcat(a::$($JT), b::$($JT))
            Concatenate a and b into a single $($JT)
        """ vcat(a::$JT, b::$JT)
    end

    @eval *(a::AbstractArray{T, 2}, b::$JT) where T = matmulT(a, b)
end

function vcat(a::Array{T, 1}) where T<:Union{judiVector, judiWeights, judiWavefield}
    return vcat(a...)
end
