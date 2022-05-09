import Base: depwarn

export Info
# Compatibility with `info`

const IntNum = Union{Integer, Tuple{Integer,Integer}, Tuple{Integer,Integer,Integer}}

mutable struct Info
    n::IntNum
    nsrc::Integer
    nt::Array{Integer,1}
end

function Info(n::IntNum, nsrc::Integer, nt::Integer)
    depwarn("Info is deprecated and will be removed in future versions", :Info; force=true)
    Info(n, nsrc, [nt for i=1:nsrc])
end


for f in [:judiModeling, :judiProjection, :judiWavefield]
    @eval function $f(info::Info, ar...;kw...)
        depwarn("$($f)(info::Info, ar...; kw...) is deprecated, use $($f)(ar...; kw...)", Symbol($f); force=true)
        $f(ar...; kw...)
    end
end

for f in [:judiLRWF, :judiRHS]
    @eval $f(info::Info, ar...;kw...) = throw(ArgumentError("$($f)(info::Info, ...) is deprecated and requires a time sampling rate `dt`"))
end
