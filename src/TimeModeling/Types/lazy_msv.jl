
struct LazyData{D} <: AbstractVector{D}
    msv::judiMultiSourceVector{D}
end

getindex(ld::LazyData{D}, i) where D = get_data(ld.msv[i]).data

setindex!(::LazyData{D}, ::Any, ::Any) where D = throw(MethodError(setindex!, "LazyData is read-only"))

size(A::LazyData) = size(A.msv)

get_data(ld::LazyData{D}) where D = get_data(ld.msv)

"""
    LazyAdd
        nsrc
        A
        B
        sign

Lazy addition of two RHS (currently only judiVector). The addition isn't evaluated to avoid
large memory allocation but instead evaluates the addition (with sign `sign`) `A + sign * B`
for a single source at propagation time.
"""

struct LazyAdd{D} <: judiMultiSourceVector{D}
    nsrc::Integer
    A
    B
    sign
end


getindex(la::LazyAdd{D}, i::RangeOrVec) where D = LazyAdd{D}(length(i), la.A[i], la.B[i], la.sign)


function eval(ls::LazyAdd{D}) where D
    aloc = eval(ls.A)
    bloc = eval(ls.B)
    ga = aloc.geometry
    gb = bloc.geometry
    @assert (ga.nt == gb.nt && ga.dt == gb.dt && ga.t == gb.t)
    xloc = [vcat(ga.xloc[1], gb.xloc[1])]
    yloc = [vcat(ga.yloc[1], gb.yloc[1])]
    zloc = [vcat(ga.zloc[1], gb.zloc[1])]
    geom = GeometryIC{D}(xloc, yloc, zloc, ga.dt, ga.nt, ga.t)
    data = hcat(aloc.data[1], ls.sign*bloc.data[1])
    judiVector{D, Matrix{D}}(1, geom, [data])
end

function make_src(ls::LazyAdd{D}) where D
    q = eval(ls)
    return q.geometry[1], q.data[1]
end


"""
    LazyMul
        nsrc
        A
        B
        sign

Lazy addition of two RHS (currently only judiVector). The addition isn't evaluated to avoid
large memory allocation but instead evaluates the addition (with sign `sign`) `A + sign * B`
for a single source at propagation time.
"""

struct LazyMul{D} <: judiMultiSourceVector{D}
    nsrc::Integer
    P::joAbstractLinearOperator
    msv::judiMultiSourceVector{D}
end

getindex(la::LazyMul{D}, i::RangeOrVec) where D = LazyMul{D}(length(i), la.P[i], la.msv[i])

function make_input(lm::LazyMul{D}) where D
    @assert lm.nsrc == 1
    return make_input(lm.P * get_data(lm.msv))
end

get_data(lm::LazyMul{D}) where D = lm.P * get_data(lm.msv)

function getproperty(lm::LazyMul{D}, s::Symbol) where D
    if s == :data
        return LazyData(lm)
    elseif s == :geometry
        return lm.msv.geometry
    else
        return getfield(lm, s)
    end
end