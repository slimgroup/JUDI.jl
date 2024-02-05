module JLD2JUDIExt

isdefined(Base, :get_extension) ? (using JUDI) : (using ..JUDI)
using JLD2

function JLD2.rconvert(::Type{Geometry}, x::JLD2.ReconstructedMutable{N, FN, NT}) where {N, FN, NT}
    args = [JUDI.tof32(getproperty(x, f)) for f in FN]
    return Geometry(args...)
end

function JUDI.tof32(x::JLD2.ReconstructedStatic{N, FN, NT}) where {N, FN, NT}
    # Drop "typed" signature
    reconstructT = Symbol(split(string(N), "{")[1])
    return JUDI.tof32(eval(reconstructT)([getproperty(x, f) for f in FN]...))
end

end