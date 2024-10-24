module JLD2JUDIExt

using JUDI
isdefined(Base, :get_extension) ? (using JLD2) : (using ..JLD2)

JLD2.rconvert(::Type{Geometry}, x::JLD2.ReconstructedMutable{N, FN, NT}) where {N, FN, NT} = Geometry([JUDI.tof32(getproperty(x, f)) for f in FN]...)
JUDI.Geometry(x::JLD2.ReconstructedMutable{N, FN, NT}) where {N, FN, NT} = Geometry([JUDI.tof32(getproperty(x, f)) for f in FN]...)


function JUDI.tof32(x::JLD2.ReconstructedStatic{N, FN, NT}) where {N, FN, NT}
    # Drop "typed" signature
    reconstructT = Symbol(split(string(N), "{")[1])
    return JUDI.tof32(getproperty(@__MODULE__, reconstructT)([getproperty(x, f) for f in FN]...))
end

end