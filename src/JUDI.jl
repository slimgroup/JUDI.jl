__precompile__()
module JUDI

using PyCall, JOLI, SegyIO, Distributed, Pkg, Printf, DSP

export JUDIPATH
JUDIPATH = dirname(pathof(JUDI))

# submodule TimeModeling
include("TimeModeling/TimeModeling.jl")

end

