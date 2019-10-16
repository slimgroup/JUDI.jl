#__precompile__()

module JUDI

using PyCall, JOLI, SegyIO, Dierckx, Distributed, Pkg, Printf

export JUDIPATH
JUDIPATH = dirname(pathof(JUDI))

# submodule TimeModeling
include("TimeModeling/TimeModeling.jl")

# submodule Optimization
include("Optimization/SLIM_optim.jl")

end
