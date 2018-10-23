module JUDI

using PyCall, JOLI, SeisIO, Dierckx, Distributed, Pkg, Printf

# function to prepend Python paths
function prependMyPyPath(d::String)
    myd = joinpath(dirname(pathof(JUDI)), d)
    pushfirst!(PyVector(pyimport("sys")["path"]), myd)
end

# prepend Python paths
prependMyPyPath("Python")

# submodule TimeModeling
include("TimeModeling/TimeModeling.jl")

# submodule Optimization
include("Optimization/SLIM_optim.jl")

end
