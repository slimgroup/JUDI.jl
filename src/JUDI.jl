module JUDI

using PyCall,JOLI,SeisIO,Dierckx

# function to prepend Python paths
function prependMyPyPath(d::String)
    mypath=Pkg.dir("JUDI")
    myd=joinpath(mypath,d)
    unshift!(PyVector(pyimport("sys")["path"]), myd)
end

# prepend Python paths
prependMyPyPath("src/Python")

# submodule TimeModeling
include("TimeModeling/TimeModeling.jl")

# submodule Optimization
include("Optimization/SLIM_optim.jl")

end
