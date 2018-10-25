# Information structure for linear modeling opeartors
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
#

const IntNum = Union{Integer, Tuple{Integer,Integer}, Tuple{Integer,Integer,Integer}}

export Info, compareInfo

# Object for velocity/slowness models
mutable struct Info
    n::IntNum
    nsrc::Integer
    nt::Array{Any,1}
end

# Constructor if nt is not passed as a cell
"""
    Info
        n::Integer
        nsrc::Integer
        nt::Array{Any,1}

Info structure that contains information from which the dimensions of Modeling\\
and Projection operators can be inferred.

Constructor
===========

Input arguments are total number of grid points `n`, number of source positions `nsrc` and\\
number of computational time steps `nt` (either as single integer or cell array):

    Info(n, nsrc, nt)

"""
function Info(n::IntNum, nsrc::Integer, nt::Integer)
    ntCell = Array{Any}(undef, nsrc)
    for j=1:nsrc
        ntCell[j] = nt
    end
    return Info(n,nsrc,ntCell)
end

##########################################################################################################

function compareInfo(info_A, info_B)
# Compare two info structures. Return true if structures are the same and false otherwise
    if isequal(info_A.n, info_B.n) && isequal(info_A.nsrc, info_B.nsrc) && isequal(info_A.nt, info_B.nt)
        return true
    else
        return false
    end
end
