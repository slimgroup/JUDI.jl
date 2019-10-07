# Module with functions for time-domain modeling and inversion using OPESCI/devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
#

module TimeModeling

using JUDI, PyCall, JOLI, SegyIO, Dierckx, Distributed, LinearAlgebra, Base.Broadcast, FFTW

import Base.*, Base./, Base.+, Base.-, Base.copy!, Base.sum, Base.ndims
import Base.Broadcast.broadcasted, Base.BroadcastStyle, Base.Broadcast.DefaultArrayStyle
import Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex, Base.axes, Base.ndims
import LinearAlgebra.transpose, LinearAlgebra.conj, LinearAlgebra.vcat, LinearAlgebra.adjoint
import LinearAlgebra.vec, LinearAlgebra.dot, LinearAlgebra.norm, LinearAlgebra.abs
import Base.similar, Base.isapprox, Base.isequal, Base.broadcast!
import LinearAlgebra.rmul!
#import LinearAlgebra.A_mul_B!, LinearAlgebra.Ac_mul_B!, LinearAlgebra.BLAS.axpy!


#############################################################################
# Containers
include("ModelStructure.jl")    # model container
include("InfoStructure.jl") # basic information required by all operators
include("GeometryStructure.jl") # source or receiver setup, recording time and sampling
include("OptionsStructure.jl")
include("auxiliaryFunctions.jl")

#############################################################################
# Abstract vectors
include("judiWavefield.jl") # dense RHS (wavefield)
include("judiRHS.jl")   # sparse RHS (point source(s))
include("judiVector.jl")    # Julia data container

#############################################################################
# PDE solvers
include("time_modeling_serial.jl")  # forward/adjoint linear/nonlinear modeling
include("time_modeling_parallel.jl")    # parallelization for modeling
include("fwi_objective_serial.jl")  # FWI objective function value and gradient
include("fwi_objective_parallel.jl")    # parallelization for FWI gradient

#############################################################################
# Linear operators
include("judiModeling.jl")  # nonlinear modeling operator F (no projection operators)
include("judiProjection.jl")    # source/receiver projection operator
include("judiPDEfull.jl")   # modeling operator with source and receiver projection: P*F*P'
include("judiPDE.jl")   # modeling operator with lhs projection only: P*F
include("judiJacobian.jl")  # linearized modeling operator J

#############################################################################
# Preconditioners and optimization
include("seismic_preconditioners.jl")

end
