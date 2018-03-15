# Module with functions for time-domain modeling and inversion using OPESCI/devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
#

module TimeModeling

using PyCall, JOLI, SeisIO, Dierckx, ApproXD

@pyimport PyModel as pm
@pyimport JAcoustic_codegen as ac
@pyimport TTI_operators as tti
import Base.*, Base./, Base.+, Base.-, Base.ctranspose, Base.conj, Base.vcat, Base.vec, Base.dot, Base.norm, Base.abs, Base.getindex, Base.similar, Base.copy!
import Base.LinAlg.scale!, Base.LinAlg.A_mul_B!, Base.LinAlg.Ac_mul_B!, Base.BLAS.axpy!, Base.broadcast!


#############################################################################
# Containers
include("ModelStructure.jl")    # model container
include("InfoStructure.jl") # basic information required by all operators
include("GeometryStructure.jl") # source or receiver setup, recording time and sampling
include("OptionsStructure.jl")
include("auxiliaryFunctions.jl")

#############################################################################
# Abstract vectors
include("judiRHS.jl")   # RHS to be multiplied with linear operator
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
