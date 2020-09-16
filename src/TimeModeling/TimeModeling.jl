# Module with functions for time-domain modeling and inversion using OPESCI/devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
#
#__precompile__()
module TimeModeling

using JUDI, LinearAlgebra, Base.Broadcast, FFTW, Pkg, Printf, Distributed, IterativeSolvers
using PyCall, JOLI, SegyIO, DSP, Dierckx

import Base.*, Base./, Base.+, Base.-, Base.copy!, Base.copy, Base.sum, Base.ndims, Base.reshape, Base.fill!
import Base.Broadcast.broadcasted, Base.BroadcastStyle, Base.Broadcast.DefaultArrayStyle
import Base.getindex, Base.setindex!, Base.firstindex, Base.lastindex, Base.axes, Base.ndims
import Base.similar, Base.isapprox, Base.isequal, Base.broadcast!, Base.materialize!
import Base.eltype, Base.length, Base.size, Base.iterate

import LinearAlgebra.transpose, LinearAlgebra.conj, LinearAlgebra.vcat, LinearAlgebra.adjoint
import LinearAlgebra.vec, LinearAlgebra.dot, LinearAlgebra.norm, LinearAlgebra.abs
import LinearAlgebra.rmul!, LinearAlgebra.mul!, Base.isfinite

import IterativeSolvers.zerox


#############################################################################
# Containers
include("ModelStructure.jl")    # model container
include("InfoStructure.jl") # basic information required by all operators
include("GeometryStructure.jl") # source or receiver setup, recording time and sampling
include("OptionsStructure.jl")

#############################################################################
# Abstract vectors
include("judiWavefield.jl") # dense RHS (wavefield)
include("judiRHS.jl")   # sparse RHS (point source(s))
include("judiExtendedSource.jl")   # sparse RHS (point source(s))
include("judiWeights.jl")    # Extended source weight vector
include("judiVector.jl")    # Julia data container
include("judiComposites.jl")    # A composite type to work with hcat/vcat of judi types
include("auxiliaryFunctions.jl")

#############################################################################
# PDE solvers
include("python_interface.jl")  # forward/adjoint linear/nonlinear modeling
include("time_modeling_serial.jl")  # forward/adjoint linear/nonlinear modeling
include("time_modeling_parallel.jl")    # parallelization for modeling
include("extended_source_interface_serial.jl")  # forward/adjoint linear/nonlinear modeling w/ extended source
include("extended_source_interface_parallel.jl")    # parallelization for modeling w/ extended source
include("fwi_objective_serial.jl")  # FWI objective function value and gradient
include("fwi_objective_parallel.jl")    # parallelization for FWI gradient
include("lsrtm_objective_serial.jl")  # LSRTM objective function value and gradient
include("lsrtm_objective_parallel.jl")    # parallelization for LSRTM gradient

#############################################################################
# Linear operators
include("judiModeling.jl")  # nonlinear modeling operator F (no projection operators)
include("judiProjection.jl")    # source/receiver projection operator
include("judiLRWF.jl")   # low rank wavefield (point source(s))
include("judiPDEfull.jl")   # modeling operator with source and receiver projection: P*F*P'
include("judiPDEextended.jl")   # modeling operator for extended sources
include("judiPDE.jl")   # modeling operator with lhs projection only: P*F
include("judiJacobian.jl")  # linearized modeling operator J
include("judiJacobianExtendedSource.jl")  # Jacobian of extended source modeling operator

#############################################################################
# Preconditioners and optimization
include("seismic_preconditioners.jl")

end
