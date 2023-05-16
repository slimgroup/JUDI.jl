# Functions for time-domain modeling and inversion using devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
# Updated, December 2020, Mathias Louboutin, mlouboutin3@gatech.edu

#############################################################################
# Basics
include("LinearOperators/basics.jl")

#############################################################################
# Containers
include("Types/ModelStructure.jl")    # model container
include("Types/GeometryStructure.jl") # source or receiver setup, recording time and sampling
include("Types/OptionsStructure.jl")

#############################################################################
# Abstract vectors
include("Types/abstract.jl")
include("Types/broadcasting.jl")
include("Types/judiWavefield.jl") # dense RHS (wavefield)
include("Types/judiWeights.jl")    # Extended source weight vector
include("Types/judiVector.jl")    # Julia data container
include("Types/judiComposites.jl")    # A composite type to work with hcat/vcat of judi types

# Utility types
const SourceType{T} = Union{Vector{T}, Matrix{T}, judiMultiSourceVector{T}, PhysicalParameter{T}}
const dmType{T} = Union{Vector{T}, PhysicalParameter{T}}

#############################################################################
# Utils
include("Utils/auxiliaryFunctions.jl")
include("Modeling/losses.jl")

#############################################################################
# Linear operators
include("LinearOperators/lazy.jl")
include("LinearOperators/operators.jl")
include("LinearOperators/callable.jl")

#############################################################################
# PDE solvers
include("Modeling/distributed.jl") # Modeling functions utilities
include("Modeling/python_interface.jl")  # forward/adjoint linear/nonlinear modeling
include("Modeling/time_modeling_serial.jl")  # forward/adjoint linear/nonlinear modeling
include("Modeling/misfit_fg.jl")  # FWI/LSRTM objective function value and gradient
include("Modeling/twri_objective.jl")  # TWRI objective function value and gradient
include("Modeling/propagation.jl")

#############################################################################
# Preconditioners
include("Preconditioners/base.jl")
include("Preconditioners/utils.jl")
include("Preconditioners/DataPreconditioners.jl")
include("Preconditioners/ModelPreconditioners.jl")
