# Functions for time-domain modeling and inversion using devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
# Updated, December 2020, Mathias Louboutin, mlouboutin3@gatech.edu

#############################################################################
# Basics
include("LinearOperators/basics.jl")

#############################################################################
# Containers
const Pdtypes = Union{Float32, Float16}

include("Types/ModelStructure.jl")    # model container
include("Types/GeometryStructure.jl") # source or receiver setup, recording time and sampling
include("Types/OptionsStructure.jl")

#############################################################################
# Abstract vectors
include("Types/abstract.jl")
include("Types/lazy_msv.jl")
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
include("Utils/time_utilities.jl")
include("Modeling/losses.jl")

#############################################################################
# Linear operators
include("LinearOperators/lazy.jl")
include("LinearOperators/operators.jl")
include("LinearOperators/callable.jl")


#############################################################################
# Preconditioners
include("Preconditioners/base.jl")
include("Preconditioners/utils.jl")
include("Preconditioners/DataPreconditioners.jl")
include("Preconditioners/ModelPreconditioners.jl")

#############################################################################
# PDE solvers
include("Modeling/distributed.jl") # Modeling functions utilities
include("Modeling/python_interface.jl")  # forward/adjoint linear/nonlinear modeling
include("Modeling/time_modeling_serial.jl")  # forward/adjoint linear/nonlinear modeling
include("Modeling/misfit_fg.jl")  # FWI/LSRTM objective function value and gradient
include("Modeling/twri_objective.jl")  # TWRI objective function value and gradient
include("Modeling/propagation.jl")

#############################################################################
# Extra that need all imports

############################################################################################################################
# Enforce right precedence. Mainly we always want (rightfully)
# - First data operation on the right
# - Then propagation
# - Then right preconditioning
# I.e Ml * P * M * q must do Ml * (P * (M * q))
# It''s easier to just hard code the few cases that can happen

for T in [judiMultiSourceVector, dmType]
    @eval *(Ml::Preconditioner, P::judiPropagator, Mr::Preconditioner, v::$(T)) = Ml * (P * (Mr * v))
    @eval *(P::judiPropagator, Mr::Preconditioner, v::$(T)) = P * (Mr * v)
    @eval *(Ml::Preconditioner, P::judiPropagator, v::$(T)) = Ml * (P * v)
end