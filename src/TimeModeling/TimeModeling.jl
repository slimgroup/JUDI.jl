# Functions for time-domain modeling and inversion using devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
# Updated, December 2020, Mathias Louboutin, mlouboutin3@gatech.edu

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

#############################################################################
# Utils
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
include("twri_objective_serial.jl")  # TWRI objective function value and gradient
include("twri_objective_parallel.jl")    # parallelization for TWRI gradient

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

#############################################################################
# Utility types
const SourceTypes = Union{judiVector, Tuple{judiWeights, judiLRWF}}
# PDE types are callable w.r.t non-linear parameters, i.e F(model) or F(;m=m, epsilon=new_epsilon)
const pde_types = Union{judiModeling, judiModelingAdjoint, judiPDEfull, judiPDE, judiPDEadjoint,
                        judiJacobian, judiJacobianExQ, judiPDEextended}

function __init__()
  if VERSION>v"1.2"
    @eval function (F::pde_types)(m::Model)
      Fl = deepcopy(F)
      Fl.model.n = m.n
      Fl.model.d = m.d
      Fl.model.o = m.o
      for (k, v) in m.params
        Fl.model.params[k] = v
      end
      Fl
    end

    @eval function (F::pde_types)(;kwargs...)
      Fl = deepcopy(F)
      for (k, v) in kwargs
        k in keys(Fl.model.params) && Fl.model.params[k] .= v
      end
      Fl
    end
  end
end
