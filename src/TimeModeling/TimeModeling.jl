# Functions for time-domain modeling and inversion using devito
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January, 2017
# Updated, December 2020, Mathias Louboutin, mlouboutin3@gatech.edu

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
include("LinearOperators/basics.jl")
include("LinearOperators/lazy.jl")
include("LinearOperators/operators.jl")

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

#############################################################################
if VERSION>v"1.2"
  function (F::judiPropagator)(m::AbstractModel)
    Fl = deepcopy(F)
    Fl.model.n = m.n
    Fl.model.d = m.d
    Fl.model.o = m.o
    for (k, v) in m.params
      Fl.model.params[k] = v
    end
    _track_illum(m, Fl.model)
    Fl
  end

  function (F::judiPropagator)(;kwargs...)
    Fl = deepcopy(F)
    for (k, v) in kwargs
      k in keys(Fl.model.params) && Fl.model.params[k] .= v
    end
    Fl
  end

  function (F::judiPropagator)(m, q)
    Fm = F(;m=m)
    _track_illum(F.model, Fm.model)
    return Fm*as_src(q)
  end

  function (F::judiPropagator)(m::AbstractArray)
    @info "Assuming m to be squared slowness for $(typeof(F))"
    return F(;m=m)
  end

  (F::judiPropagator)(m::AbstractModel, q) = F(m)*as_src(q)
  
  function (J::judiJacobian{D, O, FT})(q::judiMultiSourceVector) where {D, O, FT}
    newJ = judiJacobian{D, O, FT}(J.m, J.n, J.F, q)
    _track_illum(J.model, newJ.model)
    return newJ
  end

  function (J::judiJacobian{D, O, FT})(x::Array{D, N}) where {D, O, FT, N}
    if length(x) == prod(J.model.n)
      return J(;m=m)
    end
    new_q = _as_src(J.qInjection.op, J.model, x)
    newJ = judiJacobian{D, O, FT}(J.m, J.n, J.F, new_q)
    _track_illum(J.model, newJ.model)
    return newJ
  end

end
