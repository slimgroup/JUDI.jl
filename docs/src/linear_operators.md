# Linear Operators

JUDI is building on [JOLI.jl](https://github.com/slimgroup/JOLI.jl) to implement matrix-free linear operators. These operators represent the discretized wave-equations and sensitivit (Jacobian) for different acquisition schemes. 

```@contents
Pages = ["linear_operators.md"]
```

## judiModeling

Seismic modeling operator for solving a wave equation for a given right-hand-side.

```@docs
judiModeling{DDT, RDT}
```

**Construction:**

 * Construct a modeling operator **without** source/receiver projections:

```julia
F = judiModeling(info, model; options=opt)
```

 * Construct a modeling operator **with** source/receiver projections:

```julia
F = judiModeling(info, model, src_geometry, rec_geometry)
```

Not that in this case, all the information can be infered from the model and geometries. Therefore the operator can be created without the need for the `info` input:


```julia
F = judiModeling(model, src_geometry, rec_geometry)
```

 * Construct a modeling operator from an **existing** operator without geometries and projection operators:

```julia
F = Pr*F*Ps'
```

where `Ps` and `Pr` are source/receiver projection operators of type `judiProjection`.

 * Construct a modeling operator for **extended source modeling**:

```julia
F = Pr*F*Pw'
```

where `Pw` is a `judiLRWF` (low-rank-wavefield) projection operator.


```@docs
judiModeling(info::Info, model::Model; options=Options(), DDT::DataType=Float32, RDT::DataType=DDT)
```

**Accessible fields:**

```julia
# Info structure
F.info

# Model structure
F.model

# Source geometry (if available)
F.srcGeometry

# Receiver geometry (if available)
F.recGeometry

# Options structure
F.options
```

**Usage:**

```julia
# Forward modeling (F w/ geometries)
d_obs = F*q

# Adjoint modeling (F w/ geometries)
q_ad = F'*d_obs

# Forward modeling (F w/o geometries)
d_obs = Pr*F*Ps'*q

# Adjoint modelng (F w/o geometries)
q_ad = Ps*F'*Pr'*d_obs

# Extended source modeling (F w/o geometries)
d_obs  = Pr*F*Pw'*w

# Adjoint extended source modeling (F w/o geometries)
w_ad = Pw*F'*Pr'*d_obs

# Forward modeling and return full wavefield (F w/o geometries)
u = F*Ps'*q

# Adjoint modelnig and return wavefield (F w/o geometries)
v = F'*Pr'*d_obs

# Forward modeling with full wavefield as source (F w/o geometries)
d_obs = Pr*F*u

# Adjoint modeling with full wavefield as source (F w/o geometries)
q_ad = Ps*F*v
```


## judiJacobian

Jacobian of a non-linear forward modeling operator. Corresponds to linearized Born modeling (forward mode) and reverse-time migration (adjoint mode).

```@docs
judiJacobian
```

**Construction:**

 * A `judiJacobian` operator can be create from an exisiting forward modeling operator and a source vector:

```julia
J = judiJacobian(F, q)  # F w/ geometries
```

```julia
J = judiJacobian(Pr*F*Ps', q)   # F w/o geometries
```

where `Ps` and `Pr` are source/receiver projection operators of type `judiProjection`.

 * A Jacobian can also be created for an extended source modeling operator:

```julia
J = judiJacobian(Pr*F*Pw', w)
```

where `Pw` is a `judiLRWF` operator and `w` is a `judiWeights` vector (or 2D/3D Julia array).


**Accessible fields::**

```julia
# Info structure
J.info

# Model structure
J.model

# Source geometry (if available)
J.srcGeometry

# Receiver geometry
J.recGeometry

# Source wavelet
J.wavelet

# Weights (extended source modeling only)
J.weights

# Options structure
J.options
```

**Usage:**

```julia
# Linearized modeilng
d_lin = J*dm

# RTM
rtm = J'*d_lin

# Matrix-free normal operator
H = J'*J
```

## judiProjection

Abstract linear operator for source/receiver projections. A (transposed) `judiProjection` operator symbolically injects the data with which it is multiplied during modeling. If multiplied with a forward modeling operator, it samples the wavefield at the specified source/receiver locations.

```@docs
judiProjection
```

**Accessible fields:**

```julia
# Info structure
P.info

# Source/receiver geometry
P.geometry
```

**Usage:**

```julia
# Multiply with judiVector to create a judiRHS
rhs1 = Pr'*d_obs
rhs2 = Ps'*q

# Sample wavefield at source/receiver location during modeling
d_obs = Pr*F*Ps'*q
q_ad = Ps*F*Pr'*d_obs
```

## judiLRWF


Abstract linear operator for sampling a seismic wavefield as a sum over all time steps, weighted by a time-varying wavelet. Its transpose *injects* a time-varying wavelet at every grid point in the model.

```@docs
judiLRWF
```

**Accessible fields:**

```julia
# Info structure
P.info

# Wavelet of i-th source location
P.wavelet[i]
```

**Usage:**

```julia
# Multiply with a judiWeight vector to create a judiExtendedSource
ex_src = Pw'*w

# Sample wavefield as a sum over time, weighted by the source
u_ex = Pw*F'*Pr'*d_obs
```
