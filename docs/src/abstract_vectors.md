# Abstract Vectors

JUDI provides abstract vector types that encapsulate seismic related objects. In particula, JUDI defines thre main types for seismic data [`judiVector`](@ref), full time-space wavefields [`judiWavefield`](@ref) and extended source weights [`judiWeights`](@ref).

```@contents
Pages = ["abstract_vectors.md"]
```

At the core of JUDI's vector types is the abstract type `judiMultiSourceVector` that represent a dimensionalized `Vector{Array}` where each sub-array correspond to a single source. All JUDI vector types inhert from this abstract type that implements most of the arithmetic and julia core utilities. As an abstract types, `judiMultiSourceVector` should not be instantiated but new concrete types based on it should be created if one desire to create its own JUDI multi-source vector type.

All sub-type of `judiMultiSourceVector` must implement the following methods to be compatible with JUDI. The following JUDI core types are examples of sub-types.

## judiVector

The class `judiVector` is the basic data structure for seismic shot records or seismic sources. From JUDI's perspective, both are treated the same and can be multiplied with modeling operators.

**Construction:**

In the most basic way, `judiVectors` are contstructed from a `Geometry` object (containing either source or receiver geometry) and a cell array of data:

```@docs
judiVector(geometry::Geometry, data::Array{T, N}) where {T, N}
```

**Access fields (in-core data containers):**

```julia
# Access i-th shot record
x.data[i]

# Extract judiVector for i-th shot
x1 = x[i]

# Access j-th receiver location of i-th shot
x.geometry.xloc[i][j]
```

**Access fields (out-of-core data containers):**

```julia
# Access data container of i-th shot
x.data[i]

# Read data from i-th shot into memory
x.data[i][1].data

# Access out-of-core geometry
x.geometry

# Load OOC geometry into memory
Geometry(x.geometry)
```

**Operations:**

In-core `judiVectors` can be used like regular Julia arrays and support common operations such as:


```julia
x = judiVector(geometry, data)

# Size (as if all data was vectorized)
size(x)

# Norms
norm(x)

# Inner product
dot(x, x)

# Addition, subtraction (geometries must match)
y = x + x
z = x - y

# Scaling
α = 2f0
y = x * α

# Concatenate
y = vcat(x, x)
```


## judiWavefield

Abstract vector class for wavefields. 

**Construction:**

```@docs
 judiWavefield
```

**Access fields:**

```julia
# Access wavefield from i-th shot location
u.data[i]
```

**Operations:**

Supports some basic arithmetric operations:

```julia
# Size 
size(u)

# Norms
norm(u)

# Inner product 
dot(u, y)

# Addition, subtraction
v = u + u
z = u - v

# Absolute value
abs(u)

# Concatenation
v = vcat(u, u)
```

## judiRHS

Abstract vector class for a right-hand-side (RHS). A RHS has the size of a full wavefield, but only contains the data of the source wavelet of shot records in memory, as well as the geometry information of where the data is injected during modeling.

**Construction:**

```julia
rhs = judiRHS(geometry, data)
```

A JUDI RHS can also be constructed by multplying a `judiVector` and the corresponding transpose of a `judiProjection` operator:

```julia
rhs1 = Ps'*q
rhs2 = Pr'*d_obs
```

where `Ps` and `Pr` are `judiProjection` operators for sources and receivers respectively and `q` and `d_obs` are `judiVectors` with the source and receiver data.

 **Access fields:**

Accessible fields include:

```julia
# Source/receiver data
rhs.data

# Source/receiver geometry
rhs.geometry

# Info structure
rhs.info
```

## judiWeights

Abstract vector class for extended source weights. The weights for each shot location have the dimensions of the model (namely `model.n`).

**Construction:**

```@docs
judiWeights(weights::Array{T, N}; nsrc=1, vDT::DataType=Float32) where {T<:Real, N}
```

**Parameters:**

 * `weights`: Cell array with one cell per shot location. Each cell contains a 2D/3D Julia array with the weights for the spatially extended source. Alternatively: pass a single Julia array which will be used for all source locations.

**Access fields:**

```julia
# Access weights of i-th shot locatoin
w.weights[i]
```

**Operations:**

Supports the same arithmetric operations as a `judiVector`.
