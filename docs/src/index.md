# The Julia Devito Inversion framework (JUDI.jl)

JUDI is a framework for large-scale seismic modeling and inversion and designed to enable rapid translations of algorithms to fast and efficient code that scales to industry-size 3D problems. Wave equations in JUDI are solved with [Devito](https://www.devitoproject.org/), a Python domain-specific language for automated finite-difference (FD) computations. 

## Docs overview

This documentation provides an overview over JUDI's basic data structures and abstract operators:

 * [Installation](@ref): Install guidlines for JUDI and compilers.

 * [Getting Started](@ref): A few simple guided examples to get familiar with JUDI.

 * [Data structures](@ref): Explains the `Model`, `Geometry` and `Options` data structures and how to set up acquisition geometries.

 * [Abstract Vectors](@ref): Documents JUDI's abstract vector classes `judiVector`, `judiWavefield`, `judiRHS`, `judiWeights` and `judiExtendedSource`.

 * [Linear Operators](@ref): Lists and explains JUDI's abstract linear operators `judiModeling`, `judiJacobian`, `judiProjection` and `judiLRWF`.

 * [Input/Output](@ref): Read SEG-Y data and set up `judiVectors` for shot records and sources. Read velocity models.

 * [Helper functions](@ref): API of functions that make your life easier.

 * [Seismic Inversion](@ref): Inversion utility functions to avoid recomputation and memry overhead.

 * [Seismic Preconditioners](@ref): Basic preconditioners for seismic imaging.

 * [pysource package](@ref): API reference for the propagators implementation with Devito in python. The API is the backend of JUDI handled with PyCall.