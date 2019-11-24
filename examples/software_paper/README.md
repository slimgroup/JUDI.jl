# A large-scale framework for symbolic implementations of seismic inversion algorithms in Julia

## Overview

This repository contains instructions and the scripts to reproduce the examples from the paper ["A large-scale framework for symbolic implementations of seismic inversion algorithms in Julia" (Witte et al, 2019)](https://library.seg.org/doi/abs/10.1190/geo2018-0174.1). Running the examples requires Julia (version 1.1.0) and the JUDI package. Follow the instructions from the [main page](https://github.com/slimgroup/JUDI.jl) to install JUDI and its required packages. For questions, contact Philipp Witte at pwitte3@gatech.edu.

## Abstract

Writing software packages for seismic inversion is a very challenging task because problems such as full-waveform inversion or least-squares imaging are algorithmically and computationally demanding due to the large number of unknown parameters and the fact that waves are propagated over many wavelengths. Therefore, software frameworks need to combine versatility and performance to provide geophysicists with the means and flexibility to implement complex algorithms that scale to exceedingly large 3D problems. Following these principles, we have developed the Julia Devito Inversion framework, an open-source software package in Julia for large-scale seismic modeling and inversion based on Devito, a domain-specific language compiler for automatic code generation. The framework consists of matrix-free linear operators for implementing seismic inversion algorithms that closely resemble the mathematical notation, a flexible resilient parallelization, and an interface to Devito for generating optimized stencil code to solve the underlying wave equations. In comparison with many manually optimized industry codes written in low-level languages, our software is built on the idea of independent layers of abstractions and user interfaces with symbolic operators. Through a series of numerical examples, we determined that this allows users to implement a series of increasingly complex algorithms for waveform inversion and imaging as simple Julia scripts that scale to large-scale 3D problems. This illustrates that software based on the paradigms of abstract user interfaces and automatic code generation and makes it possible to manage the complexity of the algorithms and performance optimizations, thus providing a high-performance research and production framework.


## References

The reproducible examples on this page are featured in the following journal publication:

 * Philipp A. Witte, Mathias Louboutin, Navjot Kukreja, Fabio Luporini, Michael Lange, Gerard J. Gorman and Felix J. Herrmann. A large-scale framework for symbolic implementations of seismic inversion algorithms in Julia. GEOPHYSICS, Vol. 84 (3), pp. F57-F71, 2019. <https://library.seg.org/doi/abs/10.1190/geo2018-0174.1>

 * Copyright (c) 2019 Geophysics

 * DOI: 10.1190/geo2018-0174.1

Contact authors via: pwitte3@gatech.edu and mlouboutin3@gatech.edu.
