# Compressive least squares migration with on-the-fly Fourier transforms

## Overview

## Abstract

Least-squares seismic imaging is an inversion-based approach for accurately imaging the earth's subsurface. However, in the time-domain, the computational cost and memory requirements of this approach scale with the size and recording length of the seismic experiment, thus making this approach often prohibitively expensive in practice. To overcome these issues, we borrow ideas from compressive sensing and signal processing and introduce an algorithm for sparsity-promoting seismic imaging using on-the-fly Fourier transforms. By computing gradients and functions values for random subsets of source locations and frequencies, we considerably limit the number of wave equation solves, while on-the-fly Fourier transforms allow computing an arbitrary number of monochromatic frequency-domain wavefields with a time-domain modeling code and without having to solve large-scale Helmholtz equations. The memory requirements of this approach are independent of the number of time steps and solely depend on the number of frequencies, which determine the amount of crosstalk and subsampling artifacts in the image. We show the application of our approach to several large-scale open source data sets and compare the results to a conventional time-domain approach with optimal checkpointing.

## Obtaining the velocity models

## Sigsbee 2A example

#### Figure: {#f1}
![](Sigsbee2A/figure1.png){width=80%}

#### Figure: {#f1}
![](Sigsbee2A/figure2.png){width=80%}


## BP Synthetic 2004 example


#### Figure: {#f1}
![](BP_synthetic_2004/figure1.png){width=80%}

#### Figure: {#f1}
![](BP_synthetic_2004/figure2.png){width=80%}


## References

The reproducible examples on this page are featured in the following journal publication:

 * Philipp A. Witte, Mathias Louboutin, Fabio Luporini, Gerard J. Gorman and Felix J. Herrmann. Compressive least-squares migration with on-the-fly Fourier transforms. GEOPHYSICS. 2019. Just-Accepted-Articles. <https://library.seg.org/doi/abs/10.1190/geo2018-0490.1>

Contact authors via: pwitte3@gatech.edu and mlouboutin3@gatech.edu.
