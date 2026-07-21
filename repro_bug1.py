"""
devitopro CUDA repro 1 -- gpu-opt types its staging buffer from the operator's dominant dtype.

An operator mixing a REAL TimeFunction with a COMPLEX Function makes the GPU pass stage the REAL
wavefield through a `thrust::complex<float>` register queue and then assign that into a
`__shared__ float` tile:

    thrust::complex<float> queue0[9];      // staging `float *restrict v`
    __shared__ float       s_y0[40];
    s_y0[ty + 4] = queue0[4];              // no complex -> float conversion in CUDA

    error: no suitable conversion function from "thrust::complex<float>" to "float" exists

The equation is well-formed: `v` is real, the source is explicitly Real(...), and only `uf` is
complex. Expected: the staging buffer takes the dtype of the field being staged.

    DEVITO_LANGUAGE=cuda DEVITO_PLATFORM=nvidiaX DEVITO_ARCH=cuda python3 repro_bug1.py

Workarounds: remove the complex Function (split re/im); or {'gpu-opt-steps': 1} (note 0 does NOT
work); or {'gpu-opt': False}.
"""
import numpy as np
import sympy
import devito as dv
import devitopro  # noqa: F401 -- registers the nvidiaX/cuda operator
from devito import Grid, Function, TimeFunction, Eq, Operator, Real

OPT = ('advanced', {'index-mode': 'int64', 'errctl': 'basic', 'gpu-opt': True})

grid = Grid(shape=(64, 64))
time = grid.time_dim
nfreq = 4
freq = np.linspace(0.006, 0.020, nfreq)          # cyclic, model time unit

# real wavefield
v = TimeFunction(name='v', grid=grid, time_order=2, space_order=8)
# complex frequency-slice field -- the only complex object in the operator
uf = Function(name='uf', dimensions=(dv.Dimension('freq_dim'),) + grid.dimensions,
              grid=grid, shape=(nfreq,) + grid.shape, dtype=np.complex64)

dt = time.spacing
src = sum(uf._subs(uf.indices[0], i) * sympy.exp(1j * 2 * np.pi * f * time * dt)
          for i, f in enumerate(freq))

eq = Eq(v.forward, 2 * v - v.backward + v.laplace + Real(src))

op = Operator([eq], opt=OPT, name='repro_bug1')
print(op)                                          # inspect: queue0 declared thrust::complex<float>
op.cfunction                                       # triggers nvcc -> compile error
print('COMPILED (bug not reproduced)')
