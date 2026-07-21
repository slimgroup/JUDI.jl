"""
devitopro CUDA repro 2 -- __device__ trig intrinsics emitted into hoisted HOST code.

Every field here is REAL. The source term carries phases cos(2*pi*f_i*time*dt) that are
time-dependent but space-INdependent, so devito hoists them out of the kernel into the host time
loop and passes them to the kernel as scalars. That optimization is correct and desirable; the
defect is only the emitted NAME -- __sincosf / __cosf / __sinf are __device__-only:

    for (int time = time_M, ...; time >= time_m; time -= 1, ...)
    {
      float r8 = 5.33e-2F*time*dt;
      __sincosf(r8, &r25, &r26);        // <-- __device__ intrinsic in host scope
      kernel0<<<...>>>(..., r0, r1, ...);
    }

    error: calling a __device__ function("__sincosf") from a __host__ function is not allowed

Host scope needs sincosf / cosf / sinf.

NOT a fast-math artifact: --use_fast_math comes from devito/arch/compiler.py:769-770
(`if not configuration['safe-math']`). DEVITO_SAFE_MATH=1 drops the flag but only splits the fused
__sincosf into __cosf/__sinf -- both still __device__.

    DEVITO_LANGUAGE=cuda DEVITO_PLATFORM=nvidiaX DEVITO_ARCH=cuda python3 repro_bug2.py
    DEVITO_SAFE_MATH=1 DEVITO_LANGUAGE=cuda ... python3 repro_bug2.py    # __cosf/__sinf instead

Workaround: tabulate the phases over (time, freq_dim) so no trig is emitted at all.
"""
import numpy as np
import devito as dv
import devitopro  # noqa: F401 -- registers the nvidiaX/cuda operator
from devito import Grid, Function, TimeFunction, Eq, Operator, cos, sin

OPT = ('advanced', {'index-mode': 'int64', 'errctl': 'basic', 'gpu-opt': True})

grid = Grid(shape=(64, 64))
time = grid.time_dim
nfreq = 4
freq = np.linspace(0.006, 0.020, nfreq)

v = TimeFunction(name='v', grid=grid, time_order=2, space_order=8)
fd = dv.Dimension('freq_dim')
# split real/imaginary frequency slices -- ALL REAL, no complex anywhere in this operator
ufr = Function(name='ufr', dimensions=(fd,) + grid.dimensions, grid=grid,
               shape=(nfreq,) + grid.shape, dtype=np.float32)
ufi = Function(name='ufi', dimensions=(fd,) + grid.dimensions, grid=grid,
               shape=(nfreq,) + grid.shape, dtype=np.float32)

dt = time.spacing
src = 0
for i, f in enumerate(freq):
    wt = 2 * np.pi * f * time * dt
    src += ufr._subs(fd, i) * cos(wt) - ufi._subs(fd, i) * sin(wt)

eq = Eq(v.forward, 2 * v - v.backward + v.laplace + src)

op = Operator([eq], opt=OPT, name='repro_bug2')
print(op)                                          # inspect: __sincosf in the host time loop
op.cfunction                                       # triggers nvcc -> compile error
print('COMPILED (bug not reproduced)')
