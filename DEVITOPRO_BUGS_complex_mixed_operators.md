# Two CUDA codegen issues in operators that mix real wavefields with complex `Function`s

Reported from the FULLEIV / JUDI on-the-fly-DFT (OTF) EIV probe. Both were hit while building an
operator that injects a frequency-domain wavefield as an on-the-fly inverse-DFT source into a real
acoustic wave equation and samples the result at receivers (`adjoint_wf_dft`).

Both are **CUDA-only** — the same operators build and run correctly on the CPU (`openmp`) backend.

Environment:

- devitopro at `/optslim/devitopro`, devito submodule at `/optslim/devitopro/submodules/devito`
- CUDA 12.9 (nvhpc 25.7), `-arch=sm_80`, A100-SXM4-40GB
- `DEVITO_LANGUAGE=cuda`, `DEVITO_PLATFORM=nvidiaX`, `DEVITO_ARCH=cuda`
- `opt=('advanced', {'index-mode': 'int64', 'errctl': 'basic', 'gpu-opt': True})`

Both have a workaround, so neither is blocking us — we now tabulate the phases and use split
real/imaginary mode fields, which avoids both. Reporting because the root causes look general:
**any** operator mixing a real `TimeFunction` with a complex `Function` will hit (1), and **any**
operator with a time-dependent, space-independent trig phase will hit (2).

---

## Bug 1 — `gpu-opt` types its staging buffer from the operator's dominant dtype, not per-field

**Symptom.** nvcc:

```
error: no suitable conversion function from "thrust::THRUST_200802_SM_800_NS::complex<float>"
       to "float" exists
  s_y0[ty + 4] = queue0[4];
```

**Cause.** The operator contains a real wavefield `v` (`float`) and a complex DFT-mode `Function`
`ufv` (`complex64`). The GPU pass stages `v`'s stencil reads through a register queue, but declares
that queue from the operator's dominant dtype:

```c
thrust::complex<float> queue0[9];                        // staging a float array
__shared__ float       s_y0[40] __attribute__ ((aligned (64)));
...
queue0[qq0 + 1] = vL0(t0, x0_blk0 + qq0 + 4, y + 8);     // v is `float *restrict v`
...
s_y0[ty + 4]    = queue0[4];                             // complex -> float: no conversion
```

`queue0` holds **`v`**, which is real; only `ufv` is complex. The equation itself is well-formed —
the source term is explicitly real-extracted and `v` is declared `float *restrict v` in the same
kernel signature.

**Expected.** The staging buffer takes the dtype of the field being staged (`float`), not the
operator's dominant dtype.

**Minimal repro** (`repro_bug1.py`): real `TimeFunction` + complex `Function`, source term
`Real(sum_f uf_f * exp(I*w_f*t))`, `opt=('advanced', {'gpu-opt': True})`.

**Workarounds, in decreasing order of preference.** All verified:

| workaround | result |
|---|---|
| no complex `Function` in the operator (split re/im) | compiles, `gpu-opt` unconstrained |
| `{'gpu-opt-steps': 1}` | compiles, `gpu-opt` otherwise on |
| `{'gpu-opt-steps': 0}` | still fails |
| `{'gpu-opt': False}` | compiles, optimization lost |

`gpu-opt-steps: 1` working while `0` fails is itself a little surprising and may be a useful clue.

---

## Bug 2 — CUDA printer emits `__device__` trig intrinsics into hoisted **host** code

**Symptom.** nvcc:

```
error: calling a __device__ function("__sincosf") from a __host__ function("adj_wf_dft")
       is not allowed
```

and with `DEVITO_SAFE_MATH=1`:

```
error: calling a __device__ function("__cosf") from a __host__ function("adj_wf_dft")
error: calling a __device__ function("__sinf")  from a __host__ function("adj_wf_dft")
```

**Cause.** The source term carries phases `cos(2*pi*f_i*time*dt)` / `sin(...)` that are
time-dependent but **space-independent**. devito hoists them out of the kernel into the host time
loop and passes them to the kernel as scalars — which is the **right** optimization:

```c
for (int time = time_M, ...; time >= time_m; time -= 1, ...)
{
  float r8 = 5.33803850412369e-2F*time*dt;
  float r25 = 0.0F, r26 = 0.0F;
  __sincosf(r8, &r25, &r26);          // <-- __device__ intrinsic in host scope
  const float r0 = r26, r1 = r25;
  ...
  kernel0<<<grid0,block0,0,qid0>>>(..., r0, r1, r2, r3, r4, r5, r6, r7, ...);
}
```

The hoisting and the scalar-argument passing are both correct and desirable. The only defect is the
**function name**: `__sincosf` / `__cosf` / `__sinf` are `__device__`-only. Host scope needs
`sincosf` / `cosf` / `sinf`.

**Not a fast-math artifact.** `--use_fast_math` comes from
`devito/arch/compiler.py:769-770` (`if not configuration['safe-math']`). Setting
`DEVITO_SAFE_MATH=1` drops the flag but only splits the fused `__sincosf` into `__cosf`/`__sinf` —
both still `__device__`. So the intrinsic mapping is applied independently of fast-math, and
independently of scope.

**Expected.** The printer selects the host-callable name when the expression is emitted in host
scope.

**Minimal repro** (`repro_bug2.py`): all-real operator whose source term is
`sum_i field_i * cos(w_i * time * dt)`, `DEVITO_LANGUAGE=cuda`.

**Workaround.** Precompute the phases into `(time, freq_dim)` tables so no trig is emitted:

```c
v[t1][x+2][y+2] = v[t0][x+2][y+2]
                + (ctab[time][0]*ufr[0][x+1][y+1] - stab[time][0]*ufi[0][x+1][y+1])/time_M + ...
```

Cheap (`nfreq*nt` floats) and strictly less work than recomputing per timestep, so we are happy with
it — but it should not be necessary.

---

## Bug 3 (minor, consequence of the other two) — `1.0_if` literal declared only when a complex `Function` exists

Trying real fields with a complex **scalar** exponential — the natural way to keep `cexpf`, which
*is* host-callable, while avoiding bug 1 — fails with:

```
error: user-defined literal operator not found
```

devito emits the imaginary unit as the C++ user-defined literal `1.0_if`, but the declaration for it
is only included when the operator contains a complex `Function`. With all-real fields and a complex
scalar, the literal is emitted without its declaration.

**Expected.** The complex-support header/declarations are emitted whenever a complex *expression*
appears, not only when a complex `Function` does.

---

## Repros

`repro_bug1.py` and `repro_bug2.py` in this directory are standalone (devito only, no JUDI). Run
with `DEVITO_LANGUAGE=cuda DEVITO_PLATFORM=nvidiaX DEVITO_ARCH=cuda`. Each prints the generated code
and then fails in `nvcc` at the line quoted above.
