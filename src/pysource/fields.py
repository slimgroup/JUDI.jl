import numpy as np

from devito import (TimeFunction, ConditionalDimension, Function,
                    DefaultDimension, Dimension, VectorTimeFunction,
                    TensorTimeFunction, Buffer)
from devito.builtins import initialize_function
from devito.tools import as_tuple

try:
    import devitopro as dvp  # noqa
except ImportError:
    import devito as dvp  # noqa

from utils import compression_mode


def wavefield(model, space_order, save=False, nt=None, fw=True, name='', t_sub=1,
              tfull=False):
    """
    Create the wavefield for the wave equation

    Parameters
    ----------

    model : Model
        Physical model
    space_order: int
        Spatial discretization order
    save : Bool
        Whether or not to save the time history
    nt : int (optional)
        Number of time steps if the wavefield is saved
    fw : Bool
        Forward or backward (for naming)
    name: string
        Custom name attached to default (u+name)
    tfull: Bool
        Whether need full buffer for e.g. second time derivative
    """
    name = "u"+name if fw else "v"+name
    save = False if t_sub > 1 else save
    nsave = Buffer(3 if tfull else 2) if not save else nt

    if model.is_tti:
        u = TimeFunction(name="%s1" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=nsave)
        v = TimeFunction(name="%s2" % name, grid=model.grid, time_order=2,
                         space_order=space_order, save=nsave)
        return (u, v)
    elif model.is_elastic:
        v = VectorTimeFunction(name="v", grid=model.grid, time_order=1,
                               space_order=space_order, save=Buffer(1))
        tau = TensorTimeFunction(name="tau", grid=model.grid, time_order=1,
                                 space_order=space_order, save=Buffer(1))
        return (v, tau)
    else:
        return TimeFunction(name=name, grid=model.grid, time_order=2,
                            space_order=space_order, save=nsave)


def forward_wavefield(model, space_order, save=True, nt=10, dft=False, t_sub=1, fw=True):
    """
    Return the wavefield to be used in the gradient calculations depending on the options.

    Parameters
    ----------

    model : Model
        Physical model
    space_order : int
        Spatial discretization order
    nt : int
        Number of time steps on original time axis
    t_sub : int
        Factor for time-subsampling
    dft: Bool
        Whether to use on the fly dft
    """
    u = wavefield(model, space_order, save=save, nt=nt, t_sub=t_sub, fw=fw)
    if dft:
        return fourier_modes(u, np.ones((10,)))[0]
    elif t_sub > 1:
        return wavefield_subsampled(model, u, nt, t_sub, space_order)
    else:
        return u


def src_wavefield(model, u, fw=True):
    """
    Full time-space wavefield to be used as a source during propagation.

    Parameters
    ----------

    model : Model
        Physical model
    u : TimeFunction or Array
        Data for the TimeFunction
    fw : Bool
        Forward or backward (for naming)
    """
    name = "uqwf" if fw else "vqwf"
    init = u.data if isinstance(u, TimeFunction) else u.to_numpy(copy=False)
    wf_src = TimeFunction(name=name, grid=model.grid, time_order=2,
                          space_order=0, save=u.shape[0], initializer=init)
    return wf_src


def memory_field(p):
    """
    Memory variable for viscosity modeling.

    Parameters
    ----------

    p : TimeFunction
        Forward wavefield
    """
    return TimeFunction(name='r%s' % p.name, grid=p.grid, time_order=2,
                        space_order=p.space_order, save=Buffer(2))


def wavefield_subsampled(model, u, nt, t_sub, space_order=8):
    """
    Create a subsampled wavefield

    Parameters
    ----------

    model : Model
        Physical model
    u : TimeFunction
        Forward wavefield for modeling
    nt : int
        Number of time steps on original time axis
    t_sub : int
        Factor for time-subsampling
    space_order: int
        Spatial discretization order
    """
    if t_sub > 1:
        time_subsampled = ConditionalDimension(name='t_sub', parent=model.grid.time_dim,
                                               factor=t_sub)
        nsave = int(np.ceil((nt + t_sub)/t_sub))
    else:
        return None
    wf_s = []
    for wf in as_tuple(u):
        usave = dvp.TimeFunction(name='us_%s' % wf.name, grid=model.grid, time_order=2,
                                 space_order=space_order, time_dim=time_subsampled,
                                 save=nsave, compression=compression_mode())
        wf_s.append(usave)
    return wf_s


def lr_src_fields(model, weight, wavelet, empty_w=False, rec=False):
    """
    Extended source for modeling where the source is the outer product of
    a spatially varying weight and a time-dependent wavelet i.e.:
    u.dt2 - u.laplace = w(x)*q(t)
    This function returns the extended source w(x)*q(t)

    Parameters
    ----------
    model: Model
        Physical model structure
    weight: Array
        Array of weight for the spatial Function
    wavelet: Array
        Time-series for the time-varying source
    q: Symbol or Expr (optional)
        Previously existing source to be added to (source will be q +  w(x)*q(t))
    """
    if (weight is None and not empty_w) or wavelet is None:
        return None, None
    time = model.grid.time_dim
    nt = wavelet.shape[0]
    wn = 'rec' if rec else 'src'
    wavelett = TimeFunction(name='wf_%s' % wn, dimensions=(time,), time_dim=time,
                            shape=(nt,), save=nt, grid=model.grid)
    wavelett.data[:] = np.array(wavelet)[:, 0]
    if isinstance(weight, Function):
        source_weight = weight
    else:
        source_weight = Function(name='%s_weight' % wn, grid=model.grid, space_order=0)
        if not empty_w:
            initialize_function(source_weight, weight.to_numpy(copy=False), 0)
    return source_weight, wavelett


def frequencies(freq, fdim=None):
    """
    Frequencies as a one dimensional Function

    Parameters
    ----------
    freq: List or 1D array
        List of frequencies
    """
    if freq is None:
        return None, 0
    nfreq = np.shape(freq)[0]
    freq_dim = fdim or DefaultDimension(name='freq_dim', default_value=nfreq)
    f = Function(name='f', dimensions=(freq_dim,), shape=(nfreq,))
    f.data[:] = np.array(freq[:])
    return f, nfreq


def trig_tables(freq_dim, time, freq, nt, dt):
    """
    Precomputed cos/sin tables for the on-the-fly inverse DFT: `ct[t, i] = cos(2 pi f_i t dt)`,
    `st[t, i] = sin(...)`, as REAL Functions over `(time, freq_dim)`.

    This removes trigonometry from the generated code entirely -- the idft source becomes pure loads
    and multiply-adds, `ctab[time][i]*ufr[i][x][y] - stab[time][i]*ufi[i][x][y]`.

    WHY. The phases are time-dependent but space-independent, so devito hoists them out of the kernel
    into the host time loop (correct -- they become scalar kernel arguments). But its CUDA printer
    maps `cos`/`sin` to the __device__ intrinsics `__cosf`/`__sinf` regardless of scope, so the
    hoisted host code does not compile:
        error: calling a __device__ function("__cosf") from a __host__ function
    Tabulating sidesteps the printer completely. It is also cheap and strictly less work than
    recomputing the phases every timestep: the tables are `nfreq x nt` floats -- ~64 KB at
    nfreq=8, nt=2001, and still negligible in 3D, where the wavefield is the only thing that scales.

    `freq_dim` MUST be the same Dimension object the mode fields use, or devito sees two distinct
    dimensions and the subs below will not line up.
    """
    nfreq = len(freq)
    ct = Function(name='ctab', dimensions=(time, freq_dim), shape=(nt, nfreq), dtype=np.float32)
    st = Function(name='stab', dimensions=(time, freq_dim), shape=(nt, nfreq), dtype=np.float32)
    ph = (2*np.pi*np.asarray(freq, dtype=np.float64)[None, :] *
          (np.arange(nt, dtype=np.float64)[:, None] * float(dt)))
    ct.data[:] = np.cos(ph).astype(np.float32)
    st.data[:] = np.sin(ph).astype(np.float32)
    return ct, st


def fourier_modes_real(u, freq):
    """
    Frequency-slice fields as SPLIT REAL/IMAGINARY pairs instead of one complex Function.

    Returns `(modes_re, modes_im, f)` with the same dimensions/shape as `fourier_modes`, but
    `dtype=np.float32`, so an operator built on them contains NO complex Function at all.

    Used by `adjoint_wf_dft` under JUDI_REAL_MODES=1. It DOES remove the complex-Function typing
    problem -- devitopro's `gpu-opt` types its register staging buffers from the operator's dominant
    dtype rather than per-field, so one complex Function makes it stage a REAL wavefield through a
    `thrust::complex<float>` queue and assign that into a `__shared__ float` tile, which nvcc
    rejects. With all-real fields that error is gone: the generated CUDA declares
    `float *d_ufrv, *d_ufiv, *d_v` and contains no thrust type at all.

    WARNING -- this does NOT currently unlock GPU; it trades one devito codegen bug for another.
    With real fields the scalar cos/sin print as the __device__ intrinsics __cosf/__sinf REGARDLESS
    of scope. devito hoists these time-dependent, space-independent scalars out of the kernel into
    the host time loop (correctly -- they become scalar kernel arguments), and a __device__ intrinsic
    there is a compile error: `calling a __device__ function("__cosf") from a __host__ function`.
    Independent of fast-math: DEVITO_SAFE_MATH=1 only splits __sincosf into __cosf/__sinf. A hybrid
    (real fields + complex scalar exp, whose cexpf IS host-callable) fails a THIRD way: devito emits
    the `1.0_if` imaginary literal but declares it only when a complex Function is present, giving
    "user-defined literal operator not found". So on GPU today the working configuration is COMPLEX
    modes + JUDI_GPU_OPT_STEPS=1, which is the DEFAULT. Keep this path for CPU correctness and make
    it the default once devito's printer is scope-aware -- it is the formulation that would let
    `gpu-opt` run unconstrained.

    `fourier_modes` is deliberately left untouched: it is shared with the gradient/FWI path, whose
    GPU workflows already work.

    Parameters
    ----------
    u: TimeFunction or Tuple
        Wavefield the modes are attached to
    freq: Array
        Array of frequencies for on-the-fly DFT
    """
    if freq is None:
        return None, None, None

    f, nfreq = frequencies(freq)
    freq_dim = f.dimensions[0]

    modes_re, modes_im = [], []
    for wf in as_tuple(u):
        for name, acc in (('ufr', modes_re), ('ufi', modes_im)):
            acc.append(Function(name='%s%s' % (name, wf.name),
                                dimensions=(freq_dim,) + wf.indices[1:],
                                grid=wf.grid, shape=(nfreq,) + wf.shape[1:],
                                dtype=np.float32))
    return as_tuple(modes_re), as_tuple(modes_im), f


def fourier_modes(u, freq):
    """
    On the fly DFT wavefield (frequency slices) and expression

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield
    freq: Array
        Array of frequencies for on-the-fly DFT
    """
    if freq is None:
        return None, None

    # Frequencies
    f, nfreq = frequencies(freq)
    freq_dim = f.dimensions[0]

    dft_modes = []
    for wf in as_tuple(u):
        uf = Function(name='uf%s' % wf.name, dimensions=(freq_dim,) + wf.indices[1:],
                      grid=wf.grid, shape=(nfreq,) + wf.shape[1:],
                      dtype=np.complex64)
        dft_modes.append(uf)
    return as_tuple(dft_modes), f


def norm_holder(v):
    """
    Single element function to compute the norm of an input TimeFunction.

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield
    """
    v0 = as_tuple(v)[0]
    i = Dimension(name="i",)
    nv = Function(name="n%s" % v0.name, shape=(1,), dimensions=(i,), grid=v0.grid)
    nvt = Function(name="n%st" % v0.name, grid=v0.grid, space_order=0)
    return nv, nvt


def illumination(u, illum):
    """
    Function for the wavefield illumination

    Parameters
    ----------
    u: TimeFunction or Tuple
        Forward wavefield
    illum: bool
        Whether to compute the illumination flag
    """
    if not illum:
        return None
    u0 = as_tuple(u)[0]
    return Function(name="I%s" % u0.name, grid=u0.grid, space_order=0)
