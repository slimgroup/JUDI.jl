from cached_property import cached_property

from devito import TimeFunction, SparseTimeFunction, Operator
from devito.tools import memoized_meth

from kernels import acoustic_kernel

class WaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    """

    def __init__(self, model, **options):
        self.model = model
        self.options = options
        self._src = SparseTimeFunction(name="src", grid=self.model.grid, npoint=1, nt=1)
        self._rec = SparseTimeFunction(name="rec", grid=self.model.grid, npoint=1, nt=1)

    @property
    def dt(self):
        return self.model.critical_dt

    # Objects creation
    @memoized_meth
    def wavefield(self, save=False):
        u = TimeFunction(name='u', grid=self.model.grid, save=save, time_order=2,
                         space_order=self.options['space_order'])
        return u

    def make_src(self, data, coords, nt):
        npoint = coords.shape[0]
        src = SparseTimeFunction(name="src", grid=self.model.grid,
                                 npoint=npoint, nt=nt)
        src.data[:] = data
        return src
    
    def make_rec(self, coords, nt):
        npoint = coords.shape[0]
        src = SparseTimeFunction(name="rec", grid=self.model.grid,
                                 npoint=npoint, nt=nt)
        return src

    # Equation creations
    def time_stepper(self, u):
        return acoustic_kernel(self.model, u)

    def source_eq(self, u):
        return self._src.inject(u, expr=self._src*self.dt**2/self.model.m)
    
    def receiver_eq(self, u):
        return self._rec.interpolate(u)

    # Operator creation
    @memoized_func
    def fwd_op(self, save=False, wsrc=True, wrec=True):
        u = self.wavefield(save=save)
        eq = self.time_stepper(u)
        src = self.source_eq(u) if wsrc else []
        rec = self.receiver_eq(u) if wrec else []
        subs = self.model.spacing_map
        op = Operator(eq + src + rec, subs=subs)
        op.cfunction
        return op

    # Call
    def forward(self, save=False, wavelet=None, src_coords=None, rec_coords=None):
        # Number of tim-steps
        nt = wavelet.shape[0] if wavelet is not None else None
        u = self.wavefield(save=nt if save else None)
        u.data.fill(0.0)
        # Get flags
        wsrc = self._src is not None
        wrec = self._rec is not None
        # Make devito src/rec
        src = self.make_src(wavelet, src_coords, nt) if wsrc else None
        rec = self.make_rec(rec_coords, nt) if wrec else None
        # kwargs
        kw = {k.name: k for k in [u, src, rec] if k is not None}
        self.fwd_op(save=save, wsrc=wsrc, wrec=wrec).apply(**kw)
        return self._rec.data
