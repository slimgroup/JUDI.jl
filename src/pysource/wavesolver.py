from cached_property import cached_property

from devito import TimeFunction, SparseTimeFunction, Operator, Function, Eq
from devito.tools import memoized_meth, as_tuple

from kernels import acoustic_kernel
from sensitivity import lin_src, grad_expr


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
        self.space_order = options['space_order']
        self.isic = options['isic']
        self.options = options
        self._src = SparseTimeFunction(name="src", grid=model.grid, npoint=1, nt=1)
        self._rec = SparseTimeFunction(name="rec", grid=model.grid, npoint=1, nt=1)

    @property
    def dt(self):
        return self.model.critical_dt

    @property
    def grid(self):
        return self.model.grid

    # Objects creation
    def wavefield(self, save=None, fw=True, lin=False):
        name = "u" if fw else "v"
        name = name+"l" if lin else name
        u = TimeFunction(name=name, grid=self.grid, save=save, time_order=2,
                         space_order=self.space_order)
        return u

    def qfull(self, q=False, nt=0):
        if not q:
            return None
        wf_src = TimeFunction(name='wf_src', grid=self.grid, time_order=2,
                              space_order=self.space_order, save=nt)
        return wf_src

    def lrsrc(self, esrc=False, nt=0, rec=False):
        if not esrc:
            return None, None
        r = '_r' if rec else ''
        wt = Function(name='wf_src'+r, dimensions=(self.grid.time_dim,), shape=(nt,))
        ws = Function(name='src_weight'+r, grid=self.grid, space_order=0)
        return wt, ws

    def image(self):
        return Function(name="gradm", grid=self.grid, space_order=0)

    # Data allocation
    def update_q(self, wf_src, q):
        if q is None:
            return
        if isinstance(q, TimeFunction):
            wf_src._data = q._data
        else:
            wf_src.data[:] = q[:]
        return

    def update_lr(self, wt, ws, wavelet, weights):
        if weights is not None:
            print(weights.shape)
            ws.data[:] = weights
        if wavelet is not None:
            print(wavelet.shape)
            wt.data[:] = wavelet[:, 0]
        return

    # Point source and rec
    def make_src(self, data, coords, nt):
        npoint = coords.shape[0]
        src = SparseTimeFunction(name="src", grid=self.grid,
                                 npoint=npoint, nt=nt, coordinates=coords)
        src.data[:] = data
        return src

    def make_rec(self, coords, nt):
        npoint = coords.shape[0]
        src = SparseTimeFunction(name="rec", grid=self.grid,
                                 npoint=npoint, nt=nt, coordinates=coords)                 
        return src

    # Equation creations
    def time_stepper(self, u, fw=True, q=None):
        return acoustic_kernel(self.model, u, fw=fw, q=q)

    def lr_rec(self, wt, ws, u):
        return [Eq(ws, ws + self.grid.time_dim.spacing * u * wt)]

    def source_eq(self, u, fw=True):
        next = u.forward if fw else u.backward
        return self._src.inject(next, expr=self._src*self.dt**2/self.model.m)

    def receiver_eq(self, u):
        return self._rec.interpolate(u)

    def lin_src(self, u):
        return lin_src(self.model, u, isic=self.isic)

    def g_expr(self, u, v, grad):
        return grad_expr(grad, u, v, self.model, isic=self.isic)

    # Naming utility
    def make_name(self, fw, lin, grad):
        if grad:
            return "adjoint_born"
        elif lin:
            return "born"
        return "forward" if fw else "adjoint"

    # Operator creation
    @memoized_meth
    def nlin_op(self, fw=True, save=None, wsrc=True, wrec=True, qfull=False, 
                esrc=False, erec=False, grad=False, lin=False):
        # Wavefield
        u = self.wavefield(fw=fw, save=save)
        ul = self.wavefield(fw=fw, lin=True) if lin else None
        # Time-space source
        q = self.qfull(q=qfull, nt=10) if qfull else 0
        # Rank 1 source
        wt, ws = self.lrsrc(esrc=esrc, nt=10)
        # PDE with source
        q += wt*ws if esrc else 0
        eq = self.time_stepper(u, fw=fw, q=q)
        eql = self.time_stepper(ul, q=self.lin_src(u)) if lin else []
        # Point source
        src = self.source_eq(u, fw=fw) if wsrc else []
        # Point receivers
        rec = self.receiver_eq(ul if lin else u) if wrec else []
        # Rank1 source measurment
        wtr, wsr = self.lrsrc(esrc=erec, nt=10, rec=True)
        lr_rec = self.lr_rec(wtr, wsr, ul if lin else u) if erec else []
        # gradient expression
        grad = self.image() if grad else None
        v = self.wavefield(fw=not fw, save=False) if grad else None
        g_eq = self.g_expr(v, u, grad) if grad else []
        # Spacing substitutions
        subs = self.model.spacing_map
        # Devito operator
        name = self.make_name(fw, lin, grad)
        op = Operator(eq + src + eql + rec + lr_rec + g_eq, subs=subs, name=name)
        # Precompile
        op.cfunction
        return op

    # Filter which output to return
    def filter_output(self, rec, u, weights, grad):
        if grad is not None:
            return grad
        elif rec is not None:
            return rec
        elif weights is not None:
            return weights
        return u

    # Non linear propagation
    def propagate(self, fw=True, save=False, wavelet=None, src_coords=None,
                  rec_coords=None, q=None, w=None, ws=None, wr=None, u=None,
                  gu=None, grad=False, lin=False):
        # Number of tim-steps
        nt = wavelet.shape[0] if wavelet is not None else as_tuple(q)[0].shape[0]
        save = nt if save else None
        # wavefield
        u = u or self.wavefield(fw=fw, save=save)
        ul = self.wavefield(lin=True) if lin else None
        # Get flags
        wsrc = src_coords is not None
        wrec = rec_coords is not None
        esrc = ws is not None
        erec = wr is not None
        qfull = q is not None
        # Make devito src/rec
        src = self.make_src(wavelet, src_coords, nt) if wsrc else None
        rec = self.make_rec(rec_coords, nt) if wrec else None
        # Full wavefield src
        qsrc = self.qfull(q=qfull, nt=nt)
        self.update_q(qsrc, q)
        # Rank 1 source
        wt, ws = self.lrsrc(esrc=esrc, nt=nt)
        self.update_lr(wt, ws, ws, w)
        # Rank 1 rec
        wtr, wsr = self.lrsrc(esrc=erec, nt=nt, rec=True)
        self.update_lr(wtr, wsr, wr, None)
        # Gradient
        grad = self.image() if grad else None
        # kwargs
        fields = [u, ul, src, rec, qsrc, wt, ws, wtr, wsr, gu, grad]
        kw = {k.name: k for k in fields if k is not None}
        kw.update({'dt': self.dt})
        op = self.nlin_op(fw=fw, save=save, wsrc=wsrc, wrec=wrec, lin=lin,
                          qfull=qfull, esrc=esrc, erec=erec, grad=grad)
        op.apply(**kw)
        out = self.filter_output(rec, u, wsr, grad)
        return out

    def forward(self, **kwargs):
        ret = self.propagate(fw=True, **kwargs)
        return ret.data

    def adjoint(self, **kwargs):
        ret = self.propagate(fw=False, **kwargs)
        return ret.data

    def born(self, **kwargs):
        ret = self.propagate(fw=True, lin=True, **kwargs)
        return ret.data

    def adjoint_born(self, wavelet=None, src_coords=None, rec_coords=None, rec_data=None):
        nt = wavelet.shape[0] if wavelet is not None else None
        # Forward modeling
        u = self.wavefield(save=nt)
        _ = self.propagate(fw=True, save=nt, wavelet=wavelet, src_coords=src_coords,
                           rec_coords=rec_coords, u=u)
        # Compute residual and compute gradient
        grad = self.propagate(fw=False, save=False, wavelet=rec_data, src_coords=rec_coords,
                              gu=u, grad=True)
        return grad.data
