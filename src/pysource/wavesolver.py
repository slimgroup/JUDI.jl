from cached_property import cached_property

from devito import TimeFunction, SparseTimeFunction, Operator, Function
from devito.tools import memoized_meth

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
        self.options = options
        self._src = SparseTimeFunction(name="src", grid=self.model.grid, npoint=1, nt=1)
        self._rec = SparseTimeFunction(name="rec", grid=self.model.grid, npoint=1, nt=1)

    @property
    def dt(self):
        return self.model.critical_dt

    # Objects creation
    @memoized_meth
    def wavefield(self, save=None, fw=True, lin=False):
        name = "u" if fw else "v"
        name = name+"l" if lin else name
        u = TimeFunction(name=name, grid=self.model.grid, save=save, time_order=2,
                         space_order=self.options['space_order'])
        return u

    @cached_property
    def image(self):
        return Function(name="gradm", grid=self.model.grid, space_order=0)

    def make_src(self, data, coords, nt):
        npoint = coords.shape[0]
        src = SparseTimeFunction(name="src", grid=self.model.grid,
                                 npoint=npoint, nt=nt, coordinates=coords)
        src.data[:] = data
        return src
    
    def make_rec(self, coords, nt):
        npoint = coords.shape[0]
        src = SparseTimeFunction(name="rec", grid=self.model.grid,
                                 npoint=npoint, nt=nt, coordinates=coords)
                                
        return src

    # Equation creations
    def time_stepper(self, u, fw=True, q=None):
        return acoustic_kernel(self.model, u, fw=fw, q=q)

    def source_eq(self, u, fw=True):
        next = u.forward if fw else u.backward
        return self._src.inject(next, expr=self._src*self.dt**2/self.model.m)
    
    def receiver_eq(self, u):
        return self._rec.interpolate(u)

    # Operator creation
    @memoized_meth
    def fwd_op(self, save=None, wsrc=True, wrec=True):
        u = self.wavefield(save=save)
        eq = self.time_stepper(u)
        src = self.source_eq(u) if wsrc else []
        rec = self.receiver_eq(u) if wrec else []
        subs = self.model.spacing_map
        op = Operator(eq + src + rec, subs=subs, name="forward")
        op.cfunction
        return op
    
    @memoized_meth
    def born_op(self, save=None, wsrc=True, wrec=True):
        u = self.wavefield(save=save)
        ul = self.wavefield(lin=True)
        eq = self.time_stepper(u)
        eql = self.time_stepper(ul, q=lin_src(self.model, u, isic=self.options['isic']))
        src = self.source_eq(u) if wsrc else []
        rec = self.receiver_eq(ul) if wrec else []
        subs = self.model.spacing_map
        op = Operator(eq + src + eql + rec, subs=subs, name="born")
        op.cfunction
        return op
    
    @memoized_meth
    def adj_op(self, save=None, wsrc=True, wrec=True):
        u = self.wavefield(save=save, fw=False)
        eq = self.time_stepper(u, fw=False)
        src = self.source_eq(u, fw=False) if wsrc else []
        rec = self.receiver_eq(u) if wrec else []
        subs = self.model.spacing_map
        op = Operator(eq + src + rec, subs=subs, name="adjoint")
        op.cfunction
        return op

    @memoized_meth
    def grad_op(self, isic=False):
        u = self.wavefield(save=10)
        v = self.wavefield(fw=False)
        grad = self.image
        grad_eq = grad_expr(grad, u, v, self.model, isic=isic)
        eq = self.time_stepper(v, fw=False)
        src = self.source_eq(v, fw=False)
        subs = self.model.spacing_map
        op = Operator(eq + src + grad_eq, subs=subs, name="gradient")
        op.cfunction
        return op

    # Call
    def forward(self, save=False, wavelet=None, src_coords=None, rec_coords=None):

        # Number of tim-steps
        nt = wavelet.shape[0] if wavelet is not None else None
        save = nt if save else None
        u = self.wavefield(save=save)
        u.data.fill(0.0)
        # Get flags
        wsrc = self._src is not None
        wrec = self._rec is not None
        # Make devito src/rec
        src = self.make_src(wavelet, src_coords, nt) if wsrc else None
        rec = self.make_rec(rec_coords, nt) if wrec else None
        # kwargs
        kw = {k.name: k for k in [u, src, rec] if k is not None}
        kw.update({'dt': self.dt})
        self.fwd_op(save=save, wsrc=wsrc, wrec=wrec).apply(**kw)
        return rec.data

    def adjoint(self, save=False, wavelet=None, src_coords=None, rec_coords=None):
        # Number of tim-steps
        nt = wavelet.shape[0] if wavelet is not None else None
        save = nt if save else None
        u = self.wavefield(save=save, fw=False)
        u.data.fill(0.0)
        # Get flags
        wsrc = self._src is not None
        wrec = self._rec is not None
        # Make devito src/rec
        src = self.make_src(wavelet, src_coords, nt) if wsrc else None
        rec = self.make_rec(rec_coords, nt) if wrec else None
        # kwargs
        kw = {k.name: k for k in [u, src, rec] if k is not None}
        kw.update({'dt': self.dt})
        self.adj_op(save=save, wsrc=wsrc, wrec=wrec).apply(**kw)
        return rec.data

    def born(self, save=False, wavelet=None, src_coords=None, rec_coords=None):
        # Number of tim-steps
        nt = wavelet.shape[0] if wavelet is not None else None
        save = nt if save else None
        # wavefields
        u = self.wavefield(save=save)
        u.data.fill(0.0)
        ul = self.wavefield(lin=True)
        ul.data.fill(0.0)
        # Get flags
        wsrc = self._src is not None
        wrec = self._rec is not None
        # Make devito src/rec
        src = self.make_src(wavelet, src_coords, nt) if wsrc else None
        rec = self.make_rec(rec_coords, nt) if wrec else None
        # kwargs
        kw = {k.name: k for k in [u, ul, src, rec] if k is not None}
        kw.update({'dt': self.dt})
        self.born_op(save=save, wsrc=wsrc, wrec=wrec).apply(**kw)
        return rec.data

    def adjoint_born(self, wavelet=None, src_coords=None, rec_coords=None, rec_data=None):
        nt = wavelet.shape[0] if wavelet is not None else None
        self.forward(save=nt, wavelet=wavelet, src_coords=src_coords,
                     rec_coords=rec_coords)
        grad = Function(name="gradm", grid=self.model.grid, space_order=0)
        u = self.wavefield(save=nt)
        v = self.wavefield(fw=False)
        src = self.make_src(rec_data, rec_coords, nt)
        # kwargs
        kw = {k.name: k for k in [u, v, src, grad] if k is not None}
        kw.update({'dt': self.dt})
        self.grad_op(isic=self.options['isic']).apply(**kw)
        return grad.data
