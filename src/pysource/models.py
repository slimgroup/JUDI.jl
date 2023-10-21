import numpy as np
import warnings
from sympy import finite_diff_weights as fd_w
from devito import (Grid, Function, SubDomain, SubDimension, Eq, Inc,
                    Operator, mmin, mmax, initialize_function, switchconfig,
                    Abs, sqrt, sin)
from devito.data.allocators import ExternalAllocator
from devito.tools import as_tuple, memoized_func

__all__ = ['Model']


def getmin(f):
    try:
        return mmin(f)
    except ValueError:
        return np.min(f)


def getmax(f):
    try:
        return mmax(f)
    except ValueError:
        return np.max(f)


_thomsen = [('epsilon', 1), ('delta', 1), ('theta', 0), ('phi', 0)]


class PhysicalDomain(SubDomain):

    name = 'nofsdomain'

    def __init__(self, so, fs=False):
        super(PhysicalDomain, self).__init__()
        self.so = so
        self.fs = fs

    def define(self, dimensions):
        map_d = {d: d for d in dimensions}
        if self.fs:
            map_d[dimensions[-1]] = ('middle', self.so, 0)
        return map_d


class FSDomain(SubDomain):

    name = 'fsdomain'

    def __init__(self, so):
        super(FSDomain, self).__init__()
        self.size = so

    def define(self, dimensions):
        """
        Definition of the top part of the domain for wrapped indices FS
        """
        z = dimensions[-1]
        map_d = {d: d for d in dimensions}
        map_d.update({z: ('left', self.size)})
        return map_d


@memoized_func
def damp_op(ndim, padsizes, mask_abc, fs):
    """
    Create damping field initialization operator.

    Parameters
    ----------
    ndim : int, number of dimensions
    padsizes : List of tuple
        Number of points in the damping layer for each dimension and side.
    spacing :
        Grid spacing coefficient.
    mask_abc : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    fs : bool, free surface
    """
    damp = Function(name="damp", grid=Grid(tuple([11]*ndim)), space_order=0)
    eqs = [Eq(damp, 1.0 if mask_abc else 0.0)]
    for (nbl, nbr), d in zip(padsizes, damp.dimensions):
        # 3 Point buffer to avoid weird interaction with abc
        nbr = nbr - 3
        if not fs or d is not damp.dimensions[-1]:
            nbl = nbl - 3
            dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbl)
            # left
            dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                      thickness=nbl)
            pos = Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
            val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
            val = -val if mask_abc else val
            eqs += [Inc(damp.subs({d: dim_l}), val/d.spacing)]
        # right
        dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbr)
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbr)
        pos = Abs((nbr - (d.symbolic_max - dim_r) + 1) / float(nbr))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask_abc else val
        eqs += [Inc(damp.subs({d: dim_r}), val/d.spacing)]

    return Operator(eqs, name='initdamp')


@switchconfig(log_level='ERROR')
def initialize_damp(damp, padsizes, mask_abc=False, fs=False):
    """
    Initialise damping field with an absorbing boundary layer.
    Includes basic constant Q setup (not interfaced yet) and assumes that
    the peak frequency is 1/(10 * spacing).

    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    padsizes : List of tuple
        Number of points in the damping layer for each dimension and side.
    mask_abc : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    fs : bool, free surface
    """
    op = damp_op(damp.grid.dim, padsizes, mask_abc, fs)
    op(damp=damp)


@memoized_func
def pml_op(ndim, padsizes, mask_abc, fs):
    """
    Create damping field initialization operator.

    Parameters
    ----------
    ndim : int, number of dimensions
    padsizes : List of tuple
        Number of points in the damping layer for each dimension and side.
    mask_abc : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    fs : bool, free surface
    """
    dims = Grid(shape=(2 for d in range(0,ndim))).dimensions
    eqs = []
    for (nbl, nbr), d in zip(padsizes, dims):
        f0_name = "pml"+d.name+"0"
        f1_name = "pml"+d.name+"1"
        pml0 = Function(name=f0_name, grid=Grid(tuple([11]*ndim)), space_order=0)
        pml1 = Function(name=f1_name, grid=Grid(tuple([11]*ndim)), space_order=0, staggered=dims)
        eqs += [Eq(pml0, 1.0 if mask_abc else 0.0)]
        eqs += [Eq(pml1, 1.0 if mask_abc else 0.0)]
        # 3 Point buffer to avoid weird interaction with abc
        nbr = nbr - 3
        if not fs or d is not dims[-1]:
            nbl = nbl - 3
            dampcoeff = 0.05
            # left
            dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                      thickness=nbl)
            pos = Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
            val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
            val = -val if mask_abc else val
            eqs += [Inc(pml0.subs({d: dim_l}), val/d.spacing)]
            eqs += [Inc(pml1.subs({d: dim_l}), val/d.spacing)]
        # right
        dampcoeff = 0.05
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbr)
        pos = Abs((nbr - (d.symbolic_max - dim_r) + 1) / float(nbr))
        val = dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        val = -val if mask_abc else val
        eqs += [Inc(pml0.subs({d: dim_r}), val/d.spacing)]
        eqs += [Inc(pml1.subs({d: dim_r}), val/d.spacing)]

    return Operator(eqs, name='initpml')


@switchconfig(log_level='ERROR')
def initialize_pml(pmllist, padsizes, mask_abc=False, fs=False):
    """
    Initialise damping field with an pml.

    Parameters
    ----------
    pmllist : list of pml Functions
        The damping field for pml boundary condition.
    padsizes : List of tuple
        Number of points in the damping layer for each dimension and side.
    mask_abc : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    fs : bool, free surface
    """
    op = pml_op(pmllist[0].grid.dim, padsizes, mask_abc, fs)
    op(**{pml.name : pml for pml in pmllist})


class Model(object):
    """
    The physical model used in seismic inversion
        shape_pml = np.array(shape) + 2 * self.nbl processes.

    Parameters
    ----------
    origin : tuple of floats
        Origin of the model in m as a tuple in (x,y,z) order.
    spacing : tuple of floats
        Grid size in m as a Tuple in (x,y,z) order.
    shape : tuple of int
        Number of grid points size in (x,y,z) order.
    space_order : int
        Order of the spatial stencil discretisation.
    m : array_like or float
        Squared slownes in s^2/km^2
    nbl : int, optional
        The number of absorbin layers for boundary damping.
    abc_type: str
        Type of absorbing boundary condition (default "damp" or "pml").
    dtype : np.float32 or np.float64
        Defaults to 32.
    epsilon : array_like or float, optional
        Thomsen epsilon parameter (0<epsilon<1).
    delta : array_like or float
        Thomsen delta parameter (0<delta<1), delta<epsilon.
    theta : array_like or float
        Tilt angle in radian.
    phi : array_like or float
        Asymuth angle in radian.
    dt: Float
        User provided computational time-step
    """
    def __init__(self, origin, spacing, shape, space_order=8, nbl=40, abc_type="damp", dtype=np.float32,
                 m=None, epsilon=None, delta=None, theta=None, phi=None, rho=None,
                 b=None, qp=None, lam=None, mu=None, dm=None, fs=False, **kwargs):
        # Setup devito grid
        self.shape = tuple(shape)
        self.nbl = int(nbl)
        self.origin = tuple([dtype(o) for o in origin])
        mask_abc = True if (qp is not None or mu is not None) else False
        self.fs = fs
        # Origin of the computational domain with boundary to inject/interpolate
        # at the correct index
        origin_pml = [dtype(o - s*nbl) for o, s in zip(origin, spacing)]
        shape_pml = np.array(shape) + 2 * self.nbl
        if fs:
            fsdomain = FSDomain(space_order + 1)
            physdomain = PhysicalDomain(space_order + 1, fs=fs)
            subdomains = (physdomain, fsdomain)
            origin_pml[-1] = origin[-1]
            shape_pml[-1] -= self.nbl
        else:
            subdomains = ()
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=tuple(origin_pml),
                         dtype=dtype, subdomains=subdomains)
        
        # Absorbing boundary layer
        if self.nbl != 0:
            if abc_type == "damp":
                # Create dampening field as symbol `damp`
                self._physical_parameters = ['damp']
                self.damp = Function(name=self._physical_parameters[0], grid=self.grid)
                initialize_damp(self.damp, self.padsizes, mask_abc=mask_abc, fs=fs)
            elif abc_type == "pml":
                # Create dampening field as symbols following pattern: "pmlx0", "pmlx1" etc
                self._physical_parameters = []
                pmllist = []
                for d in self.grid.dimensions:
                    pml0 = "pml"+d.name+"0"
                    pml1 = "pml"+d.name+"1"
                    self._physical_parameters += [pml0]
                    self._physical_parameters += [pml1]
                    setattr(self, pml0, Function(name=pml0, grid=self.grid))
                    setattr(self, pml1, Function(name=pml1, grid=self.grid,staggered=self.grid.dimensions))
                    pmllist += [getattr(self, pml0)]
                    pmllist += [getattr(self, pml1)]
                initialize_pml(pmllist, self.padsizes, mask_abc=mask_abc, fs=fs)
        else:
            self._physical_parameters = []
            if abc_type == "damp":
                self.damp = 1
            elif abc_type == "pml":
                for d in self.grid.dimensions:
                    setattr(self, "pml"+d.name+"0", 1)
                    setattr(self, "pml"+d.name+"1", 1)

        # Seismic fields and properties
        self.scale = 1
        self._space_order = space_order
        # Create square slowness of the wave as symbol `m`
        if m is not None:
            self._m = self._gen_phys_param(m, 'm', space_order)
        # density
        self._init_density(rho, b, space_order)
        self._dm = self._gen_phys_param(dm, 'dm', space_order)

        # Model type
        self._is_viscoacoustic = qp is not None
        self._is_elastic = mu is not None
        self._is_tti = any(p is not None for p in [epsilon, delta, theta, phi])

        # Additional parameter fields for Viscoacoustic operators
        if self._is_viscoacoustic:
            self.qp = self._gen_phys_param(qp, 'qp', space_order)

        # Additional parameter fields for TTI operators
        if self._is_tti:
            epsilon = 1 if epsilon is None else 1 + 2 * epsilon
            delta = 1 if delta is None else 1 + 2 * delta
            self.epsilon = self._gen_phys_param(epsilon, 'epsilon', space_order)
            self.scale = np.sqrt(np.max(epsilon))
            self.delta = self._gen_phys_param(delta, 'delta', space_order)
            self.theta = self._gen_phys_param(theta, 'theta', space_order)
            if self.grid.dim == 3:
                self.phi = self._gen_phys_param(phi, 'phi', space_order)

        # Additional parameter fields for elastic
        if self._is_elastic:
            self.lam = self._gen_phys_param(lam, 'lam', space_order, is_param=True)
            self.mu = self._gen_phys_param(mu, 'mu', space_order, is_param=True)
        # User provided dt
        self._dt = kwargs.get('dt')

    def _init_density(self, rho, b, so):
        """
        Initialize density parameter. Depending on variance in density
        either density or inverse density is setup.
        """
        if rho is not None:
            rm, rM = np.amin(rho), np.amax(rho)
            if rm/rM > .1:
                self.irho = self._gen_phys_param(np.reciprocal(rho), 'irho', so)
                self.rho = 1 / self.irho
            else:
                self.rho = self._gen_phys_param(rho, 'rho', so)
                self.irho = 1 / self.rho
        elif b is not None:
            self.irho = self._gen_phys_param(b, 'irho', so)
        else:
            self.irho = 1

    @property
    def padsizes(self):
        padsizes = [(self.nbl, self.nbl) for _ in range(self.dim-1)]
        padsizes.append((0 if self.fs else self.nbl, self.nbl))
        return tuple(p for p in padsizes)

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        params = {i: kwargs.get(i, getattr(self, i)) for i in self.physical_parameters
                  if isinstance(getattr(self, i), Function)}

        if not kwargs.get('born', False):
            params.pop('dm', None)
        return params

    @property
    def zero_thomsen(self):
        out = {}
        for (t, v) in _thomsen:
            try:
                out.update({getattr(self, t): v})
            except AttributeError:
                pass
        return out

    @switchconfig(log_level='ERROR')
    def _gen_phys_param(self, field, name, space_order, is_param=False,
                        default_value=0):
        """
        Create symbolic object an initiliaze its data
        """
        if field is None:
            return default_value
        if isinstance(field, np.ndarray) and (name == 'm' or
                                              np.min(field) != np.max(field)):
            if field.shape == self.shape:
                function = Function(name=name, grid=self.grid, space_order=space_order,
                                    parameter=is_param)
                initialize_function(function, field, self.padsizes)
            else:
                # We take advantage of the external allocator
                function = Function(name=name, grid=self.grid, space_order=space_order,
                                    allocator=ExternalAllocator(field),
                                    initializer=lambda x: None, parameter=is_param)
        else:
            return np.amin(field)
        self._physical_parameters.append(name)
        return function

    @property
    def physical_parameters(self):
        """
        List of physical parameteres
        """
        return as_tuple(self._physical_parameters)

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def spacing(self):
        """
        Grid spacing for all fields in the physical model.
        """
        return self.grid.spacing

    @property
    def space_dimensions(self):
        """
        Spatial dimensions of the grid
        """
        return self.grid.dimensions

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))

    @property
    def space_order(self):
        """
        Spatial discretization order
        """
        return self._space_order

    @property
    def dt(self):
        """
        User provided dt
        """
        return self._dt

    @dt.setter
    def dt(self, dt):
        """
        Set user provided dt to overwrite the default CFL value.
        """
        self._dt = dt

    @property
    def is_tti(self):
        """
        Whether the model is TTI or isotopic
        """
        return self._is_tti

    @property
    def is_viscoacoustic(self):
        """
        Whether the model is TTI or isotopic
        """
        return self._is_viscoacoustic

    @property
    def is_elastic(self):
        """
        Whether the model is TTI or isotopic
        """
        return self._is_elastic

    @property
    def _max_vp(self):
        """
        Maximum velocity
        """
        if self.is_elastic:
            return np.sqrt(getmin(self.irho) * (getmax(self.lam) + 2 * getmax(self.mu)))
        else:
            return np.sqrt(1./getmin(self.m))

    @property
    def _cfl_coeff(self):
        """
        Courant number from the physics and spatial discretization order.
        The CFL coefficients are described in:
        - https://doi.org/10.1137/0916052 for the elastic case
        - https://library.seg.org/doi/pdf/10.1190/1.1444605 for the acoustic case
        """
        # Elasic coefficient (see e.g )
        if self.is_elastic:
            so = max(self._space_order // 2, 2)
            coeffs = fd_w(1, range(-so, so), .5)
            c_fd = sum(np.abs(coeffs[-1][-1])) / 2
            return np.sqrt(self.dim) / self.dim / c_fd
        a1 = 4  # 2nd order in time
        so = max(self._space_order // 2, 4)
        coeffs = fd_w(2, range(-so, so), 0)[-1][-1]
        return .9 * np.sqrt(a1/float(self.grid.dim * sum(np.abs(coeffs))))

    @property
    def _thomsen_scale(self):
        # Update scale for tti
        if self.is_tti:
            return np.sqrt(1 + 2 * getmax(self.epsilon))
        return 1

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        dt = self._cfl_coeff * np.min(self.spacing) / (self._thomsen_scale*self._max_vp)
        dt = self.dtype("%.3e" % dt)
        if self.dt:
            if self.dt > dt:
                warnings.warn("Provided dt=%s is bigger than maximum stable dt %s "
                              % (self.dt, dt))
            else:
                return self.dtype("%.3e" % self.dt)
        return dt

    @property
    def dm(self):
        """
        Model perturbation for linearized modeling
        """
        return self._dm

    @dm.setter
    def dm(self, dm):
        """
        Set a new model perturbation.

        Parameters
        ----------
        dm : float or array
            New model perturbation
        """
        # Update the square slowness according to new value
        if isinstance(dm, np.ndarray):
            if not isinstance(self._dm, Function):
                self._dm = self._gen_phys_param(dm, 'dm', self.space_order)
            elif dm.shape == self.shape:
                initialize_function(self._dm, dm, self.padsizes)
            elif dm.shape == self.dm.shape:
                self.dm.data[:] = dm[:]
            else:
                raise ValueError("Incorrect input size %s for model of size" % dm.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     self.dm.shape))
        else:
            try:
                self._dm.data = dm
            except AttributeError:
                self._dm = dm

    @property
    def m(self):
        """
        Function holding the squared slowness in s^2/km^2.
        """
        return self._m

    @m.setter
    def m(self, m):
        """
        Set a new squared slowness model.

        Parameters
        ----------
        m : float or array
            New squared slowness in s^2/km^2.
        """
        # Update the square slowness according to new value
        if isinstance(m, np.ndarray):
            if m.shape == self.m.shape:
                self.m.data[:] = m[:]
            elif m.shape == self.shape:
                initialize_function(self._m, m, self.padsizes)
            else:
                raise ValueError("Incorrect input size %s for model of size" % m.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     self.m.shape))
        else:
            self._m.data = m

    @property
    def vp(self):
        """
        Symbolic representation of the velocity
        vp = sqrt(1 / m)
        """
        return sqrt(1 / self.m)

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        sp_map = self.grid.spacing_map
        sp_map.update({self.grid.time_dim.spacing: self.critical_dt})
        return sp_map


class EmptyModel(object):
    """
    An pseudo Model structure that does not contain any physical field
    but only the necessary information to create an operator.
    This Model should not be used for propagation.
    """

    def __init__(self, tti, visco, elastic, spacing, fs, space_order, p_params):
        self.is_tti = tti
        self.is_viscoacoustic = visco
        self.is_elastic = elastic
        self.spacing = spacing
        self.fs = fs
        N = 2 * space_order + 1
        if fs:
            fsdomain = FSDomain(N)
            physdomain = PhysicalDomain(N, fs=fs)
            subdomains = (physdomain, fsdomain)
        else:
            subdomains = ()
        self.grid = Grid(tuple([N]*len(spacing)),
                         extent=[s*(N-1) for s in spacing],
                         subdomains=subdomains)
        self.dimensions = self.grid.dimensions

        # Create the function for the physical parameters
        self.damp = Function(name='damp', grid=self.grid)
        excl = [self.damp.name]
        for d in self.grid.dimensions:
            pml0 = "pml"+d.name+"0"
            pml1 = "pml"+d.name+"1"
            setattr(self, pml0, Function(name=pml0, grid=self.grid))
            setattr(self, pml1, Function(name=pml1, grid=self.grid,staggered=self.grid.dimensions))
            excl += [pml0]
            excl += [pml1]
        for p in set(p_params) - set(excl):
            setattr(self, p, Function(name=p, grid=self.grid, space_order=space_order))
        if 'irho' not in p_params:
            self.irho = 1 if 'rho' not in p_params else 1 / self.rho

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

    @property
    def critical_dt(self):
        """
        User provided dt
        """
        return self.grid.time_dim.spacing

    @property
    def dim(self):
        """
        Spatial dimension of the problem and model domain.
        """
        return self.grid.dim

    @property
    def zero_thomsen(self):
        out = {}
        for (t, v) in _thomsen:
            try:
                out.update({getattr(self, t): v})
            except AttributeError:
                pass
        return out
