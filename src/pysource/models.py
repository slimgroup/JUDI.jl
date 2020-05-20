import numpy as np
from sympy import sin, Abs
import warnings
from devito import (Grid, Function, SubDomain, SubDimension, Eq, Inc,
                    Operator, mmax, initialize_function)
from devito.tools import as_tuple


__all__ = ['Model']


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


def initialize_damp(damp, nbl, fs=False):
    """
    Initialise damping field with an absorbing boundary layer.

    Parameters
    ----------
    damp : Function
        The damping field for absorbing boundary condition.
    nbl : int
        Number of points in the damping layer.
    spacing :
        Grid spacing coefficient.
    mask : bool, optional
        whether the dampening is a mask or layer.
        mask => 1 inside the domain and decreases in the layer
        not mask => 0 inside the domain and increase in the layer
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (nbl)

    eqs = [Eq(damp, 1.0)]
    scaling = 10
    z = damp.dimensions[-1]
    for d in damp.dimensions:
        if not fs or d is not z:
            # left
            dim_l = SubDimension.left(name='abc_%s_l' % d.name, parent=d,
                                      thickness=nbl)
            pos = Abs((nbl - (dim_l - d.symbolic_min) + 1) / float(nbl))
            val = -dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
            eqs += [Inc(damp.subs({d: dim_l}), val/scaling)]
        # right
        dim_r = SubDimension.right(name='abc_%s_r' % d.name, parent=d,
                                   thickness=nbl)
        pos = Abs((nbl - (d.symbolic_max - dim_r) + 1) / float(nbl))
        val = -dampcoeff * (pos - sin(2*np.pi*pos)/(2*np.pi))
        eqs += [Inc(damp.subs({d: dim_r}), val/scaling)]

    Operator(eqs, name='initdamp')()


class GenericModel(object):
    """
    General model class with common properties
    """
    def __init__(self, origin, spacing, shape, space_order, nbl=20,
                 dtype=np.float32, fs=False):
        self.shape = shape
        self.nbl = int(nbl)
        self.origin = tuple([dtype(o) for o in origin])
        self.fs = fs
        # Origin of the computational domain with boundary to inject/interpolate
        # at the correct index
        origin_pml = [dtype(o - s*nbl) for o, s in zip(origin, spacing)]
        shape_pml = np.array(shape) + 2 * self.nbl
        if fs:
            fsdomain = FSDomain(space_order//2)
            physdomain = PhysicalDomain(space_order//2, fs=fs)
            subdomains = (physdomain, fsdomain)
            origin_pml[-1] = origin[-1]
            shape_pml[-1] -= self.nbl
        else:
            subdomains = ()
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml, origin=tuple(origin_pml),
                         dtype=dtype, subdomains=subdomains)

        if self.nbl != 0:
            # Create dampening field as symbol `damp`
            self.damp = Function(name="damp", grid=self.grid)
            initialize_damp(self.damp, self.nbl, fs=fs)
            self._physical_parameters = ['damp']
        else:
            self.damp = 1
            self._physical_parameters = []

    @property
    def padsizes(self):
        padsizes = [(self.nbl, self.nbl) for _ in range(self.dim-1)]
        padsizes.append((0 if self.fs else self.nbl, self.nbl))
        return padsizes

    def physical_params(self, **kwargs):
        """
        Return all set physical parameters and update to input values if provided
        """
        known = [getattr(self, i) for i in self.physical_parameters]
        return {i.name: kwargs.get(i.name, i) or i for i in known}

    def _gen_phys_param(self, field, name, space_order, is_param=False,
                        default_value=0, func=lambda x: x):
        """
        Create symbolic object an initiliaze its data
        """
        if field is None:
            return default_value
        if isinstance(field, np.ndarray):
            function = Function(name=name, grid=self.grid, space_order=space_order,
                                parameter=is_param)
            initialize_function(function, field, self.padsizes)
        else:
            return field
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
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        return self.grid.spacing_map

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


class Model(GenericModel):
    """
    The physical model used in seismic inversion processes.

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
    vp : array_like or float
        Velocity in km/s.
    nbl : int, optional
        The number of absorbin layers for boundary damping.
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
    def __init__(self, origin, spacing, shape, vp, space_order=2, nbl=40,
                 dtype=np.float32, epsilon=None, delta=None, theta=None, phi=None,
                 rho=1, dm=None, fs=False, **kwargs):
        super(Model, self).__init__(origin, spacing, shape, space_order, nbl, dtype,
                                    fs=fs)

        self.scale = 1
        self._space_order = space_order
        # Create square slowness of the wave as symbol `m`
        self._vp = self._gen_phys_param(vp, 'vp', space_order)
        # density
        self.irho = self._gen_phys_param(rho, 'irho', space_order, func=lambda x: 1/x)
        self._dm = self._gen_phys_param(dm, 'dm', space_order)
        # Additional parameter fields for TTI operators
        self._is_tti = any(p is not None for p in [epsilon, delta, theta, phi])
        if self._is_tti:
            epsilon = 1 if epsilon is None else 1 + 2 * epsilon
            delta = 1 if delta is None else 1 + 2 * delta
            self.epsilon = self._gen_phys_param(epsilon, 'epsilon', space_order)
            self.scale = np.sqrt(np.max(epsilon))
            self.delta = self._gen_phys_param(delta, 'delta', space_order)
            self.theta = self._gen_phys_param(theta, 'theta', space_order)
            self.phi = self._gen_phys_param(phi, 'phi', space_order)
        # User provided dt
        self._dt = kwargs.get('dt')

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
    def _max_vp(self):
        """
        Maximum velocity
        """
        return mmax(self.vp)

    @property
    def critical_dt(self):
        """
        Critical computational time step value from the CFL condition.
        """
        # For a fixed time order this number decreases as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * np.min(self.spacing) / (self.scale*self._max_vp))
        if self.dt:
            if self.dt > dt:
                warnings.warn("Provided dt=%s is bigger than maximum stable dt %s "
                              % (self.dt, dt))
            else:
                return self.dtype("%.3e" % self.dt)
        return self.dtype("%.3e" % dt)

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
        vp : float or array
            New velocity in km/s.
        """
        # Update the square slowness according to new value
        if isinstance(dm, np.ndarray):
            if not isinstance(self._dm, Function) and dm.shape == self.shape:
                self._dm = self._gen_phys_param(dm, 'dm', self.space_order)
            elif dm.shape == self.shape:
                initialize_function(self._dm, dm, self.nbl)
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
    def vp(self):
        """
        Function holding the model velocity in km/s.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """
        Set a new velocity model.

        Parameters
        ----------
        vp : float or array
            New velocity in km/s.
        """
        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            if vp.shape == self.vp.shape:
                self.vp.data[:] = vp[:]
            elif vp.shape == self.shape:
                initialize_function(self._vp, vp, self.nbl)
            else:
                raise ValueError("Incorrect input size %s for model of size" % vp.shape +
                                 " %s without or %s with padding" % (self.shape,
                                                                     self.vp.shape))
        else:
            self._vp.data = vp

    @property
    def m(self):
        """
        Symbolic representation of the squared slowness
        m = 1/vp^2
        """
        return 1 / (self.vp * self.vp)

    @property
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each `SpaceDimension`.
        """
        map = self.grid.spacing_map
        map.update({self.grid.time_dim.spacing: self.critical_dt})
        return map
