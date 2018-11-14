import numpy as np
import os

from devito import Grid, Function, Constant
from devito.logger import error


__all__ = ['Model']

def damp_boundary(damp, nbpml, spacing):
    """Initialise damping field with an absorbing PML layer.

    :param damp: Array data defining the damping field
    :param nbpml: Number of points in the damping layer
    :param spacing: Grid spacing coefficent
    """
    dampcoeff = 1.5 * np.log(1.0 / 0.001) / (40.)
    ndim = len(damp.shape)
    for i in range(nbpml):
        pos = np.abs((nbpml - i + 1) / float(nbpml))
        val = dampcoeff * (pos - np.sin(2*np.pi*pos)/(2*np.pi))
        if ndim == 2:
            damp[i, :] += val/spacing[0]
            damp[-(i + 1), :] += val/spacing[0]
            damp[:, i] += val/spacing[1]
            damp[:, -(i + 1)] += val/spacing[1]
        else:
            damp[i, :, :] += val/spacing[0]
            damp[-(i + 1), :, :] += val/spacing[0]
            damp[:, i, :] += val/spacing[1]
            damp[:, -(i + 1), :] += val/spacing[1]
            damp[:, :, i] += val/spacing[2]
            damp[:, :, -(i + 1)] += val/spacing[2]


def initialize_function(function, data, nbpml):
    """Initialize a :class:`Function` with the given ``data``. ``data``
    does *not* include the PML layers for the absorbing boundary conditions;
    these are added via padding by this method.
    :param function: The :class:`Function` to be initialised with some data.
    :param data: The data array used for initialisation.
    :param nbpml: Number of PML layers for boundary damping.
    """
    pad_list = [(nbpml + i.left, nbpml + i.right) for i in function._offset_domain]
    function.data_with_halo[:] = np.pad(data, pad_list, 'edge')


class Model(object):
    """The physical model used in seismic inversion processes.
    :param origin: Origin of the model in m as a tuple in (x,y,z) order
    :param spacing: Grid size in m as a Tuple in (x,y,z) order
    :param shape: Number of grid points size in (x,y,z) order
    :param vp: Velocity in km/s
    :param nbpml: The number of PML layers for boundary damping
    :param dm: Model perturbation in s^2/km^2
    The :class:`Model` provides two symbolic data objects for the
    creation of seismic wave propagation operators:
    :param m: The square slowness of the wave
    :param damp: The damping field for absorbing boundarycondition
    """
    def __init__(self, origin, spacing, shape, vp, rho=1, nbpml=20, dtype=np.float32, dm=None,
                 space_order=8):
        self.shape = shape
        self.nbpml = int(nbpml)

        shape_pml = np.array(shape) + 2 * self.nbpml
        # Physical extent is calculated per cell, so shape - 1
        extent = tuple(np.array(spacing) * (shape_pml - 1))
        self.grid = Grid(extent=extent, shape=shape_pml,
                         origin=origin, dtype=dtype)

        # Create square slowness of the wave as symbol `m`
        if isinstance(vp, np.ndarray):
            self.m = Function(name="m", grid=self.grid, space_order=space_order)
        else:
            self.m = 1/vp**2

        if isinstance(rho, np.ndarray):
            self.rho = Function(name="rho", grid=self.grid, space_order=space_order)
            initialize_function(self.rho, rho, self.nbpml)
        else:
            self.rho = rho

        # Set model velocity, which will also set `m`
        self.vp = vp

        # Create dampening field as symbol `damp`
        self.damp = Function(name="damp", grid=self.grid)
        damp_boundary(self.damp.data, self.nbpml, spacing=self.spacing)

        # Additional parameter fields for TTI operators
        self.scale = 1.

        if dm is not None:
            self.dm = Function(name="dm", grid=self.grid)
            self.dm.data[:] = self.pad(dm)
        else:
            self.dm = 1

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
    def spacing_map(self):
        """
        Map between spacing symbols and their values for each :class:`SpaceDimension`
        """
        return self.grid.spacing_map

    @property
    def origin(self):
        """
        Coordinates of the origin of the physical model.
        """
        return self.grid.origin

    @property
    def dtype(self):
        """
        Data type for all assocaited data objects.
        """
        return self.grid.dtype

    @property
    def shape_domain(self):
        """Computational shape of the model domain, with PML layers"""
        return tuple(d + 2*self.nbpml for d in self.shape)

    @property
    def domain_size(self):
        """
        Physical size of the domain as determined by shape and spacing
        """
        return tuple((d-1) * s for d, s in zip(self.shape, self.spacing))

    @property
    def critical_dt(self):
        """Critical computational time step value from the CFL condition."""
        # For a fixed time order this number goes down as the space order increases.
        #
        # The CFL condtion is then given by
        # dt <= coeff * h / (max(velocity))
        coeff = 0.38 if len(self.shape) == 3 else 0.42
        dt = self.dtype(coeff * np.min(self.spacing) / (self.scale*np.max(self.vp)))
        return 0.001 * int(1000.*dt)

    @property
    def vp(self):
        """:class:`numpy.ndarray` holding the model velocity in km/s.
        .. note::
        Updating the velocity field also updates the square slowness
        ``self.m``. However, only ``self.m`` should be used in seismic
        operators, since it is of type :class:`Function`.
        """
        return self._vp

    @vp.setter
    def vp(self, vp):
        """Set a new velocity model and update square slowness
        :param vp : new velocity in km/s
        """
        self._vp = vp

        # Update the square slowness according to new value
        if isinstance(vp, np.ndarray):
            initialize_function(self.m, 1 / (self.vp * self.vp), self.nbpml)
        else:
            self.m.data = 1 / vp**2

    def pad(self, data):
        """Padding function PNL layers in every direction for for the
        absorbing boundary conditions.

        :param data : Data array to be padded"""
        pad_list = [(self.nbpml, self.nbpml) for _ in self.shape]
        return np.pad(data, pad_list, 'edge')
