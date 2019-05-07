import numpy as np

from scipy import ndimage
from cached_property import cached_property

from devito import error, DefaultDimension, Eq
from devito.tools import Pickable

from PySource import *
__all__ = ['scipy_smooth', 'AcquisitionGeometry', 'freesurface']

def freesurface(field, stencil_s, npml, forward=True):
    """
    Generate the stencil that mirrors the field as a free surface modeling for
    the acoustic wave equation
    """
    fs = DefaultDimension(name="fs", default_value=stencil_s)

    field_m = field.forward if forward else field.backward

    lhs = field_m.subs({field.indices[-1]: npml - fs - 1})
    rhs = -field_m.subs({field.indices[-1]: npml + fs + 1})
    return [Eq(lhs, rhs)]



class TimeAxis(object):
    """
    Data object to store the TimeAxis. Exactly three of the four key arguments
    must be prescribed. Because of remainder values it is not possible to create
    a TimeAxis that exactly adhears to the inputs therefore start, stop, step
    and num values should be taken from the TimeAxis object rather than relying
    upon the input values.
    The four possible cases are:
    start is None: start = step*(1 - num) + stop
    step is None: step = (stop - start)/(num - 1)
    num is None: num = ceil((stop - start + step)/step);
                 because of remainder stop = step*(num - 1) + start
    stop is None: stop = step*(num - 1) + start
    Parameters
    ----------
    start : float, optional
        Start of time axis.
    step : float, optional
        Time interval.
    num : int, optional
        Number of values (Note: this is the number of intervals + 1).
        Stop value is reset to correct for remainder.
    stop : float, optional
        End time.
    """
    def __init__(self, start=None, step=None, num=None, stop=None):
        try:
            if start is None:
                start = step*(1 - num) + stop
            elif step is None:
                step = (stop - start)/(num - 1)
            elif num is None:
                num = int(np.ceil((stop - start + step)/step))
                stop = step*(num - 1) + start
            elif stop is None:
                stop = step*(num - 1) + start
            else:
                raise ValueError("Only three of start, step, num and stop may be set")
        except:
            raise ValueError("Three of args start, step, num and stop may be set")

        if not isinstance(num, int):
            raise TypeError("input argument must be of type int")

        self.start = start
        self.stop = stop
        self.step = step
        self.num = num

    def __str__(self):
        return "TimeAxis: start=%g, stop=%g, step=%g, num=%g" % \
               (self.start, self.stop, self.step, self.num)

    def _rebuild(self):
        return TimeAxis(start=self.start, stop=self.stop, num=self.num)

    @cached_property
    def time_values(self):
        return np.linspace(self.start, self.stop, self.num)


class AcquisitionGeometry(Pickable):
    """
    Encapsulate the geometry of an acquisition:
    - receiver positions and number
    - source positions and number
    In practice this would only point to a segy file with the
    necessary information
    """

    def __init__(self, model, rec_positions, src_positions, t0, tn, **kwargs):
        """
        In practice would be __init__(segyfile) and all below parameters
        would come from a segy_read (at property call rather than at init)
        """
        self.rec_positions = rec_positions
        self._nrec = rec_positions.shape[0]
        self.src_positions = src_positions
        self._nsrc = src_positions.shape[0]
        self._src_type = kwargs.get('src_type')
        assert self.src_type in sources
        self._f0 = kwargs.get('f0')
        if self._src_type is not None and self._f0 is None:
            error("Peak frequency must be provided in KH" +
                  " for source of type %s" % self._src_type)

        self._model = model
        self._dt = model.critical_dt
        self._t0 = t0
        self._tn = tn

    def resample(self, dt):
        self._dt = dt
        return self

    @property
    def time_axis(self):
        return TimeAxis(start=self.t0, stop=self.tn, step=self.dt)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def src_type(self):
        return self._src_type

    @property
    def grid(self):
        return self.model.grid

    @property
    def f0(self):
        return self._f0

    @property
    def tn(self):
        return self._tn

    @property
    def t0(self):
        return self._t0

    @property
    def dt(self):
        return self._dt

    @property
    def nt(self):
        return self.time_axis.num

    @property
    def nrec(self):
        return self._nrec

    @property
    def nsrc(self):
        return self._nsrc

    @property
    def dtype(self):
        return self.grid.dtype

    @property
    def rec(self):
        return Receiver(name='rec', grid=self.grid,
                        ntime=self.time_axis.num, npoint=self.nrec,
                        coordinates=self.rec_positions)

    @property
    def src(self):
        if self.src_type is None:
            return PointSource(name='src', grid=self.grid,
                               time=self.time_axis.time_values, npoint=self.nsrc,
                               coordinates=self.src_positions)
        else:
            return sources[self.src_type](name='src', grid=self.grid, f0=self.f0,
                                          time=self.time_axis.time_values, npoint=self.nsrc,
                                          coordinates=self.src_positions)

    _pickle_args = ['model', 'rec_positions', 'src_positions', 't0', 'tn']
    _pickle_kwargs = ['f0', 'src_type']


sources = {'Ricker': RickerSource, 'Gabor': GaborSource}
