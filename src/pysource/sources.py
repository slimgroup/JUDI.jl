import sympy

from cached_property import cached_property

from devito import Dimension
from devito.types import SparseTimeFunction
from devito.logger import error
import numpy as np


__all__ = ['PointSource', 'Receiver', 'Shot', 'RickerSource', 'GaborSource', 'TimeAxis']


class TimeAxis(object):
    """
    Data object to store the TimeAxis. Exactly three of the four key arguments
    must be prescribed. Because of remainder values it is not possible to create
    a TimeAxis that exactly adhears to the inputs therefore start, stop, step
    and num values should be taken from the TimeAxis object rather than relying
    upon the input values.
    The four possible cases are:
    * start is None: start = step*(1 - num) + stop
    * step is None: step = (stop - start)/(num - 1)
    * num is None: num = ceil((stop - start + step)/step) and
    because of remainder stop = step*(num - 1) + start
    * stop is None: stop = step*(num - 1) + start

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


class PointSource(SparseTimeFunction):
    """
    Symbolic data object for a set of sparse point sources

    Parameters
    ----------
    name: String
        Name of the symbol representing this source
    grid: Grid
        Grid object defining the computational domain.
    coordinates: Array
        Point coordinates for this source
    data: (Optional) Data
        values to initialise point data
    ntime: Int (Optional)
        Number of timesteps for which to allocate data
    npoint: Int (Optional)
    Number of sparse points represented by this source
    dimension: Dimension (Optional)
         object for representing the number of points in this source
    Note, either the dimensions `ntime` and `npoint` or the fully
    initialised `data` array need to be provided.
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})

        key = cls
        obj = cls._cache_get(key)

        if obj is not None:
            newobj = sympy.Function.__new__(cls, *args, **options)
            newobj.__init_cached__(key)
            return newobj

        p_dim = kwargs.get('dimension', Dimension('p_%s' % kwargs.get("name")))
        npoint = kwargs.get("npoint")
        coords = kwargs.get("coordinates")
        if npoint is None:
            if coords is None:
                raise TypeError("Need either `npoint` or `coordinates`")
            else:
                npoint = coords.shape[0]

        grid = kwargs.get("grid")
        ntime = kwargs.get("ntime")
        if kwargs.get("data") is None:
            if ntime is None:
                error('Either data or ntime are required to'
                      'initialise source/receiver objects')
        else:
            ntime = kwargs.get("ntime") or kwargs.get("data").shape[0]

        # Create the underlying SparseTimeFunction object
        kwargs["nt"] = ntime
        kwargs['npoint'] = npoint
        obj = SparseTimeFunction.__new__(cls, dimensions=[grid.time_dim, p_dim], **kwargs)

        # If provided, copy initial data into the allocated buffer
        if kwargs.get("data") is not None:
            obj.data[:] = kwargs.get("data")

        return obj


Receiver = PointSource
Shot = PointSource


class WaveletSource(PointSource):
    """
    Abstract base class for symbolic objects that encapsulate a set of
    sources with a pre-defined source signal wavelet.
    name: Name for the resulting symbol
    grid: :class:`Grid` object defining the computational domain.
    f0: Peak frequency for Ricker wavelet in kHz
    time: Discretized values of time in ms
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})

        key = cls
        obj = cls._cache_get(key)

        if obj is not None:
            newobj = sympy.Function.__new__(cls, *args, **options)
            newobj.__init_cached__(key)
            return newobj

        time = kwargs.get('time')
        npoint = kwargs.get('npoint', 1)
        kwargs['ntime'] = len(time)
        kwargs['npoint'] = npoint
        obj = PointSource.__new__(cls, *args, **kwargs)

        obj.time = time
        obj.f0 = kwargs.get('f0')
        for p in range(npoint):
            obj.data[:, p] = obj.wavelet(obj.f0, obj.time)
        return obj

    def wavelet(self, f0, t):
        """
        Defines a wavelet with a peak frequency f0 at time t.
        f0: Peak frequency in kHz
        t: Discretized values of time in ms
        """
        raise NotImplementedError('Wavelet not defined')


class RickerSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Ricker wavelet:
    http://subsurfwiki.org/wiki/Ricker_wavelet
    name: Name for the resulting symbol
    grid: :class:`Grid` object defining the computational domain.
    f0: Peak frequency for Ricker wavelet in kHz
    time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Ricker wavelet with a peak frequency f0 at time t.
        f0: Peak frequency in kHz
        t: Discretized values of time in ms
        """
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)


class GaborSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Gabor wavelet:
    https://en.wikipedia.org/wiki/Gabor_wavelet
    name: Name for the resulting symbol
    grid: :class:`Grid` object defining the computational domain.
    f0: Peak frequency for Ricker wavelet in kHz
    time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Gabor wavelet with a peak frequency f0 at time t.
        f0: Peak frequency in kHz
        t: Discretized values of time in ms
        """
        agauss = 0.5 * f0
        tcut = 1.5 / agauss
        s = (t-tcut) * agauss
        return np.exp(-2*s**2) * np.cos(2 * np.pi * s)
