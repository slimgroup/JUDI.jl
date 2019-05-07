
import sympy

from devito import Dimension
from devito.types.basic import _SymbolCache
from devito.types import SparseTimeFunction
from devito.logger import error
import numpy as np


__all__ = ['PointSource', 'Receiver', 'Shot', 'RickerSource', 'GaborSource']


class PointSource(SparseTimeFunction):
    """Symbolic data object for a set of sparse point sources
    :param name: Name of the symbol representing this source
    :param grid: :class:`Grid` object defining the computational domain.
    :param coordinates: Point coordinates for this source
    :param data: (Optional) Data values to initialise point data
    :param ntime: (Optional) Number of timesteps for which to allocate data
    :param npoint: (Optional) Number of sparse points represented by this source
    :param dimension: :(Optional) class:`Dimension` object for
                       representing the number of points in this source
    Note, either the dimensions `ntime` and `npoint` or the fully
    initialised `data` array need to be provided.
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})
        if cls in _SymbolCache:
            obj = sympy.Function.__new__(cls, *args, **options)
            obj._cached_init()
        else:
            p_dim = kwargs.get('dimension', Dimension('p_%s' % kwargs.get("name")))
            npoint = kwargs.get("npoint")
            coords = kwargs.get("coordinates")
            if npoint is None:
                if coords is None:
                    raise TypeError("Need either `npoint` or `coordinates`")
                else:
                    npoint = coords.shape[0]
            name = kwargs.get("name")
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

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(PointSource, self).__init__(*args, **kwargs)


Receiver = PointSource
Shot = PointSource


class WaveletSource(PointSource):
    """
    Abstract base class for symbolic objects that encapsulate a set of
    sources with a pre-defined source signal wavelet.
    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def __new__(cls, *args, **kwargs):
        options = kwargs.get('options', {})
        if cls in _SymbolCache:
            obj = sympy.Function.__new__(cls, *args, **options)
            obj._cached_init()
        else:
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

    def __init__(self, *args, **kwargs):
        if not self._cached():
            super(WaveletSource, self).__init__(*args, **kwargs)

    def wavelet(self, f0, t):
        """
        Defines a wavelet with a peak frequency f0 at time t.
        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        raise NotImplementedError('Wavelet not defined')


class RickerSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Ricker wavelet:
    http://subsurfwiki.org/wiki/Ricker_wavelet
    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Ricker wavelet with a peak frequency f0 at time t.
        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        r = (np.pi * f0 * (t - 1./f0))
        return (1-2.*r**2)*np.exp(-r**2)


class GaborSource(WaveletSource):
    """
    Symbolic object that encapsulate a set of sources with a
    pre-defined Gabor wavelet:
    https://en.wikipedia.org/wiki/Gabor_wavelet
    :param name: Name for the resulting symbol
    :param grid: :class:`Grid` object defining the computational domain.
    :param f0: Peak frequency for Ricker wavelet in kHz
    :param time: Discretized values of time in ms
    """

    def wavelet(self, f0, t):
        """
        Defines a Gabor wavelet with a peak frequency f0 at time t.
        :param f0: Peak frequency in kHz
        :param t: Discretized values of time in ms
        """
        agauss = 0.5 * f0
        tcut = 1.5 / agauss
        s = (t-tcut) * agauss
        return np.exp(-2*s**2) * np.cos(2 * np.pi * s)
