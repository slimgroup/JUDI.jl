from devito.tools import Pickable

from sources import *


def src_rec(model, u, src_coords=None, rec_coords=None, wavelet=None, fw=True, nt=None):
    """
    Generates the source injection and receiver interpolation.
    This function is fully abstracted and does not care whether this is a forward or adjoint wave-equation.
    The source is the source term of the equation
    The receiver is the measurment term

    Therefore, for the adjoint, this function has to be called as:
    src_rec(model, v, src_coords=rec_coords, ...)
    because the data is the sources

    Parameters
    ----------
    model : Model
        Physical model
    u : TimeFunction or tuple
        Wavefield to inject into and read from
    src_coords : Array
        Physical coordinates of the sources
    rec_coords : Array
        Physical coordinates of the receivers
    wavelet: Array
        Data for the source
    fw=True:
        Whether the direction is forward or backward in time 
    nt: int
        Number of time steps
    """
    m, irho = model.m, model.irho
    m = m * irho
    dt = model.grid.time_dim.spacing
    geom_expr = []
    src = None
    nt = nt or wavelet.shape[0]
    if src_coords is not None:
        src = PointSource(name="src", grid=model.grid, ntime=nt, coordinates=src_coords)
        src.data[:] = wavelet[:] if wavelet is not None else 0.
        if model.is_tti:
            u_n = (u[0].forward, u[1].forward) if fw else (u[0].backward, u[1].backward)
            geom_expr += src.inject(field=u_n[0], expr=src*dt**2/m)
            # geom_expr += src.inject(field=u_n[1], expr=src*dt**2/m)
        else:
            u_n = u.forward if fw else u.backward
            geom_expr += src.inject(field=u_n, expr=src*dt**2/m)
    # Setup adjoint wavefield sampling at source locations
    rcv = None
    if rec_coords is not None:
        rcv = Receiver(name="rcv", grid=model.grid, ntime=nt, coordinates=rec_coords)
        rec_expr = u[0] if model.is_tti else u
        adj_rcv = rcv.interpolate(expr=rec_expr)
        geom_expr += adj_rcv
    return geom_expr, src, rcv


# class AcquisitionGeometry(Pickable):
#     """
#     Encapsulate the geometry of an acquisition:
#     - receiver positions and number
#     - source positions and number
#     In practice this would only point to a segy file with the
#     necessary information
#     """

#     def __init__(self, model, rec_positions, src_positions, t0, tn, **kwargs):
#         """
#         In practice would be __init__(segyfile) and all below parameters
#         would come from a segy_read (at property call rather than at init)
#         """
#         self.rec_positions = rec_positions
#         self._nrec = rec_positions.shape[0]
#         self.src_positions = src_positions
#         self._nsrc = src_positions.shape[0]
#         self._src_type = kwargs.get('src_type')
#         assert self.src_type in sources
#         self._f0 = kwargs.get('f0')
#         if self._src_type is not None and self._f0 is None:
#             error("Peak frequency must be provided in KH" +
#                   " for source of type %s" % self._src_type)

#         self._model = model
#         self._dt = model.critical_dt
#         self._t0 = t0
#         self._tn = tn

#     def resample(self, dt):
#         self._dt = dt
#         return self

#     @property
#     def time_axis(self):
#         return TimeAxis(start=self.t0, stop=self.tn, step=self.dt)

#     @property
#     def model(self):
#         return self._model

#     @model.setter
#     def model(self, model):
#         self._model = model

#     @property
#     def src_type(self):
#         return self._src_type

#     @property
#     def grid(self):
#         return self.model.grid

#     @property
#     def f0(self):
#         return self._f0

#     @property
#     def tn(self):
#         return self._tn

#     @property
#     def t0(self):
#         return self._t0

#     @property
#     def dt(self):
#         return self._dt

#     @property
#     def nt(self):
#         return self.time_axis.num

#     @property
#     def nrec(self):
#         return self._nrec

#     @property
#     def nsrc(self):
#         return self._nsrc

#     @property
#     def dtype(self):
#         return self.grid.dtype

#     @property
#     def rec(self):
#         return Receiver(name='rec', grid=self.grid,
#                         ntime=self.time_axis.num, npoint=self.nrec,
#                         coordinates=self.rec_positions)

#     @property
#     def src(self):
#         if self.src_type is None:
#             return PointSource(name='src', grid=self.grid,
#                                time=self.time_axis.time_values, npoint=self.nsrc,
#                                coordinates=self.src_positions)
#         else:
#             return sources[self.src_type](name='src', grid=self.grid, f0=self.f0,
#                                           time=self.time_axis.time_values,
#                                           npoint=self.nsrc,
#                                           coordinates=self.src_positions)

#     _pickle_args = ['model', 'rec_positions', 'src_positions', 't0', 'tn']
#     _pickle_kwargs = ['f0', 'src_type']


sources = {'Ricker': RickerSource, 'Gabor': GaborSource}
