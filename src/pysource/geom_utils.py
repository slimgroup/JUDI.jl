from devito.tools import as_tuple

from sources import *


def src_rec(model, u, src_coords=None, rec_coords=None, wavelet=None, fw=True, nt=None, ivp_adj=False):
    """
    Generates the source injection and receiver interpolation.
    This function is fully abstracted and does not care whether this is a
    forward or adjoint wave-equation.
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
    namef = as_tuple(u)[0].name
    if src_coords is not None:
        if isinstance(wavelet, PointSource):
            src = wavelet
        else:
            src = PointSource(name="src%s" % namef, grid=model.grid, ntime=nt,
                              coordinates=src_coords)
            src.data[:] = wavelet[:] if wavelet is not None else 0.
        u_n = as_tuple(u)[0].forward if fw else as_tuple(u)[0].backward
        if(ivp_adj):
            #for PA adjoint we need -(derivative of source)   
            geom_expr += src.inject(field=u_n, expr=-src.dt*dt**2/m)
        else:
            geom_expr += src.inject(field=u_n, expr=src*dt**2/m)
    # Setup adjoint wavefield sampling at source locations
    rcv = None
    if rec_coords is not None:
        rcv = Receiver(name="rcv%s" % namef, grid=model.grid, ntime=nt,
                       coordinates=rec_coords)
        rec_expr = u[0] if model.is_tti else u
        adj_rcv = rcv.interpolate(expr=rec_expr)
        geom_expr += adj_rcv
    return geom_expr, src, rcv
