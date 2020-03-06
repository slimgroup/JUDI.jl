from sources import *


def src_rec(model, u, src_coords=None, rec_coords=None, wavelet=None, fw=True, nt=None):
    m, irho = model.m, model.irho
    m = m * irho
    dt = model.grid.time_dim.spacing
    geom_expr = []
    src = None
    nt = nt or  wavelet.shape[0]
    if src_coords is not None:
        src = PointSource(name="src", grid=model.grid, ntime=nt, coordinates=src_coords)
        src.data[:] = wavelet[:] if wavelet is not None else 0.
        if model.is_tti:
            u_n = (u[0].forward, u[1].forward) if fw else (u[0].backward, u[1].backward)
            geom_expr += src.inject(field=u_n[0], expr=src*dt**2/m)
            geom_expr += src.inject(field=u_n[1], expr=src*dt**2/m)
        else:
            u_n = u.forward if fw else u.backward
            geom_expr += src.inject(field=u_n, expr=src*dt**2/m)
    # Setup adjoint wavefield sampling at source locations
    rcv = None
    if rec_coords is not None:
        rcv = Receiver(name="rcv", grid=model.grid, ntime=nt, coordinates=rec_coords)
        rec_expr = (u[0] + u[1]) if model.is_tti else u
        adj_rcv = rcv.interpolate(expr=rec_expr)
        geom_expr += adj_rcv
    return geom_expr, src, rcv
