from kernels import wave_kernel
from geom_utils import src_rec
from wave_utils import wavefield, grad_expr, lin_src, otf_dft

from devito import Operator, Function


def name(model):
    return "tti" if model.is_tti else ""


def op_kwargs(model, fs=False):
    kw = {}
    if fs:
        z = model.grid.dimensions[-1].name
        kw.update({'%s_m' % z: model.nbl})
    return kw


# Forward propagation
def forward(model, src_coords, rcv_coords, wavelet, space_order=8, save=False,
            q=0, free_surface=False, return_op=False, freq_list=None, dft_sub=None):
    """
    Compute forward wavefield u = A(m)^{-1}*f and related quantities (u(xrcv))
    """
    # Setting adjoint wavefield
    u = wavefield(model, space_order, save=save, nt=wavelet.shape[0])

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, q=q, fs=free_surface)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, u, src_coords=src_coords,
                                  rec_coords=rcv_coords, wavelet=wavelet)
    
    # On-the-fly Fourier
    dft, dft_modes = otf_dft(u, freq_list, model.critical_dt, factor=dft_sub)
    
    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + dft, subs=subs, name="forward"+name(model))

    if return_op:
        return op, u, rcv
    op(**op_kwargs(model, fs=free_surface))

    # Output
    return getattr(rcv, 'data', None), dft_modes or u


def adjoint(model, y, src_coords, rcv_coords, space_order=8, q=0,
            save=False, free_surface=False):
    """
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Setting adjoint wavefield
    v = wavefield(model, space_order, save=save, nt=y.shape[0], fw=False)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, q=q, fw=False, fs=free_surface)

    # Setup source and receiver
    geom_expr, _, rcv = src_rec(model, v, src_coords=rcv_coords,
                                rec_coords=src_coords, wavelet=y, fw=False)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr, subs=subs, name="adjoint"+name(model))
    op(**op_kwargs(model, fs=free_surface))

    # Output
    return getattr(rcv, 'data', None), v


def gradient(model, residual, rcv_coords, u, return_op=False, space_order=8,
             w=None, free_surface=False, freq=None, dft_sub=None,):
    """
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Setting adjoint wavefieldgradient
    v = wavefield(model, space_order, fw=False)

    # Set up PDE expression and rearrange
    pde = wave_kernel(model, v, fw=False, fs=free_surface)

    # Setup source and receiver
    geom_expr, src, _ = src_rec(model, v, src_coords=rcv_coords,
                                wavelet=residual, fw=False)

    # Setup gradient wrt m
    gradm = Function(name="gradm", grid=model.grid)
    w = w or model.grid.time_dim.spacing * model.irho
    g_expr = grad_expr(gradm, u, v, w=w, freq=freq, dft_sub=dft_sub)

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + g_expr, subs=subs, name="gradient"+name(model))

    if return_op:
        return op, gradm
    op(**op_kwargs(model, fs=free_surface))

    # Output
    return gradm.data


def born(model, src_coords, rcv_coords, wavelet, space_order=8,
         save=False, free_surface=False):
    """
    Compute adjoint wavefield v = adjoint(F(m))*y
    and related quantities (||v||_w, v(xsrc))
    """
    # Setting adjoint wavefield
    u = wavefield(model, space_order, save=save, nt=wavelet.shape[0])
    ul = wavefield(model, space_order, name="l")
    
    # Set up PDE expression and rearrange
    pde = wave_kernel(model, u, fs=free_surface)
    pdel = wave_kernel(model, ul, q=lin_src(model, u), fs=free_surface)
    
    # Setup source and receiver
    geom_expr, _, _ = src_rec(model, u, src_coords=src_coords, wavelet=wavelet)
    geom_exprl, _, rcvl = src_rec(model, ul, rec_coords=rcv_coords, nt=wavelet.shape[0])

    # Create operator and run
    subs = model.spacing_map
    op = Operator(pde + geom_expr + pdel + geom_exprl, subs=subs,
                  name="born"+name(model))
    op(**op_kwargs(model, fs=free_surface))

    # Output
    return rcvl.data, u
