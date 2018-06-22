import numpy as np
from scipy import ndimage, interpolate
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model
import matplotlib.pyplot as plt
from TTI_operators import forward_modeling, adjoint_born

if __name__ == "__main__":
    description = ("Overthrust RTM")
    parser = ArgumentParser(description=description)
    parser.add_argument('-t', dest='t_sub', default=20, type=int,
                        help="Time subsampling factor")
    parser.add_argument('-tti', dest='is_phi', default=False, action='store_true',
                        help="Wether or not to use azymuth")
    parser.add_argument('-b', dest='born', default=False, action='store_true',
                        help="Wether or not generate linearized data")
    parser.add_argument('-isic', dest='isic', default=False, action='store_true',
                        help="Wether or not use inverse scatering imaging condition")

    print("running with setup \n isic: %s \n asimuth: %s \n Born: %s \n time subsampling rate: %s" % (args.isic, args.is_phi, args.born,args.t_sub))
    # Load original v and tti params
    v = np.fromfile("overthrust_vp.rsf@", sep="", dtype=np.float32, count=801*801*187)
    v = np.reshape(v, (801, 801, 187))/1350




    # Cut wanted part and interpolate
    buffer_cut_x = 1000. # (in m)
    buffer_cut_y = 500. # (in m)
    xsrc = 5000
    ysrc = 5000
    indstartx = int((xsrc - buffer_cut_x) / 25.)
    xrecmax = 13000.
    indendx = int((xrecmax + buffer_cut_x) / 25.)
    #for 3d
    yrecmax = ysrc+450.
    yrecmin = ysrc-450.
    indstarty = int((yrecmin - buffer_cut_y) / 25.)
    indendy = int((yrecmax + buffer_cut_y) / 25.)


    # indices for the final sum of RTM images
    start_rtm_x = int((xsrc - buffer_cut_x) / 15.)
    end_rtm_x = int((xrecmax + buffer_cut_x) / 15.)
    start_rtm_y = int((yrecmin - buffer_cut_y) / 15.)
    end_rtm_y = int((yrecmax + buffer_cut_y) / 15.)
    # cut v
    v_cut = v[indstartx:indendx+1, indstarty:indendy+1,:]
    shape_orig = v_cut.shape

    # Setup interpolation
    x = [i*25 for i in range(shape_orig[0])]
    xnew = [i*5 for i in range(5*(shape_orig[0]-1) + 1)]
    y = [i*25 for i in range(shape_orig[1])]
    ynew = [i*5 for i in range(5*(shape_orig[1]-1) + 1)]
    z = [i*25 for i in range(shape_orig[2])]
    znew = [i*5 for i in range(5*(shape_orig[2]-1) + 1)]
    nx = len(xnew)
    ny = len(ynew)
    nz = len(znew)

    v_cut_fine = np.zeros((nx, ny, nz), dtype=np.float32)

    # for i in range(shape_orig[0]):
    #     f = interpolate.interp2d(z, y, v_cut[i, :, :], kind='linear')
    #     v_cut_fine[i*5, :, :] = f(znew, ynew)
    #
    # for i in range(ny):
    #     for j in range(nz):
    #         f = interpolate.interp1d(x, v_cut_fine[::5, i, j], kind='linear')
    #         v_cut_fine[:, i, j] = f(xnew)
    # Add water layer
    v_cut_fine = np.concatenate((1.5*np.ones((v_cut_fine.shape[0], v_cut_fine.shape[1], 200)), v_cut_fine), axis=2)

    # Setup model

    spacing = (5.0, 5.0)
    shape = v_cut_fine.shape
    origin = (xsrc - buffer_cut_x, yrecmin - buffer_cut_y, 0.)

    # Fake TTI for testing
    epsilon = (v_cut_fine - 1.5) / 10
    delta = (v_cut_fine - 1.5) / 15
    theta = (v_cut_fine - 1.5) / 10
    phi = .5 * theta if args.is_phi else None

    print("original size was %s, resampled size after cut is %s" % (shape_orig, shape))
    model = Model(origin, spacing, shape, v_cut_fine, nbpml=40,
                  epsilon=epsilon, delta=delta, theta=theta, phi=phi, space_order=12)

    # ################### Source and rec ################### 
    # # Source
    t0 = 0.
    tn = 3500.
    dt = model.critical_dt
    print(dt)
    nt = int(1 + (tn-t0) / dt)
    time = np.linspace(t0,tn,nt)
    f0 = 0.040
    src = RickerSource(name='src', grid=model.grid, f0=f0, time=time)
    src.coordinates.data[0, 0] = xsrc
    src.coordinates.data[0, -1] = 20.
    # Receiver for observed data
    nrec = 321
    nline = 10
    rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec*nline, ntime=nt)
    for i in range(nline):
        rec_t.coordinates.data[i*321:(i+1)*321:, 0] = np.linspace(xsrc, xrecmax, nrec)
        rec_t.coordinates.data[i*321:(i+1)*321, 1] = yrecmax - i*100.

    numel = np.prod([d+80 for d in shape])
    numel_sub = np.prod([d+80 for d in shape])/27
    t_sub_factor = args.t_sub
    n_save = int(nt / t_sub_factor)+2

    n_full_fields = 6 + 6 if args.is_phi else 5 +6
    n_sub_fields = n_save * 2 + 1

    size_shot_rec = (3+nt)*(nrec*nline)

    print("estimated memory usage is %s Gb" % (4*(numel * n_full_fields + numel_sub * n_sub_fields + size_shot_rec)))

    # # Observed data
    # dobs, utrue, v1 = forward_modeling(model, src.coordinates.data, src.data, rec_t.coordinates.data, save=True, t_sub_factor=t_sub_factor)
    # if args.born:
    #     dD, _, _ = forward_born(model, src.coordinates.data, src.data, rec_t.coordinates.data)
    # g1 = adjoint_born(model, rec_t.coordinates.data, dD if args.born else dobs, u=utrue, v=v1, dt=dt, isic=args.isic)
