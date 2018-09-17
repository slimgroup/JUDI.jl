import numpy as np
from argparse import ArgumentParser
from scipy import ndimage, interpolate
from obspy.signal.filter import highpass
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model
import matplotlib.pyplot as plt
from TTI_Staggered import forward_modeling, adjoint_born, forward_born

from io import BytesIO
import boto3
import os

def write_to_s3(array, name='file.npy', ext=''):
    # Create an S3 client
    s3 = boto3.client('s3')

    filename = name

    np.save(filename, array)

    bucket_name = 'slim-bucket-common'

    # Uploads the given file using a managed uploader, which will split up large
    # files automatically and upload parts in parallel.
    s3.upload_file(filename, bucket_name, ext+filename)
    os.remove(filename)

def get_from_s3(filename):

    client = boto3.client('s3')


    obj = client.get_object(Bucket='slim-bucket-common', Key=filename)
    return np.load(BytesIO(obj['Body'].read()))


if __name__ == "__main__":
    description = ("Overthrust RTM")
    parser = ArgumentParser(description=description)
    parser.add_argument('-t', dest='t_sub', default=20, type=int,
                        help="Time subsampling factor")
    parser.add_argument('-tti', dest='is_phi', default=False, action='store_true',
                        help="Wether or not to use azymuth")
    parser.add_argument('-b', dest='born', default=False, action='store_true',
                        help="Wether or not generate linearized data")
    parser.add_argument("-name", default="file",
                        help="filename for saving")
    parser.add_argument('-isic', default='noop', choices=['noop', 'isotropic', 'rotated'],
                        help="Wether or not use inverse scatering imaging condition")
    args = parser.parse_args()
    print("running with setup \n isic: %s \n asimuth: %s \n Born: %s \n time subsampling rate: %s" % (args.isic, args.is_phi, args.born,args.t_sub))
    # Load original v and tti params
    # Replace by s3 load
    # v = get_from_s3('Overthrust_tti/over_3d_vp.npy')
    v = np.load('/data/mlouboutin3/overthrust/over_3d_vp.npy')

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
    xnew = [i*7.5 for i in range(int(25/7.5*(shape_orig[0]-1) + 1))]
    y = [i*25 for i in range(shape_orig[1])]
    ynew = [i*7.5 for i in range(int(25/7.5*(shape_orig[1]-1) + 1))]
    z = [i*25 for i in range(shape_orig[2])]
    znew = [i*7.5 for i in range(int(25/7.5*(shape_orig[2]-1) + 1))]
    nx = len(xnew)
    ny = len(ynew)
    nz = len(znew)


    # Estimate memory usage
    nrec = 321
    nline = 10
    numel = np.prod([d+80 for d in (nx, ny, nz)])
    numel_sub = np.prod([d+80 for d in (nx, ny, nz)])/8
    t_sub_factor = args.t_sub
    n_save = int(12355 / t_sub_factor) + 2

    n_full_fields = 18 if args.is_phi else 17
    n_sub_fields = n_save * 2 + 1

    size_shot_rec = (3+12355)*(nrec*nline)

    print("estimated memory usage is %s Gb" % (4*(numel * n_full_fields + numel_sub * n_sub_fields + size_shot_rec)/(1024**3)))

    # Interpolate onto new grid
    interpolator = interpolate.RegularGridInterpolator((x, y, z), v_cut)
    gridnew = np.ix_(xnew, ynew, znew)
    v_cut_fine = interpolator(gridnew)

    # Add water layer
    v_cut_fine = np.concatenate((1.5*np.ones((v_cut_fine.shape[0], v_cut_fine.shape[1], 100)), v_cut_fine), axis=2)
    v_cut_fine0 = ndimage.gaussian_filter(v_cut_fine, sigma=5)
    dm = 1/v_cut_fine0**2 - 1/v_cut_fine**2

    # Setup model
    spacing = (7.5, 7.5, 7.5)
    shape = v_cut_fine.shape
    origin = (xsrc - buffer_cut_x, yrecmin - buffer_cut_y, 0.)

    # Fake TTI for testing
    epsilon =  ndimage.gaussian_filter((v_cut_fine - 1.5) / 13, sigma=5)
    delta = ndimage.gaussian_filter((v_cut_fine - 1.5) / 15, sigma=5)
    theta = ndimage.gaussian_filter((v_cut_fine - 1.5) / 110, sigma=8)
    phi = .5 * theta if args.is_phi else None

    print("original size was %s, resampled size after cut is %s" % (shape_orig, shape))
    model = Model(origin, spacing, shape, v_cut_fine0, nbpml=40, dm=dm,
                  epsilon=epsilon, delta=delta, theta=theta, phi=phi, space_order=16)

    epsilon = []
    delta = []
    theta = []
    phi = []
    # ################### Source and rec ################### 
    # # Source
    t0 = 0.
    tn = 6000.
    dt = model.critical_dt
    nt = int(1 + (tn-t0) / dt)
    time = np.linspace(t0,tn,nt)
    f0 = 0.040
    src = RickerSource(name='src', grid=model.grid, f0=f0, time=time)
    src.coordinates.data[0, 0] = xsrc
    src.coordinates.data[0, -1] = 20.

    wavelet = get_from_s3('Overthrust_tti/wavelet.npy')
    nn = wavelet.shape[0]
    timestep = .002
    filtered = highpass(wavelet, 10, 1/timestep, corners=1, zerophase=True)

    twave = [i*2 for i in range(nn)]
    tnew = [i*dt for i in range(int(1 + (twave[-1]-t0) / dt))]
    f = interpolate.interp1d(twave, filtered, kind='linear')
    src.data[:len(tnew), 0] = f(tnew)
    # Receiver for observed data
    rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec*nline, ntime=nt)
    for i in range(nline):
        rec_t.coordinates.data[i*321:(i+1)*321:, 0] = np.linspace(xsrc, xrecmax, nrec)
        rec_t.coordinates.data[i*321:(i+1)*321, 1] = yrecmax - i*100.
        rec_t.coordinates.data[i*321:(i+1)*321, 2] = 50.

    # Observed data
    # if args.born:
    #     dD, utrue, v1 = forward_born(model, src.coordinates.data, src.data, rec_t.coordinates.data, save=True, t_sub_factor=t_sub_factor, h_sub_factor=2, space_order=20)
    # else:
    #     dD, utrue, v1 = forward_modeling(model, src.coordinates.data, src.data, rec_t.coordinates.data, save=True, t_sub_factor=t_sub_factor, h_sub_factor=2, space_order=20)
    # g1 = adjoint_born(model, rec_t.coordinates.data, dD.data[:], ph=utrue, pv=v1, isic=args.isic, space_order=20)
    # # Save to S3 bucket.
    # write_to_s3(g1, name='3d-grad.npy', ext='Overthrust_tti.npy')
