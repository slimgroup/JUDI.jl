import numpy as np
from argparse import ArgumentParser
from scipy import ndimage, interpolate
from obspy.signal.filter import highpass
from PySource import RickerSource, PointSource, Receiver
from PyModel import Model
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
    parser.add_argument('-t', dest='t_sub', default=80, type=int,
                        help="Time subsampling factor")
    parser.add_argument('-tti', dest='is_phi', default=False, action='store_true',
                        help="Wether or not to use azymuth")
    parser.add_argument('-b', dest='born', default=False, action='store_true',
                        help="Wether or not generate linearized data")
    parser.add_argument('-isic', default='noop', choices=['noop', 'isotropic', 'rotated'],
                        help="Wether or not use inverse scatering imaging condition")
    args = parser.parse_args()
    print("running with setup \n isic: %s \n asimuth: %s \n Born: %s \n time subsampling rate: %s" % (args.isic, args.is_phi, args.born,args.t_sub))
    # Load original v and tti params
    # Replace by s3 load
    v = get_from_s3('Overthrust_tti/over_2d_vp.npy')
    print(v.shape)
    # Cut wanted part and interpolate
    buffer_cut_x = 1000. # (in m)
    xsrc = 5000
    indstartx = int((xsrc - buffer_cut_x) / 25.)
    xrecmax = 13000.
    indendx = int((xrecmax + buffer_cut_x) / 25.)

    # indices for the final sum of RTM images
    start_rtm_x = int((xsrc - buffer_cut_x) / 15.)
    end_rtm_x = int((xrecmax + buffer_cut_x) / 15.)
    # cut v
    v_cut = v[indstartx:indendx+1, :]
    shape_orig = v_cut.shape

    # Setup interpolation
    x = [i*25 for i in range(shape_orig[0])]
    xnew = [i*7.5 for i in range(int(25/7.5*(shape_orig[0]-1) + 1))]
    z = [i*25 for i in range(shape_orig[1])]
    znew = [i*7.5 for i in range(int(25/7.5*(shape_orig[1]-1) + 1))]
    nx = len(xnew)
    nz = len(znew)

    f = interpolate.interp2d(z, x, v_cut, kind='linear')
    v_cut_fine = f(znew, xnew)

    v_cut_fine = np.concatenate((1.5*np.ones((v_cut_fine.shape[0], 100)), v_cut_fine), axis=1)

    v_cut_fine0 = ndimage.gaussian_filter(v_cut_fine, sigma=5)
    dm = 1/v_cut_fine0**2 - 1/v_cut_fine**2
    # Setup model

    spacing = (7.5, 7.5)
    shape = v_cut_fine.shape
    origin = (xsrc - buffer_cut_x, 0.)

    # Fake TTI for testing
    epsilon =(v_cut_fine - 1.5) / 10
    delta = (v_cut_fine - 1.5) / 15
    theta = (v_cut_fine - 1.5) / 10

    print("original size was %s, resampled size after cut is %s" % (shape_orig, shape))
    model = Model(origin, spacing, shape, v_cut_fine0, nbpml=40, dm=dm,
                  epsilon=epsilon, delta=delta, theta=theta, space_order=12)

    # ################### Source and rec ################### 
    # # Source
    t0 = 0.
    tn = 4500.
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
    f = interpolate.interp1d(twave, wavelet, kind='linear')
    src.data[:len(tnew), 0] = f(tnew)
    # Receiver for observed data
    nrec = 321
    rec_t = Receiver(name='rec_t', grid=model.grid, npoint=nrec, ntime=nt)
    rec_t.coordinates.data[:, 0] = np.linspace(xsrc, xrecmax, nrec)
    rec_t.coordinates.data[:, 1] = 1000.

    numel = np.prod([d+80 for d in shape])
    numel_sub = np.prod([d+80 for d in shape])/4
    t_sub_factor = args.t_sub
    n_save = int(nt / t_sub_factor) + 2

    n_full_fields = 20 if args.born else 24
    n_sub_fields = n_save * 2 + 1

    size_shot_rec = (3+nt)*(nrec)

    print("estimated memory usage is %s Gb" % (4*(numel * n_full_fields + numel_sub * n_sub_fields + size_shot_rec)/(1024**3)))

    # Observed data
    space_order = 20
    if args.born:
        dD, utrue, v1 = forward_born(model, src.coordinates.data, src.data, rec_t.coordinates.data, save=True, t_sub_factor=t_sub_factor, h_sub_factor=2, space_order=space_order)
    else:
        dD, utrue, v1 = forward_modeling(model, src.coordinates.data, src.data, rec_t.coordinates.data, save=True, t_sub_factor=t_sub_factor, h_sub_factor=2, space_order=space_order)
    g1 = adjoint_born(model, rec_t.coordinates.data, dD.data[:], ph=utrue, pv=v1, isic=args.isic, space_order=space_order)
    # Save to S3 bucket.
    from IPython import embed; embed()
    write_to_s3(g1, name='2d-grad.npy', ext='Overthrust_tti/')
