# -*- coding: utf-8 -*-

import os
import numpy as np
#from colour import CMFS, ILLUMINANTS_RELATIVE_SPDS, SpectralPowerDistribution
from colour.plotting import *
import colour
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# colour_plotting_defaults()
from skimage.color import colorconv, deltaE_ciede2000, deltaE_cie76
from skimage.transform import resize
import h5py
from spectral import *

### to avoid importing pyresources.assemple data
def dim_ordering_tf2th(img_list_ndarray):
    """
    convert ndarray with dimensions ordered as tf to th
    'tf' expects (nb_imgs, nb_rows, nb_cols, nb_channels) < -- compatible with plt.imshow(img_list[0,:,:,:])
    'th' expects (nb_imgs, nb_channels, nb_rows, nb_cols)

    Parameters
    ----------
    img_list_ndarray: ndarray
        Input ndarray of dimensions coherent with 'tf': (nb_imgs, nb_rows, nb_cols, nb_channels)

    Returns
    -------
    img_ndarray: ndarray
        Output ndarray of dimensions coherent with 'th': (nb_imgs, nb_channels, nb_rows, nb_cols)
    """
    if len(img_list_ndarray.shape) == 4:
        img_list_ndarray = np.rollaxis(img_list_ndarray, 3, 1)
    elif len(img_list_ndarray.shape) == 3:  # single image
        img_list_ndarray = np.rollaxis(img_list_ndarray, 2, 0)
    else:
        raise NotImplementedError('Input must be 3 or 4 dimnesional ndarray')



    return img_list_ndarray


def dim_ordering_th2tf(img_list_ndarray):
    """
    convert ndarray with dimensions ordered as th to tf
    'tf' expects (nb_imgs, nb_rows, nb_cols, nb_channels) < -- compatible with plt.imshow(img_list[0,:,:,:])
    'th' expects (nb_imgs, nb_channels, nb_rows, nb_cols)

    Parameters
    ----------
    img_list_ndarray: ndarray
        Input ndarray of dimensions coherent with 'th': (nb_imgs, nb_channels, nb_rows, nb_cols)

    Returns
    -------
    img_ndarray: ndarray
        Output ndarray of dimensions coherent with 'tf': (nb_imgs, nb_rows, nb_cols, nb_channels)
    """
    if len(img_list_ndarray.shape) == 4:
        img_list_ndarray = np.rollaxis(img_list_ndarray, 1, 4)
    elif len(img_list_ndarray.shape) == 3:  # single image
        img_list_ndarray = np.rollaxis(img_list_ndarray, 0, 3)
    else:
        raise NotImplementedError('Input must be 3 or 4 dimnesional ndarray')

    return img_list_ndarray

###
def sample_spectra_from_hsimg(hsimg, savename='spectra.png', image_data_format='channels_last'):
    """
    
    Parameters
    ----------
    hsimg
        Single hyperspectral image in dim_ordering shape
    savename

    Returns
    -------

    """

    if image_data_format == 'channels_first':
        hsimg = assemble_data.dim_ordering_th2tf(hsimg)
    elif image_data_format != 'channels_last':
        raise AttributeError('Wrong image_data_format')

    #now we should have channels_last
    h, w, c = hsimg.shape
    step_factor = 20
    h_step = h / step_factor
    w_step = w / step_factor

    # check that spectra are ok (E.g. increasing plots for redish pixels). Here there are still neg values!
    plt.close('all')
    plt.plot(hsimg[::h_step, ::w_step, :].reshape(-1, c).transpose())  # this is ok, checked that does not flip spectral axis
    plt.savefig(savename)
    plt.close('all')


def spectral2dummyRGB_img(spectral, rgb_channel_ixs=(22, 15, 6), image_data_format_in='channels_last', image_data_format_out='channels_last'):

    if image_data_format_in == 'channels_first':
        spectral = dim_ordering_th2tf(spectral)
    elif image_data_format_in != 'channels_last':
        raise AttributeError('Wrong image_data_format_in')

    dummyRGB = spectral[:,:,rgb_channel_ixs]

    if image_data_format_out == 'channels_first':
        dummyRGB = dim_ordering_tf2th(dummyRGB)
    elif image_data_format_out != 'channels_last':
        raise AttributeError('Wrong image_data_format_out')

    return dummyRGB


def spectral2sRGB_sample(spectral, cmfs):
    data = dict(zip(range(400, 701, 10), spectral))

    spd = colour.SpectralPowerDistribution('Sample', data)
    # illuminant = colour.ILLUMINANTS_RELATIVE_SPDS['D65']
    # xyz = colour.spectral_to_XYZ_integration(spd, cmfs, illuminant)  # xyz in [0, 100]
    XYZ = colour.spectral_to_XYZ_integration(spd, cmfs)  # xyz in [0, 100]
    sRGB = colour.XYZ_to_sRGB(XYZ=XYZ/100.)

    return sRGB

def spectral2XYZ_img_vectorized(cmfs, R):
    """
    
    Parameters
    ----------
    cmfs
    R:   np.ndarray (nb_pixels, 3) in [0., 1.]

    Returns
    -------

    """


    # playing with a illuminant to separate reflectance from it
    # S0 = colour.ILLUMINANTS_RELATIVE_SPDS['D50'].values[20:81:2] / 100.  # Equal-energy radiator (ones) sample_spectra_from_hsimg 300 to xxx with delta=5nm
    # print S

    # playing with reflectance
    # R = R/R.max()  # if we do this over each tile, when stiching them won't have the global lightness ok
    # print '\nRmin: ' + str(R.min())
    # print 'Rmax: ' + str(R.max())

    # R = np.ones(R.shape)
    # R[0:R.shape[0]/10,:] = 1.
    # R[R.shape[0]/10:R.shape[0]*2/10, :] = 0.5
    # R = R/S0

    # ix_wl_first = np.where(cmfs.wavelengths == 400.)[0][0]
    # ix_wl_last = np.where(cmfs.wavelengths == 701.)[0][0]
    # x_bar, y_bar, z_bar = colour.tsplit(cmfs.values[ix_wl_first:ix_wl_last:10, :])  #tested: OK. x_bar is the double one, the rightmost one (red). z_bar is the leftmost one (blue)
    x_bar, y_bar, z_bar = colour.tsplit(cmfs)  # tested: OK. x_bar is the double one, the rightmost one (red). z_bar is the leftmost one (blue)
    plt.close('all')
    plt.plot(np.array([z_bar, y_bar, x_bar]).transpose())
    plt.savefig('cmf_cie1964_10.png')
    plt.close('all')
    # S = colour.ones_spd(colour.STANDARD_OBSERVERS_CMFS.get('CIE 1964 10 Degree Standard Observer')).values[ix_wl_first:ix_wl_last:10, :]

    # illuminant. We assume that the captured R is reflectance with illuminant E (although it really is not, it is reflected radiance with an unknown illuminant, but the result is the same)
    S = colour.ILLUMINANTS_RELATIVE_SPDS['E'].values[20:81:2] / 100.  # Equal-energy radiator (ones) sample_spectra_from_hsimg 300 to xxx with delta=5nm
    # print S

    # R = spd.values
    # dw = cmfs.shape.interval
    dw = 10

    k = 100 / (np.sum(y_bar * S) * dw)
    # k = 100 / (np.sum(y_bar * S))

    X_p = R * x_bar * S * dw  # R(N,31) * x_bar(31,) * S(31,) * dw(1,)
    Y_p = R * y_bar * S * dw
    Z_p = R * z_bar * S * dw

    XYZ = k * np.sum(np.array([X_p, Y_p, Z_p]), axis=-1)
    XYZ = np.rollaxis(XYZ, 1, 0)  # th2tf() but for 2D input
    # print XYZ
    return XYZ

def spectral2XYZ_img(hs, cmf_name, image_data_format='channels_last'):
    """
    Convert spectral image input to XYZ (tristimulus values) image

    Parameters
    ----------
    hs:    numpy.ndarray
        3 dimensional numpy array containing the spectral information in either (h,w,c) ('channels_last') or (c,h,w) ('channels_first') formats 
    cmf_name:   basestring
        String describing the color matching functions to be used
    image_data_format:   basestring {'channels_last', 'channels_first'}. Default: 'channels_last'
        Channel dimension ordering of the input spectral image. the rgb output will follow the same dim ordering format

    Returns
    -------
    XYZ:    numpy.ndarray
        3 dimensional numpy array containing the tristimulus value information in either (h,w,3) ('channels_last') or (3,h,w) ('channels_first') formats

    """
    if image_data_format == 'channels_first':
        hs = dim_ordering_th2tf(hs)  # th2tf (convert to channels_last

    elif image_data_format == 'channels_last':
        pass
    else:
        raise AttributeError('Wrong image_data_format parameter ' + image_data_format)

    # flatten
    h, w, c = hs.shape
    hs = hs.reshape(-1, c)

    cmfs = get_cmfs(cmf_name=cmf_name, nm_range=(400., 700.), nm_step=10, split=False)

    XYZ = spectral2XYZ_img_vectorized(cmfs, hs)  # (nb_px, 3)

    # recover original shape (needed to call to xyz2rgb()
    XYZ = XYZ.reshape((h, w, 3))

    if image_data_format == 'channels_first':
        # convert back to channels_first
        XYZ = dim_ordering_tf2th(XYZ)

    return XYZ


def spectral2sRGB_img(spectral, cmf_name, image_data_format='channels_last'):
    """
    Convert spectral image input to rgb image
    
    Parameters
    ----------
    spectral:    numpy.ndarray
        3 dimensional numpy array containing the spectral information in either (h,w,c) ('channels_last') or (c,h,w) ('channels_first') formats 
    cmf_name:   basestring
        String describing the color matching functions to be used
    image_data_format:   basestring {'channels_last', 'channels_first'}. Default: 'channels_last'
        Channel dimension ordering of the input spectral image. the rgb output will follow the same dim ordering format

    Returns
    -------
    rgb:    numpy.ndarray
        3 dimensional numpy array containing the spectral information in either (h,w,3) ('channels_last') or (3,h,w) ('channels_first') formats
    
    """

    XYZ = spectral2XYZ_img(hs=spectral, cmf_name=cmf_name, image_data_format=image_data_format)

    if image_data_format == 'channels_first':
        XYZ = dim_ordering_th2tf(XYZ)  # th2tf (convert to channels_last

    elif image_data_format == 'channels_last':
        pass
    else:
        raise AttributeError('Wrong image_data_format parameter ' + image_data_format)

    #we need to pass in channels_last format to xyz2rgb
    sRGB = colorconv.xyz2rgb(XYZ/100.)

    if image_data_format == 'channels_first':
        # convert back to channels_first
        sRGB = dim_ordering_tf2th(sRGB)


    return sRGB


def save_hs_as_envi(fpath, hs31, image_data_format_in='channels_last'):#, image_data_format_out='channels_last'):
    #output is always channels_last
    if image_data_format_in == 'channels_first':
        hs31 = dim_ordering_th2tf(hs31)
    elif image_data_format_in != 'channels_last':
        raise AttributeError('Wrong image_data_format_in')

    # dst_dir = os.path.dirname(fpath)

    # hdr_fpath = os.path.splitext(fpath)[0] + '.hdr'
    hdr_fpath = fpath + '.hdr'
    wl = np.arange(400, 701, 10)

    hs31_envi = envi.create_image(hdr_file=hdr_fpath,
                                  metadata=generate_metadata(wl=wl),
                                  shape=hs31.shape,  # Must be in (Rows, Cols, Bands)
                                  force=True,
                                  dtype=np.float32,  # np.float32, 32MB/img  np.ubyte: 8MB/img
                                  ext='.envi31')
    mm = hs31_envi.open_memmap(writable=True)
    mm[:, :, :] = hs31

def restore_img_range(img):
    """
    Works for any number of channels.
    Parameters
    ----------
    img

    Returns
    -------

    """
    # img = img * 127.5 + 127.5  # now in [0., 255.]
    # img /= 255.  # now in [0., 1.]  commment if dtype is np.ubyte in create_image()
    img = img * 0.5 + 0.5  # now in [0., 1.]
    # img /= 255.  # now in [0., 1.]  commment if dtype is np.ubyte in create_image()
    # print hs31.min()
    # print hs31.max()

    return img

def generate_metadata(wl):
    md = dict()
    md['interleave'] = 'bsq'  # (Rows, Cols, Bands) <->(lines, samples, bands)
    md['data type'] = 12
    md['wavelength'] = wl
    md['default bands'] = [22, 15, 6]  # for spectral2dummyRGB
    md['fwhm'] = np.diff(wl)
    # md['vroi'] = [1, len(wl)]

    return md

def load_envi(fpath_envi, fpath_hdr=None):
    if fpath_hdr is None:
        fpath_hdr = os.path.splitext(fpath_envi)[0] + '.hdr'

    hs = io.envi.open(fpath_hdr, fpath_envi)

    return hs


def get_cmfs(cmf_name='cie1964_10', nm_range=(400., 700.), nm_step=10, split=True):

    if cmf_name == 'cie1931_2':
        cmf_full_name = 'CIE 1931 2 Degree Standard Observer'
    elif cmf_name == 'cie1931_10':
        cmf_full_name = 'CIE 1931 10 Degree Standard Observer'
    elif cmf_name == 'cie1964_2':
        cmf_full_name = 'CIE 1964 2 Degree Standard Observer'
    elif cmf_name == 'cie1964_10':
        cmf_full_name = 'CIE 1964 10 Degree Standard Observer'
    else:
        raise AttributeError('Wrong cmf name')
    cmfs = colour.STANDARD_OBSERVERS_CMFS[cmf_full_name]

    # subsample and trim range
    ix_wl_first = np.where(cmfs.wavelengths == nm_range[0])[0][0]
    ix_wl_last = np.where(cmfs.wavelengths == nm_range[1] + 1.)[0][0]
    cmfs = cmfs.values[ix_wl_first:ix_wl_last:int(nm_step), :]  # make sure the nm_step is an int

    if split:
        x_bar, y_bar, z_bar = colour.tsplit(cmfs)  #tested: OK. x_bar is the double one, the rightmost one (red). z_bar is the leftmost one (blue)
        return x_bar, y_bar, z_bar
    else:
        return cmfs


# def GFC(ref_spectrum, test_spectrum):
#     """
#     Goodness of fit metric as defined in:
#     J. Hernández-Andrés, J. Romero, and J., Raymond L. Lee, “Colorimetric and spectroradiometric characteristics of narrow-field-of-view clear skylight in Granada, Spain,” J. Opt. Soc. Am. A, vol. 18, no. 2, pp. 412–420, Feb. 2001.
#     Used and compared in:
#     F. H. Imai, M. R. Rosen, and R. S. Berns, “Comparative study of metrics for spectral match quality,” in Conference on Colour in Graphics, Imaging, and Vision, 2002, vol. 2002, pp. 492–496.
#     We asume equal length
#     """
#     max_value_in_pair = float(max(np.concatenate((ref_spectrum, test_spectrum))))
#     print max_value_in_pair
#     print max(ref_spectrum)
#     ref_spectrum = ref_spectrum/max_value_in_pair
#     print max(test_spectrum)
#     test_spectrum = test_spectrum/max_value_in_pair
#     gfc = np.abs(np.sum(test_spectrum * ref_spectrum)) / (np.sqrt(np.abs(np.sum(test_spectrum**2))) * np.sqrt(np.abs(np.sum(ref_spectrum**2))))

    #print 'gfc: ' +str(gfc)
    # return gfc

# def gfc_map(ref_spectra, test_spectra):
#     max_value_in_pair = float(max(np.concatenate((ref_spectra, test_spectra))))
#     print max_value_in_pair
#     print max(ref_spectra)
#     ref_spectra = ref_spectra / max_value_in_pair
#     print max(test_spectra)
#     test_spectra = test_spectra / max_value_in_pair
#     gfc = np.abs(np.sum(test_spectra * ref_spectra)) / (
#     np.sqrt(np.abs(np.sum(test_spectra ** 2))) * np.sqrt(np.abs(np.sum(ref_spectra ** 2))))
#
#     print 'gfc: ' +str(gfc)
    # return gfc


def get_gfc(real, pred, dim_ordering='channels_last'):
    """
    Compute GFC metrics for spectral image pair.

    Parameters
    ----------
    real
    pred


    Returns
    -------
    gfc:    ndarray
        Same spatial dimensions as originals, single channel. Range: [0, 1]

    """
    if real is None or pred is None:
        raise AttributeError('Need both (real and pred)')

    if dim_ordering == 'channels_last':
        dim_ch = 2
    elif dim_ordering == 'channels_first':
        dim_ch = 0
    else:
        raise AttributeError('Wrong attribute dim_ordering')

    #fixme do we need to divide by max?
    gfc_map = np.sum(real * pred, axis=dim_ch) / (np.sqrt(np.sum(real ** 2, axis=dim_ch)) * np.sqrt(np.sum(pred ** 2, axis=dim_ch)))

    return gfc_map


def get_signed_error_cube(real, pred):
    return pred - real


def get_deltae00_from_spectral(hs1, hs2, cmf_name='cie1964_10', deltae_illum='D65'):
    XYZ1 = spectral2XYZ_img(hs1, cmf_name, image_data_format='channels_first')
    XYZ2 = spectral2XYZ_img(hs2, cmf_name, image_data_format='channels_first')

    return deltaE_ciede2000(lab1=colorconv.xyz2lab(xyz=assemble_data.dim_ordering_th2tf(XYZ1), illuminant=deltae_illum, observer='10'),  #fixme 10 not harcoded
                            lab2=colorconv.xyz2lab(xyz=assemble_data.dim_ordering_th2tf(XYZ2), illuminant=deltae_illum, observer='10'))  #fixme 10 not harcoded


def convert_to_rgb(img, cmf_name='cie1964_10', is_binary=False):
    """
    Given an image, make sure it has 3 channels and that it is between 0 and 1.

    Parameters
    ----------
    img: [-1,1]
    cmf_name
    is_binary

    Returns
    -------

    """

    if len(img.shape) != 3:
        raise Exception("""Image must have 3 dimensions (channels x height x width). """
                        """Given {0}""".format(len(img.shape)))

    img_ch, _, _ = img.shape

    imgp = img
    if img_ch == 1:
        imgp = np.repeat(img, 3, axis=0)

    if not is_binary:
        #  restore to [0,1] range
        # imgp = imgp * 127.5 + 127.5  # now in [0., 255.]
        # imgp /= 255.  # now in [0., 1.]

        # we no longer need to do this, since the model does it in GPU from g256_5 (?) and d256_4 (?) onwards
        # imgp = imgp * 0.5 + 0.5  # now in [0., 1.]
        pass

    if img_ch != 3 and img_ch != 1:
        # check that spectra are ok (E.g. increasing plots for redish pixels). Here there are still neg values!
        # color.sample_spectra_from_hsimg(hsimg=imgp, savename='cmfs3.png', image_data_format='channels_first')

        # imgp is channels_first
        imgp = spectral2sRGB_img(spectral=imgp, cmf_name=cmf_name, image_data_format='channels_first')
        # imgp = color.spectral2dummyRGB_img(imgp, rgb_channel_ixs=(22, 15, 6),
        #                                    image_data_format_in='channels_first', image_data_format_out='channels_first')
        # selected_channels = (22, 15, 6)
        # img = img[selected_channels, :, :]
        # raise Exception("""Unsupported number of channels. """
        #                 """Must be 1 or 3, given {0}.""".format(img_ch))

    return np.clip(imgp.transpose((1, 2, 0)), 0, 1)

