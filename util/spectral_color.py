# -*- coding: utf-8 -*-

import os
import numpy as np
from colour.plotting import *
import colour
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.color import colorconv
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


def spectral2XYZ_img_vectorized(cmfs, R):
    """
    
    Parameters
    ----------
    cmfs
    R:   np.ndarray (nb_pixels, 3) in [0., 1.]

    Returns
    -------

    """

    x_bar, y_bar, z_bar = colour.tsplit(cmfs)  # tested: OK. x_bar is the double one, the rightmost one (red). z_bar is the leftmost one (blue)
    plt.close('all')
    plt.plot(np.array([z_bar, y_bar, x_bar]).transpose())
    plt.savefig('cmf_cie1964_10.png')
    plt.close('all')
    # illuminant. We assume that the captured R is reflectance with illuminant E (although it really is not, it is reflected radiance with an unknown illuminant, but the result is the same)
    S = colour.ILLUMINANTS_RELATIVE_SPDS['E'].values[20:81:2] / 100.  # Equal-energy radiator (ones) sample_spectra_from_hsimg 300 to xxx with delta=5nm
    # print S

    # dw = cmfs.shape.interval
    dw = 10

    k = 100 / (np.sum(y_bar * S) * dw)

    X_p = R * x_bar * S * dw  # R(N,31) * x_bar(31,) * S(31,) * dw(1,)
    Y_p = R * y_bar * S * dw
    Z_p = R * z_bar * S * dw

    XYZ = k * np.sum(np.array([X_p, Y_p, Z_p]), axis=-1)
    XYZ = np.rollaxis(XYZ, 1, 0)  # th2tf() but for 2D input

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

