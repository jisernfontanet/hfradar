# System modules

import os

# Import scientific modules

import numpy as np


def lmercator(xin_, yin_, lon0=0, lat0=0, radius=6371000.0, inverse=False):
    """
    Compute the Mercator projection centered at the center of the image

    Parameters
    -----
    xin: ndarray
        Longitude (inverse=False) or x-coordinate (inverse=True)
    yin: ndarray
        Latitude (inverse=False) or y-coordinate (inverse=True)
    lon0: float, optional, default=0
        Central longitude used to compute the projection (inverse=True)
    lat0: float, optional, default=0
        Central latitude used to compute the projection (inverse=True)

    Returns
    -------
    xout: ndarray
        x-coordinate (inverse=False) or Longitude (inverse=True)
    yout: ndarray
        y-coordinate (inverse=False) or Latitude (inverse=True)
    """

    dtor = np.pi / 180
    rtod = 1 / dtor

    if inverse:
        x = xin_.copy()
        y = yin_.copy()
        lon = x / radius
        lat = 2 * np.arctan(np.exp(y / radius)) - np.pi / 2.
        xout = lon * rtod + lon0
        yout = lat * rtod + lat0
    else:
        lon = (xin_ - lon0) * dtor
        lat = (yin_ - lat0) * dtor
        x = radius * lon
        y = radius * np.log(np.tan(np.pi / 4 + lat / 2.))
        xout = x
        yout = y

    return xout, yout


def get_hfr_noise(datain, maxit=10, ksigma=3, return_coeff=False, nobs=None):

    # B3-Splines

    wfilter = np.array([1, 4, 6, 4, 1]) / 16
    index = np.array([-2, -1, 0, 1, 2])
    sigmae = 0.7

    # Prepare the data

    data = datain.data
    data[datain.mask] = 0
    if len(data.shape) == 1:
        data = datain.data[np.newaxis, :, np.newaxis]
        mask = datain.mask[np.newaxis, :, np.newaxis]
    elif len(data.shape) == 2:
        data = datain.data[np.newaxis, :, :]
        mask = datain.mask[np.newaxis, :, :]
    else:
        data = datain.data
        mask = datain.mask
    data[mask] = 0
    mask = np.logical_not(mask).astype(int)

    # Get dimensions

    nt, nr, nb = data.shape

    # NUmber of observations

    if nobs is None:
        nobs = np.ones((nt, nr, nb))

    # Build the Low-pass filter (à trous wavelet transform) for the first scale

    j = 0

    indexstep = index * 2 ** j
    indexstep = indexstep - np.min(indexstep)

    filterstep = np.zeros(np.max(indexstep) + 1)
    filterstep[indexstep] = wfilter

    # Convolve over ranges to get the wavelet coefficients

    w1 = np.zeros((nt, nr, nb))

    for n in np.arange(nt):
        for m in np.arange(nb):
            w1[n, :, m] = data[n, :, m] - np.convolve(data[n, :, m], filterstep, mode='same')

    w1 = np.ma.array(data=w1, mask=mask == 0).reshape((nt, nb * nr))

    # Add the mask and normalize to get noise. If the input field is the average of different
    # values, correct for this. Normalize to the maximum number of observations

    w1 = w1 / sigmae * np.sqrt(nobs.reshape((nt, nb * nr))) / np.sqrt(np.max(nobs))

    # Apply k-sigma clipping over each time to estimate noise std

    for n in np.arange(maxit):
        sigma = np.std(w1, axis=1)
        w1.mask = np.logical_or(w1.mask, np.abs(w1.data) > ksigma * sigma[:, np.newaxis])
    sigma = np.std(w1, axis=1)

    # Depending on the output, return wavelet coefficients

    if return_coeff:
        #w1 = w1 / sigma[:, np.newaxis]
        return sigma, w1.reshape((nt, nr, nb))
    else:
        return sigma
