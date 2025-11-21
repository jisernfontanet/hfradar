# System modules

import os

# Import scientific modules

import numpy as np


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

    # Add the mask and normalize to get noise. If the input field is the averagen of differentn
    # value, correct for this. Normalize to the maximum number of observations

    w1 = w1 / sigmae * np.sqrt(nobs.reshape((nt, nb * nr))) / np.sqrt(np.max(nobs))
    # w1 = w1 / sigmae * np.sqrt(nobs.reshape((nt, nb * nr)))

    # Apply k-sigma clipping over each time to estimate noise std

    for n in np.arange(maxit):
        sigma = np.std(w1, axis=1)
        w1.mask = np.logical_or(w1.mask, np.abs(w1.data) > ksigma * sigma[:, np.newaxis])
    sigma = np.std(w1, axis=1)

    # Depending on the output, return wavelet coefficients

    if return_coeff:
        w1 = w1 / sigma[:, np.newaxis]
        return sigma, w1.reshape((nt, nr, nb))
    else:
        return sigma
