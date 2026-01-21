# System modules

import os

# Import scientific modules

import numpy as np


def lmercator(xin_, yin_, lon0=0, lat0=0, radius=6371000.0, inverse=False):
    """
    Compute the Mercator projection centered at the center of the image

    Parameters
    ----------
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


def synchronize(xref, x, dxmax=None):
    """
    Given a refernce coordinate xref, find the closest point of x to each point in xref
    """

    n, = xref.shape
    index = np.zeros(n, dtype=np.int64)
    for i in np.arange(n):
        j = np.argmin(np.abs(x - xref[i]))
        index[i] = j

    if dxmax is not None:
        dx = np.abs(xref - x[index])
        out = np.ma.array(data=index, mask=dx > dxmax)
    else:
        out = np.ma.array(data=index, mask=np.zeros(n, dtype=bool))

    return out


def getinfo(ds, nmin=10, flagname='PRIM', flagvalue=1):
    # Flag the valid values

    flag = xr.zeros_like(ds[flagname], dtype=int)
    flag.data[ds[flagname].data == flagvalue] = 1

    # Count the number of points and flag invalid data

    mask = flag.sum(dim='time').astype(int).data < nmin

    # Mask out lat/lon points with not enough valid data

    lon = np.ma.array(data=ds.lon.data, mask=mask)
    lat = np.ma.array(data=ds.lat.data, mask=mask)

    return lon, lat, flag


def get_rmse_pairs(ds1, ds2, maxdist=150, nmin=10, flagname='PRIM', flagvalue=1):
    """
    Compute the Root Mean Square Error between two time-series
    """
    # Parameters

    bname = 'bearing'  # Name for the bearing coordinate
    velname = 'VELO'  # Name for the radial velocity
    dtmax = np.timedelta64(55, 'm')  # Maximum seoaration of time

    # Sincronize time series. Get the indices that match the same times in both time series

    it2 = synchronize(ds1.time.data, ds2.time.data, dxmax=dtmax)
    it1 = np.arange(ds1.time.data.shape[0])[np.logical_not(it2.mask)]
    it2 = it2.data[np.logical_not(it2.mask)]

    # Extract information from the DtaSets in the needed format

    lon1, lat1, flag1 = getinfo(ds1, nmin=nmin, flagname=flagname, flagvalue=flagvalue)
    lon2, lat2, flag2 = getinfo(ds2, nmin=nmin, flagname=flagname, flagvalue=flagvalue)

    # Find pairs of data. The first index corresponds to the range, the
    # second to the bearing

    ir1, ib1, ir2, ib2 = findpairs2d(lon1, lat1, lon2, lat2, maxdist=maxdist, latlon=True)

    # Compute the angle between bearings

    angle = ds1[bname][ib1].data - ds2[bname][ib2].data

    # Extract the data for the selected pairs. Use the sincronization indices computed before.
    # It does not exploits the flexibility of xarray, but it works.

    u1 = ds1[velname].data[:, ir1, ib1][it1, :]
    u2 = ds2[velname].data[:, ir2, ib2][it2, :]
    flag1 = flag1.data[:, ir1, ib1][it1, :]
    flag2 = flag2.data[:, ir2, ib2][it2, :]

    # Compute the RMSE of velocities

    npairs = ir1.shape[0]
    rmse = np.zeros(npairs)
    for k in np.arange(npairs):
        mask = np.logical_and(flag1[:, k] == 1, flag2[:, k] == 1)
        rmse[k] = np.sqrt(np.mean((u1[mask, k] + u2[mask, k]) ** 2))

    return angle, rmse, ir1, ib1, ir2, ib2


def findpairs(xin1, yin1, xin2, yin2, maxdist=150, lon0=None, lat0=None, latlon=False):
    """
    Find pairs between two sets of points separated a distance less
    than a given value.

    Parameters
    ----------
    xin1 : ndarray
        one-dimensional array with x-coordinates in meters or
        longitudes of the first set of points
    yin1 : ndarray
        one-dimensional array with y-coordinate in meters or
        latitudes of the first set of points
    xin2 : ndarray
        one-dimensional array with x-coordinate in meters or
        longitudes of the second set of points
    yin2 : ndarray
        one-dimensional array with y-coordinate in meters or
        latitudes of the second set of points
    maxdist : scalar
        Màximum distance between points in meters
    lon0 : scalar, default=None
        Central latitude used in the mercator projection. Ignored if
        latlon=False
    lat0 : scalar, default=None
         Central latitude used in the mercator projection. Ignored if
         latlon=False
    latlon : optional, defalut=False
        Set this keyword to True if the input data are latitudes and
        longitudes
    Return
    ------
    i1 : ndarray
        Index of points in the first set of positions
    i2 : ndarray
        Index of points in the second set of positions
    """
    # Retrieve the coordinates of valid points

    index1 = np.logical_not(xin1.mask)
    index2 = np.logical_not(xin2.mask)

    # Mercator's projection, if needed

    if latlon:

        # Find the central point for the projection

        if lon0 is None:
            lon0 = (np.mean(xin1[index1]) + np.mean(xin2[index2])) / 2

        if lat0 is None:
            lat0 = (np.mean(yin1[index1]) + np.mean(yin2[index2])) / 2

        # Project the data and extract valid points

        x1, y1 = lmercator(xin1[index1], yin1[index1], lon0=lon0, lat0=lat0)
        x2, y2 = lmercator(xin2[index2], yin2[index2], lon0=lon0, lat0=lat0)

    else:

        # Extract the valid points

        x1 = xin1[index1]
        y1 = yin1[index1]
        x2 = xin2[index2]
        y2 = yin2[index2]

    # Find the pairs with a distance smaller than maxdist

    d = np.sqrt((x1[:, np.newaxis] - x2[np.newaxis, :]) ** 2 +
                (y1[:, np.newaxis] - y2[np.newaxis, :]) ** 2)

    k1, k2 = np.where(d <= maxdist)

    # Retrieve the original indices

    k1 = np.arange(xin1.shape[0])[index1][k1]
    k2 = np.arange(xin2.shape[0])[index2][k2]

    return k1, k2


def findpairs2d(xin1, yin1, xin2, yin2, maxdist=150, lon0=None, lat0=None, latlon=False):
    """
    Find pairs between two sets of points separated a distance less
    than a given value.

    Parameters
    ----------
    xin1 : ndarray
        two-dimensional array with x-coordinates in meters or
        longitudes of the first set of points
    yin1 : ndarray
        two-dimensional array with y-coordinate in meters or
        latitudes of the first set of points
    xin2 : ndarray
        two-dimensional array with x-coordinate in meters or
        longitudes of the second set of points
    yin2 : ndarray
        two-dimensional array with y-coordinate in meters or
        latitudes of the second set of points
    maxdist : scalar
        Màximum distance between points in meters
    lon0 : scalar, default=None
        Central latitude used in the mercator projection. Ignored if
        latlon=False
    lat0 : scalar, default=None
         Central latitude used in the mercator projection. Ignored if
         latlon=False
    latlon : optional, defalut=False
        Set this keyword to True if the input data are latitudes and
        longitudes
    Return
    ------
    j1 : ndarray
        Index of columns in the first set of positions
    i1 : ndarray
        Index of rows in the first set of positions
    j2 : ndarray
        Index of columns in the second set of positions
    i2 : ndarray
        Index of rows in the second set of positions
    """
    # Prepare the data

    x1 = xin1.flatten()
    y1 = yin1.flatten()
    x2 = xin2.flatten()
    y2 = yin2.flatten()

    # Find pairs

    k1, k2 = findpairs(x1, y1, x2, y2,
                       maxdist=maxdist,
                       lon0=lon0, lat0=lat0, latlon=latlon)

    # Retrieve the origina indices

    j1, i1 = np.unravel_index(k1, xin1.shape)
    j2, i2 = np.unravel_index(k2, xin2.shape)

    return j1, i1, j2, i2


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
