# System modules

import os

# Import scientific modules

import numpy as np
import scipy as sp
import xarray as xr


def lmercator(xin_, yin_, lon0=0, lat0=0, radius=6371000.0, inverse=False):
    """
    Compute the Mercator projection centered at the center of the image

    Parameters
    ----------
    xin : ndarray
        Longitude (inverse=False) or x-coordinate (inverse=True)
    yin : ndarray
        Latitude (inverse=False) or y-coordinate (inverse=True)
    lon0 : float, optional, default=0
        Central longitude used to compute the projection (inverse=True)
    lat0 : float, optional, default=0
        Central latitude used to compute the projection (inverse=True)
    radius : float, optional, default=6371000.0
        Earth mean radius [m]
    inverse : boolean, optional, default=False
        Set to True for the inverse Mercator transform

    Returns
    -------
    xout : ndarray
        x-coordinate (inverse=False) or Longitude (inverse=True)
    yout : ndarray
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
    Find the closest point of an array to a reference array

    Parameters
    ----------
    xref : ndarray
        Reference array
    x : ndarray
        Array to be synchronized with xref
    dxmax : float, optional, default=None
        Maximum allowed distance between points

    Returns
    -------
    index : masked array
        Index of x for the closest points in xref. It masks out those
        points with a distance larger than dxmax
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
    """
    Counts the number of times a flag has a specific value

    Parameters
    ----------
    ds : x-array dataset
        Dataset to be inspected
    nmin : integer
        Minimum number of observations
    flagname : string, optional, default='PRIM'
        Name of the flag to be inspected
    flagvalue : integer, optional, default=1
        Expected value of the flag

    Returns
    -------
    lon : masked array
        Longitudes for which the flag has a specified value
    lat : masked array
        Latitudes for which the flag has a specified value
    flag : xarray data array
        Array with 1 in those locations wehere the flag had a
        specified value
    """
    # Flag the valid values

    flag = xr.zeros_like(ds[flagname], dtype=int)
    flag.data[ds[flagname].data == flagvalue] = 1

    # Count the number of points and flag invalid data

    mask = flag.sum(dim='time').astype(int).data < nmin

    # Mask out lat/lon points with not enough valid data

    lon = np.ma.array(data=ds.lon.data, mask=mask)
    lat = np.ma.array(data=ds.lat.data, mask=mask)

    return lon, lat, flag


def hfr_rmse_model(x, stdu, stde):
    """
    Dependence of the Root-Mean-Square-Error of the radial velocity
    with the observing angle difference between two HF radar stations

    Parameters
    ----------
    x : ndarray
        Angle difference between stations
    stdu : float
        Standard deviation of the ocean currents
    stde : float
        Standrad deviation of noise of the HF radar stations

    Returns
    -------
    rmse : ndarray
        Root-Mean-Square-Error variation with the angle difference

    Notes
    -----
    Implement the theoretical model for the Root-Mean-Square-Error
    between the measurements of two different radial stations [1]_

    .. math::
        \sigma_{rms} = \left[4 \sigma_u^2 \cos^2\left(\frac{\theta_{R} - \theta_{R'}}{2}\right) + 2\sigma_{\epsilon R}^2

    where :math:`\theta_{R}` and :math:`\theta_{R'}` denote the
    azimuths (measured clockwise from North) of each radial;
    :math:`\sigma_{\epsilon R}^2` is the noise variance of the radial
    velocity; and :math:`\sigma_u^2` is the variance of the current.

    References
    ----------
    .. [1] Kim, S. Y.; Terrill, E. J.; and B. D. Cornuelle (2008).
       Mapping surface currents from HF radar radial velocity
       measurements using optimal interpolation. Journal of
       Geophysical Research: Oceans, 113, C10023.
       http://doi.org/10.1029/2007JC004244.
    """
    return np.sqrt(4 * stdu ** 2 * np.cos(x * np.pi / 360) ** 2 + 2 * stde ** 2)


def hfr_rmse_fit(angle, rmse, stdu=0.15, stde=0.08):
    """
    Fit a set of observed Root-Mean-Square-Error of the radial
    velocities between two diferen radar stations to a theoretical
    model

    Parameters
    ----------
    angle : ndarray
        Angle difference between measurements of the two antennas
        measured clockwise from North
    rmse : ndarray
        Observed Root-Mean-Square-Error
    stdu : float, optional, default=0.15
        First guess for the standard deviation of ocean currents
    stde : float, optional, default=0.08
        First guess for the standard deviation of noise

    Returns
    -------
    stdu : float
        Resulting standard deviation of ocean currents
    stde : float
        Resulting standard deviation of noise
    cov : float
        Resulting covariance

    See Also
    --------
    hfr_rmse_model : Model used for the fitting.

    """
    curve_fit = sp.optimize.curve_fit
    param, cov = curve_fit(hfr_rmse_model, angle, rmse, p0=[stdu, stde])
    return param[0], param[1], cov


def hfr_rmse_pairs(ds1, ds2, maxdist=150, nmin=10, rmin=0, rmax=500,
                   flagname='PRIM', flagvalue=1):
    """
    Compute the Root Mean Square Error between two time-series of HF
    radar measurements of radial velocities

    Parameters
    ----------
    ds1 : x-array dataset
        Dataset with the
    nmin : integer, optional, default=10
        Minimum number of observations
    flagname : string, optional, default='PRIM'
        Name of the flag to be inspected
    flagvalue : integer, optional, default=1
        Expected value of the flag
    maxdist: float, optional, default=150
        Maximum distance between points to be keept as a pair [m]
    rmin : float, optional, default=0
        Minimum distance from the antenna [km]
    rmax : float, optional, default=500
        Maximum distance fron the antenna [km]
    """
    # Parameters

    bname = 'bearing'                # Name for the bearing coordinate
    rname = 'range'                  # Name for the range coordinate
    velname = 'VELO'                 # Name for the radial velocity
    dtmax = np.timedelta64(55, 'm')  # Maximum seoaration of time

    # Synchronize time series. Get the indices that match the same
    # times in both time series

    it2 = synchronize(ds1.time.data, ds2.time.data, dxmax=dtmax)
    it1 = np.arange(ds1.time.data.shape[0])[np.logical_not(it2.mask)]
    it2 = it2.data[np.logical_not(it2.mask)]

    # Extract information from the DtaSets in the needed format

    lon1, lat1, flag1 = getinfo(ds1, nmin=nmin,
                                flagname=flagname, flagvalue=flagvalue)
    lon2, lat2, flag2 = getinfo(ds2, nmin=nmin,
                                flagname=flagname, flagvalue=flagvalue)

    # Find pairs of data. The first index corresponds to the range, the
    # second to the bearing

    ir1, ib1, ir2, ib2 = findpairs2d(lon1, lat1, lon2, lat2,
                                     maxdist=maxdist, latlon=True)

    # Keep only pairs within certain ranges

    range1 = ds1[rname][ir1].data
    range2 = ds2[rname][ir2].data
    index = np.logical_and(np.logical_and(range1 >= rmin,
                                          range1 <= rmax),
                           np.logical_and(range2 >= rmin,
                                          range2 <= rmax))
    ir1 = ir1[index]
    ib1 = ib1[index]
    ir2 = ir2[index]
    ib2 = ib2[index]

    # Compute the angle between bearings

    angle = ds1[bname][ib1].data - ds2[bname][ib2].data

    # Extract the data for the selected pairs. Use the synchronization
    # indices computed before. It does not exploit the flexibility of
    # xarray, but it works.

    u1 = ds1[velname].data[:, ir1, ib1][it1, :]
    u2 = ds2[velname].data[:, ir2, ib2][it2, :]
    flag1 = flag1.data[:, ir1, ib1][it1, :]
    flag2 = flag2.data[:, ir2, ib2][it2, :]

    # Compute the RMSE of velocities

    npairs = ir1.shape[0]
    rmse = np.ma.array(data=np.zeros(npairs),
                       mask=np.ones(npairs, dtype=bool))
    for k in np.arange(npairs):
        mask = np.logical_and(flag1[:, k] == 1, flag2[:, k] == 1)
        if np.sum(mask) > 2:
            rmse.mask[k] = False
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


def hfr_noise(datain, maxit=10, ksigma=3, return_coeff=False, nobs=None):
    """
    Estimate the noise level of High-Frequency radar measurements of
    radial velocities

    Parameters
    ----------
    datain : masked array
        Radial velocities [cm/s]. The dimensionality of velocities can
        be 1, in which case it assumed that velocities are along a
        radial; 2, in which case the first dimension is taken as the
        radial direction and the second as the bearing; or 3, in which
        case the diemsnions are taken as time, radial and bearing.
    maxit : integer, optional, default=10
        Maximum number of iterations used to compute the standard
        deviation of noise
    ksigma : integer, optional, default=3
        parameter used to distinguish values dominated by noise from
        those dominated by the signal
    return_coeff : bool, optional, default=False
        Set this coefficient to return the wavelet coefficients labeled
        as noise.
    nobs : ndarray, optional, default=None
        Set this parameter to the number of values used to average
        input data. It must have a diensionality of 3, independently of
        the dimensionality of the input data

    Returns
    -------
    noise : ndarray
        Noise standard deviation at each time step

    w1 : ndarray
        Wavelet coefficients labeled as noise

    Notes
    -----
    The method used to compute the noise standard deviation is described in [1]_

    References
    ----------
    .. [1] Isern-Fontanet, J; Quirós-Collazos, L.; Iglesias, J.;
       Martínez, J; Ballabrera-Poy, J.; Agostinho, P.; González-Haro,
       C.; and García-Ladona, E. (2026). Data-Driven Noise Estimation
       for Individual High-Frequency Radar Stations**. Submitted to
       J. Atmos. Oceanic Technol.
    """

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
