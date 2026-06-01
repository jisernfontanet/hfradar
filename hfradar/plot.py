# imports

import os                              # Operating system-related module
import glob                            # Linux-like path manipulation module

import numpy as np                     # Numerical/array module
import matplotlib.pyplot as plt        # Plotting module
import matplotlib.colors as pltcolor   # Import colors for Matplotlib
from matplotlib import cm

import cartopy.crs as ccrs             # Projection/mapping module and objects
from cartopy.feature import NaturalEarthFeature, COLORS
from shapely.geometry.polygon import Polygon


# Load Earth features from Cartopy

land = NaturalEarthFeature(category='physical',
                           name='land',
                           scale='10m',
                           facecolor=COLORS['land'],
                           edgecolor='k')


#  Get the RGB colors from the values


def data2rgb(values, vmin=None, vmax=None, cmap='jet'):

    # Get the minimum and maximum values, if not provided

    if not vmin:
        vmin = np.min(values)

    if not vmax:
        vmax = np.max(values)

    # Scalar Mappable

    sm = cm.ScalarMappable(norm=pltcolor.Normalize(vmin=vmin, vmax=vmax),
                           cmap=cmap)

    # Convert to  RGB

    rgb = sm.to_rgba(values)

    return rgb, sm


# Get the RGB from name


def color2rgb(color_name):
    return pltcolor.to_rgb(color_name)


# Mercator projection


def lmercator(xin_, yin_, lon0=0, lat0=0, radius=6371000.0, inverse=False):
    """
    Compute the Mercator projection centered at the center of the image

    Input
    -----
    xin:        Longitude (inverse=False) or x-coordinate (inverse=True)
    yin:        Latitude (inverse=False) or y-coordinate (inverse=True)

    Keywords
    --------
    lon0:       Central longitude used to compute the projection (inverse=True)
    lat0:       Central latitude used to compute the projection (inverse=True)

    Output
    -----
    xout:        x-coordinate (inverse=False) or Longitude (inverse=True)
    yout:        y-coordinate (inverse=False) or Latitude (inverse=True)
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

# Convenience functions used in other functions of this module


def lonlat2xy(lon, lat, lon0=0, lat0=0):
    return lmercator(lon, lat, lon0=lon0, lat0=lat0, inverse=False)


def xy2lonlat(x, y, lon0=0, lat0=0):
    return lmercator(x, y, lon0=lon0, lat0=lat0, inverse=True)


# Function to plot data


def plot_radial(lonin,
                latin,
                valin,
                drin,
                dangin,
                vmin=None,
                vmax=None,
                lon0=None,
                lat0=None,
                lonlat2xy_function=lonlat2xy,
                xy2lonlat_function=xy2lonlat,
                ax=None,
                label=None,
                show_colorbar=False,
                flags=None,
                cmap='jet'):

    """
    Parameters
    ---------
    lonin
    latin:
    valin:
    drin:
    dangin:
    vmin:
    vmax:
    lon0:
    lat0:
    lonlat2xy_function:
    xy2lonlat_function:
    ax:
    label:
    show_colorbar:
    flags: dictionary, optional, default=None
        Dictionary with the name of the key and the data the Hexadecimal
        color assigned to this value
    cmap:

    Return
    .-----

    """
    # Select valid data

    ind = np.isfinite(valin)
    lon = lonin[ind]
    lat = latin[ind]
    val = valin[ind]

    dr = drin * np.ones(lon.shape)
    dang = dangin * np.ones(lon.shape) * np.pi / 180

    nobs = len(val)

    # Estimate the center of the plot if it has not been provided

    if not lon0:
        lon0 = np.mean(lon)

    if not lat0:
        lat0 = np.mean(lat)

    # Convert data to distances

    x, y = lonlat2xy_function(lon, lat, lon0=lon0, lat0=lat0)

    # Compute polygons

    a = np.arctan2(y, x)
    amin = a - dang / 2
    amax = a + dang / 2

    r = np.sqrt(x ** 2 + y ** 2)
    rmin = r - dr / 2
    rmax = r + dr / 2

    num = 4
    xp = np.zeros((nobs, num))
    xp[:, 0] = rmin * np.cos(amin)
    xp[:, 1] = rmax * np.cos(amin)
    xp[:, 2] = rmax * np.cos(amax)
    xp[:, 3] = rmin * np.cos(amax)
    yp = np.zeros((nobs, num))
    yp[:, 0] = rmin * np.sin(amin)
    yp[:, 1] = rmax * np.sin(amin)
    yp[:, 2] = rmax * np.sin(amax)
    yp[:, 3] = rmin * np.sin(amax)

    # Convert distances to lat/lon

    lonp, latp = xy2lonlat_function(xp, yp, lon0=lon0, lat0=lat0)

    # Colors

    if not flags:

        # Set max/min

        if not vmin:
            vmin = np.min(val)

        if not vmax:
            vmax = np.max(val)

        # Get the RGB indices for values

        colors, c = data2rgb(val, vmin=vmin, vmax=vmax, cmap=cmap)

        # Turn off the legend

        show_legend = False

    else:

        # Select colors

        colors = np.zeros((nobs, 3))

        for key in flags:

            ind = np.fix(val) == key
            colors[ind, :] = color2rgb(flags[key])

        # Information about the colors, needed only for non-flag data

        c = None

        # Turn on the legend

        show_legend = True

    if ax is None:
        lonmin = np.min(lonp)
        lonmax = np.max(lonp)
        latmin = np.min(latp)
        latmax = np.max(latp)
        fig, ax = plt.subplots(lonmin=lonmin,
                               lonmax=lonmax,
                               latmin=latmin,
                               latmax=latmax,
                               subplot_kw=dict(projection=ccrs.Mercator()))
    else:
        fig = None

    # Print cell centers

    ax.plot(lon0, lat0, marker='.', linestyle=' ', color='r',
            transform=ccrs.PlateCarree())
    for i in np.arange(nobs):
        ax.fill(lonp[i, :], latp[i, :], color=colors[i, :],
                transform=ccrs.PlateCarree())

    # Legend

    if show_legend:
        for key in flags.keys():
            ax.plot([np.nan], [np.nan], label=str(key), color=flags[key],
                    marker='s', linestyle='')
        ax.legend()

    # Return information of the colors used to be abke to plot the colorbar

    return c
