from pathlib import Path
from typing import Hashable

import numpy as np
import xarray as xr
import simplekml
from pyproj import Geod


# AI Usage: the first version of the following function has been written
# using OpenAI GPT 5.6 and later modified


def velocity_to_kml(
    ds: xr.Dataset,
    output_file: str | Path,
    *,
    u_name: Hashable,
    v_name: Hashable,
    lon_name: Hashable = "longitude",
    lat_name: Hashable = "latitude",
    time_name: Hashable | None = "time",
    depth_name: Hashable | None = "depth",
    time_index: int = 0,
    depth_index: int = 0,
    stride: int | tuple[int, int] = 1,
    scale: float = 1000.0,
    min_speed: float = 0.0,
    max_speed: float | None = None,
    line_width: float = 2.0,
    arrowhead_fraction: float = 0.25,
    arrowhead_angle: float = 25.0,
    color_by_speed: bool = True,
    fixed_color: str = simplekml.Color.red,
    document_name: str = "Surface velocity field",
) -> Path:
    """
    Export a horizontal velocity field from a xarray Dataset to KML.

    The selected velocity field is represented using line arrows. The arrow
    direction follows the horizontal velocity vector, and its length is
    proportional to the current speed.

    Parameters
    ----------
    ds
        Input xarray Dataset.

    output_file
        Path of the output KML file.

    u_name, v_name
        Names of the zonal and meridional velocity variables.

    lon_name, lat_name
        Names of the longitude and latitude coordinates or variables.

    time_name, depth_name
        Names of the time and depth dimensions. Set either one to None if the
        corresponding dimension is absent or should not be selected.

    time_index, depth_index
        Positional indices used to select the time and depth.

    stride
        Spatial subsampling interval. An integer applies the same interval
        along both horizontal dimensions. A tuple specifies
        ``(latitude_stride, longitude_stride)``.

    scale
        Arrow-length scale in seconds. If velocity is expressed in m/s, the
        arrow length in metres is calculated as ``speed * scale``.

        This scaling controls only the visualization. For example, with
        ``scale=1000``, a velocity of 0.5 m/s is represented by a 500 m arrow.

    min_speed, max_speed
        Only vectors within this speed interval are exported. If max_speed is
        None, no upper threshold is applied.

    line_width
        Width of arrow lines in the KML visualization.

    arrowhead_fraction
        Arrowhead length as a fraction of the main arrow length.

    arrowhead_angle
        Angle, in degrees, between the arrow shaft and each arrowhead segment.

    color_by_speed
        If True, arrows are colored according to their speed. If False,
        fixed_color is used.

    fixed_color
        KML color used when color_by_speed is False. For example,
        ``simplekml.Color.red``.

    document_name
        Name assigned to the KML document.

    Returns
    -------
    pathlib.Path
        Path of the generated KML file.

    Raises
    ------
    TypeError
        If ds is not a xarray Dataset.

    KeyError
        If a required variable or coordinate is missing.

    ValueError
        If the selected velocity arrays cannot be represented on the
        longitude-latitude grid.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(
            f"'ds' must be an xarray.Dataset, not {type(ds).__name__}."
        )

    required_names = (u_name, v_name, lon_name, lat_name)
    missing = [name for name in required_names if name not in ds]

    if missing:
        raise KeyError(
            "The following variables or coordinates are missing from the "
            f"Dataset: {missing}"
        )

    if scale <= 0:
        raise ValueError("'scale' must be strictly positive.")

    if line_width <= 0:
        raise ValueError("'line_width' must be strictly positive.")

    if not 0 < arrowhead_fraction < 1:
        raise ValueError("'arrowhead_fraction' must lie between 0 and 1.")

    if not 0 < arrowhead_angle < 90:
        raise ValueError("'arrowhead_angle' must lie between 0 and 90 degrees.")

    if isinstance(stride, int):
        if stride <= 0:
            raise ValueError("'stride' must be strictly positive.")
        lat_stride = lon_stride = stride
    else:
        if len(stride) != 2:
            raise ValueError(
                "'stride' must be an integer or a two-element tuple."
            )

        lat_stride, lon_stride = stride

        if lat_stride <= 0 or lon_stride <= 0:
            raise ValueError("Both stride values must be strictly positive.")

    indexers = {}

    if (
        time_name is not None
        and time_name in ds[u_name].dims
        and time_name in ds[v_name].dims
    ):
        indexers[time_name] = time_index

    if (
        depth_name is not None
        and depth_name in ds[u_name].dims
        and depth_name in ds[v_name].dims
    ):
        indexers[depth_name] = depth_index

    u = ds[u_name].isel(indexers, drop=True)
    v = ds[v_name].isel(indexers, drop=True)

    # Align both velocity components before processing them.
    u, v = xr.align(u, v, join="exact")

    lon = ds[lon_name]
    lat = ds[lat_name]

    # Select time and depth in the coordinate arrays if they depend on them.
    coordinate_indexers = {
        dim: index
        for dim, index in indexers.items()
        if dim in lon.dims or dim in lat.dims
    }

    lon = lon.isel(
        {dim: index for dim, index in coordinate_indexers.items()
         if dim in lon.dims},
        drop=True,
    )
    lat = lat.isel(
        {dim: index for dim, index in coordinate_indexers.items()
         if dim in lat.dims},
        drop=True,
    )

    # Convert one-dimensional coordinates into a two-dimensional grid.
    if lon.ndim == 1 and lat.ndim == 1:
        lon_grid, lat_grid = xr.broadcast(lon, lat)

    # Curvilinear grids normally provide two-dimensional longitude and
    # latitude arrays with matching dimensions.
    elif lon.ndim == 2 and lat.ndim == 2:
        lon_grid, lat_grid = xr.align(lon, lat, join="exact")

    else:
        raise ValueError(
            "Longitude and latitude must both be one-dimensional or both "
            "be two-dimensional."
        )

    # Broadcast coordinates and velocities onto a common horizontal grid.
    try:
        u_grid, v_grid, lon_grid, lat_grid = xr.broadcast(
            u,
            v,
            lon_grid,
            lat_grid,
        )
    except ValueError as exc:
        raise ValueError(
            "The velocity variables cannot be broadcast onto the "
            "longitude-latitude grid. Check their dimensions."
        ) from exc

    # Remove singleton dimensions, but retain the two horizontal dimensions.
    u_grid = u_grid.squeeze(drop=True)
    v_grid = v_grid.squeeze(drop=True)
    lon_grid = lon_grid.squeeze(drop=True)
    lat_grid = lat_grid.squeeze(drop=True)

    if u_grid.ndim != 2:
        raise ValueError(
            "After selecting time and depth, the velocity variables must have "
            f"exactly two dimensions. Found dimensions: {u_grid.dims}"
        )

    # Ensure all arrays follow the same dimension ordering.
    horizontal_dims = u_grid.dims

    try:
        v_grid = v_grid.transpose(*horizontal_dims)
        lon_grid = lon_grid.transpose(*horizontal_dims)
        lat_grid = lat_grid.transpose(*horizontal_dims)
    except ValueError as exc:
        raise ValueError(
            "The velocity and coordinate arrays do not use compatible "
            "horizontal dimensions."
        ) from exc

    u_values = np.asarray(u_grid.values, dtype=float)
    v_values = np.asarray(v_grid.values, dtype=float)
    lon_values = np.asarray(lon_grid.values, dtype=float)
    lat_values = np.asarray(lat_grid.values, dtype=float)

    # Spatial subsampling.
    spatial_slice = (
        slice(None, None, lat_stride),
        slice(None, None, lon_stride),
    )

    u_values = u_values[spatial_slice]
    v_values = v_values[spatial_slice]
    lon_values = lon_values[spatial_slice]
    lat_values = lat_values[spatial_slice]

    speed = np.hypot(u_values, v_values)

    valid = (
        np.isfinite(u_values)
        & np.isfinite(v_values)
        & np.isfinite(lon_values)
        & np.isfinite(lat_values)
        & np.isfinite(speed)
        & (speed >= min_speed)
    )

    if max_speed is not None:
        valid &= speed <= max_speed

    if not np.any(valid):
        raise ValueError(
            "No valid velocity vectors remain after applying the selections "
            "and speed thresholds."
        )

    # Direction measured clockwise from geographic north.
    bearing = np.degrees(np.arctan2(u_values, v_values)) % 360.0

    # Arrow length in metres when u and v are expressed in m/s.
    arrow_length = speed * scale

    valid_speeds = speed[valid]
    speed_min = float(np.nanmin(valid_speeds))
    speed_max = float(np.nanmax(valid_speeds))

    kml = simplekml.Kml(name=document_name)
    folder = kml.newfolder(name=document_name)
    geod = Geod(ellps="WGS84")

    def speed_to_color(value: float) -> str:
        """
        Map speed to a blue-cyan-yellow-red color scale.

        simplekml expects colors in KML ABGR notation, so the conversion is
        performed with simplekml.Color.rgb().
        """
        if np.isclose(speed_max, speed_min):
            normalized = 0.5
        else:
            normalized = (value - speed_min) / (speed_max - speed_min)

        normalized = float(np.clip(normalized, 0.0, 1.0))

        if normalized < 1 / 3:
            local = normalized * 3
            red = 0
            green = round(255 * local)
            blue = 255

        elif normalized < 2 / 3:
            local = (normalized - 1 / 3) * 3
            red = round(255 * local)
            green = 255
            blue = round(255 * (1 - local))

        else:
            local = (normalized - 2 / 3) * 3
            red = 255
            green = round(255 * (1 - local))
            blue = 0

        return simplekml.Color.rgb(red, green, blue)

    for index in zip(*np.where(valid)):
        lon_start = float(lon_values[index])
        lat_start = float(lat_values[index])
        vector_speed = float(speed[index])
        vector_bearing = float(bearing[index])
        vector_length = float(arrow_length[index])

        # End point of the arrow shaft.
        lon_end, lat_end, _ = geod.fwd(
            lon_start,
            lat_start,
            vector_bearing,
            vector_length,
        )

        head_length = vector_length * arrowhead_fraction

        # Arrowhead segments point backwards from the shaft endpoint.
        left_bearing = (
            vector_bearing + 180.0 - arrowhead_angle
        ) % 360.0

        right_bearing = (
            vector_bearing + 180.0 + arrowhead_angle
        ) % 360.0

        lon_left, lat_left, _ = geod.fwd(
            lon_end,
            lat_end,
            left_bearing,
            head_length,
        )

        lon_right, lat_right, _ = geod.fwd(
            lon_end,
            lat_end,
            right_bearing,
            head_length,
        )

        color = (
            speed_to_color(vector_speed)
            if color_by_speed
            else fixed_color
        )

        description = (
            f"u: {u_values}<br>"
            f"v: {v_values}<br>"
            f"speed: {vector_speed:.6g}<br>"
            f"bearing: {vector_bearing:.2f} degrees"
        )

        # Arrow shaft.
        shaft = folder.newlinestring(
            name=f"Velocity: {vector_speed:.3g}",
            description=description,
            coords=[
                (lon_start, lat_start, 0.0),
                (lon_end, lat_end, 0.0),
            ],
        )
        shaft.altitudemode = simplekml.AltitudeMode.clamptoground
        shaft.style.linestyle.color = color
        shaft.style.linestyle.width = line_width

        # Left side of the arrowhead.
        left_head = folder.newlinestring(
            coords=[
                (lon_end, lat_end, 0.0),
                (lon_left, lat_left, 0.0),
            ]
        )
        left_head.altitudemode = simplekml.AltitudeMode.clamptoground
        left_head.style.linestyle.color = color
        left_head.style.linestyle.width = line_width

        # Right side of the arrowhead.
        right_head = folder.newlinestring(
            coords=[
                (lon_end, lat_end, 0.0),
                (lon_right, lat_right, 0.0),
            ]
        )
        right_head.altitudemode = simplekml.AltitudeMode.clamptoground
        right_head.style.linestyle.color = color
        right_head.style.linestyle.width = line_width

    output_path = Path(output_file).expanduser()

    if output_path.suffix.lower() != ".kml":
        output_path = output_path.with_suffix(".kml")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    kml.save(str(output_path))

    return output_path
