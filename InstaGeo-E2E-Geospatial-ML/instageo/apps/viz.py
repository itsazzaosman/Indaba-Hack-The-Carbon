# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""Utils for Raster Visualisation."""

import base64
import io

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import rasterio
import xarray as xr
from matplotlib.colors import Normalize
from pyproj import CRS, Transformer

epsg3857_to_epsg4326 = Transformer.from_crs(3857, 4326, always_xy=True)


def get_crs(filepath: str) -> CRS:
    """Retrieves the CRS of a GeoTiff data.

    Args:
        filepath: Path to a GeoTiff file.

    Returns:
        CRS of data stored in `filepath`
    """
    src = rasterio.open(filepath)
    return src.crs


def create_colorbar(
    vmin: float,
    vmax: float,
    cmap_name: str = "viridis",
    title: str = "Prediction Values",
    figsize: tuple = (8, 1),
) -> str:
    """Create a colorbar as a base64 encoded image.

    Args:
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        cmap_name: Name of the matplotlib colormap
        title: Title for the colorbar
        figsize: Figure size for the colorbar

    Returns:
        Base64 encoded image string of the colorbar
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, orientation="horizontal")
    cbar.set_label(title, fontsize=12)

    # Remove the main axes
    ax.remove()

    # Save to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    buffer.seek(0)

    # Convert to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)

    return f"data:image/png;base64,{img_base64}"


def add_raster_to_plotly_figure(
    xarr_dataset: xr.Dataset,
    from_crs: CRS,
    column_name: str = "band_data",
    scale: float = 1.0,
    cmap_name: str = "viridis",
    alpha: int = 180,
    value_range: tuple | None = None,
    clip_values: bool = False,
    clip_range: tuple = (0, 1),
) -> tuple:
    """Add a raster plot on a Plotly graph object figure.

    This function overlays raster data from an xarray dataset onto a Plotly map figure.
    The data is reprojected to EPSG:3857 CRS for compatibility with Mapbox's projection
    system.

    Args:
        xarr_dataset (xr.Dataset): xarray dataset containing the raster data.
        from_crs (CRS): Coordinate Reference System of data stored in xarr_dataset.
        column_name (str): Name of the column in `xarr_dataset` to be plotted. Defaults
            to "band_data".
        scale (float): Scale factor for adjusting the plot resolution. Defaults to 1.0.
        cmap_name (str): Name of the matplotlib colormap to use. Defaults to "viridis".
        alpha (int): Alpha transparency value (0-255). Defaults to 180.
        value_range (tuple): Min and max values for normalization. If None, uses data range.
        clip_values (bool): Whether to clip values to a specific range. Defaults to False.
        clip_range (tuple): Range to clip values to if clip_values is True. Defaults to (0, 1).

    Returns:
        tuple: (PIL Image, coordinates, vmin, vmax) for the raster overlay and colorbar info
    """
    # Reproject to EPSG:3857 CRS
    xarr_dataset = xarr_dataset.rio.write_crs(from_crs).rio.reproject("EPSG:3857")

    # Handle value clipping if requested
    if clip_values:
        xarr_dataset = xarr_dataset.where(
            (xarr_dataset >= clip_range[0]) & (xarr_dataset <= clip_range[1]), np.nan
        )

    # Get data values for colorbar range
    numpy_data = xarr_dataset[column_name].squeeze().to_numpy()

    # Determine value range for normalization
    if value_range is not None:
        vmin, vmax = value_range
    else:
        # Use actual data range, ignoring NaN values
        valid_data = numpy_data[~np.isnan(numpy_data)]
        if len(valid_data) > 0:
            vmin, vmax = np.min(valid_data), np.max(valid_data)
        else:
            vmin, vmax = 0, 1

    # Get Raster dimension
    plot_height, plot_width = numpy_data.shape

    # Data aggregation
    canvas = ds.Canvas(
        plot_width=int(plot_width * scale), plot_height=int(plot_height * scale)
    )
    agg = canvas.raster(xarr_dataset[column_name].squeeze(), interpolate="linear")

    coords_lat_min, coords_lat_max = (
        agg.coords["y"].values.min(),
        agg.coords["y"].values.max(),
    )
    coords_lon_min, coords_lon_max = (
        agg.coords["x"].values.min(),
        agg.coords["x"].values.max(),
    )

    # Transform coordinates from EPSG:3857 to EPSG:4326
    (
        coords_lon_min,
        coords_lon_max,
    ), (
        coords_lat_min,
        coords_lat_max,
    ) = epsg3857_to_epsg4326.transform(
        [coords_lon_min, coords_lon_max], [coords_lat_min, coords_lat_max]
    )
    # Corners of the image, which need to be passed to mapbox
    coordinates = [
        [coords_lon_min, coords_lat_max],
        [coords_lon_max, coords_lat_max],
        [coords_lon_max, coords_lat_min],
        [coords_lon_min, coords_lat_min],
    ]

    # Apply color map with specified range
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except KeyError:
        print(f"Colormap '{cmap_name}' not found, using 'viridis' instead")
        cmap = matplotlib.colormaps["viridis"]

    img = tf.shade(agg, cmap=cmap, alpha=alpha, how="linear", span=(vmin, vmax))[
        ::-1
    ].to_pil()

    # Convert PIL image to base64 for better persistence
    buffer = io.BytesIO()
    try:
        # Save with high quality PNG settings for better persistence
        img.save(buffer, format="PNG", optimize=True, compress_level=6)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        img_data_uri = f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"Warning: Image encoding failed: {e}")
        # Fallback to direct PIL image
        img_data_uri = img

    return img_data_uri, coordinates, vmin, vmax


def read_geotiff_to_xarray(filepath: str) -> tuple[xr.Dataset, CRS]:
    """Read a GeoTIFF file into an xarray Dataset.

    Args:
        filepath (str): Path to the GeoTIFF file.

    Returns:
        tuple: (xr.Dataset, CRS) The loaded xarray dataset and its CRS.
    """
    return xr.open_dataset(filepath).sel(band=1), get_crs(filepath)


def create_map_with_geotiff_tiles(
    tiles_to_overlay: list[str],
    cmap_name: str = "viridis",
    alpha: int = 180,
    show_colorbar: bool = True,
    value_range: tuple | None = None,
    clip_values: bool = False,
    clip_range: tuple = (0, 1),
) -> tuple:
    """Create a map with multiple GeoTIFF tiles overlaid.

    This function reads GeoTIFF files from a specified directory and overlays them on a
    Plotly map with heatmap visualization.

    Args:
        tiles_to_overlay (list[str]): Path to tiles to overlay on map.
        cmap_name (str): Name of the matplotlib colormap. Defaults to "viridis".
        alpha (int): Alpha transparency value (0-255). Defaults to 180.
        show_colorbar (bool): Whether to generate colorbar info. Defaults to True.
        value_range (tuple): Min and max values for normalization. If None, uses data range.
        clip_values (bool): Whether to clip values to a specific range. Defaults to False.
        clip_range (tuple): Range to clip values to if clip_values is True. Defaults to (0, 1).

    Returns:
        tuple: (Plotly Figure, colorbar_info) where colorbar_info contains colorbar details
    """
    fig = go.Figure(go.Scattermapbox())

    mapbox_layers = []
    all_vmin, all_vmax = [], []
    all_coordinates = []  # Track all coordinates for centering

    for i, tile in enumerate(tiles_to_overlay):
        if tile.endswith(".tif") or tile.endswith(".tiff"):
            xarr_dataset, crs = read_geotiff_to_xarray(tile)
            img_data_uri, coordinates, vmin, vmax = add_raster_to_plotly_figure(
                xarr_dataset,
                crs,
                "band_data",
                scale=1.0,
                cmap_name=cmap_name,
                alpha=alpha,
                value_range=value_range,
                clip_values=clip_values,
                clip_range=clip_range,
            )

            # Create layer with unique ID and persistence settings
            layer_config = {
                "sourcetype": "image",
                "source": img_data_uri,
                "coordinates": coordinates,
                "below": "traces",  # Ensure overlay is visible
                "opacity": alpha / 255.0,  # Set opacity explicitly
                "visible": True,  # Ensure layer is visible
                "name": f"overlay_{i}",  # Unique layer name
            }

            mapbox_layers.append(layer_config)
            all_vmin.append(vmin)
            all_vmax.append(vmax)
            all_coordinates.extend(coordinates)

    # Calculate center and zoom based on overlays
    if all_coordinates:
        # Extract all lons and lats
        all_lons = [coord[0] for coord in all_coordinates]
        all_lats = [coord[1] for coord in all_coordinates]

        # Calculate center
        center_lon = (min(all_lons) + max(all_lons)) / 2
        center_lat = (min(all_lats) + max(all_lats)) / 2

        # Calculate zoom level based on coordinate spread
        lon_range = max(all_lons) - min(all_lons)
        lat_range = max(all_lats) - min(all_lats)
        max_range = max(lon_range, lat_range)

        # Auto-zoom calculation (rough approximation)
        if max_range > 50:
            zoom_level = 2
        elif max_range > 20:
            zoom_level = 4
        elif max_range > 10:
            zoom_level = 5
        elif max_range > 5:
            zoom_level = 6
        elif max_range > 2:
            zoom_level = 7
        elif max_range > 1:
            zoom_level = 8
        elif max_range > 0.5:
            zoom_level = 9
        else:
            zoom_level = 10

        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=go.layout.mapbox.Center(lat=center_lat, lon=center_lon),
                zoom=zoom_level,
                bearing=0,
                pitch=0,
            ),
        )
    else:
        # Default view if no overlays
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(
                center=go.layout.mapbox.Center(lat=0, lon=20),
                zoom=2.0,
                bearing=0,
                pitch=0,
            ),
        )

    # Set figure layout with better persistence settings
    fig.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        showlegend=False,
        # Add configuration to improve persistence
        uirevision="constant",  # Prevent UI from resetting
        mapbox_layers=mapbox_layers,
    )

    # Create colorbar info if requested
    colorbar_info = None
    if show_colorbar and all_vmin and all_vmax:
        global_vmin = min(all_vmin)
        global_vmax = max(all_vmax)
        colorbar_base64 = create_colorbar(
            global_vmin, global_vmax, cmap_name, "Prediction Values"
        )
        colorbar_info = {
            "image": colorbar_base64,
            "vmin": global_vmin,
            "vmax": global_vmax,
            "cmap": cmap_name,
            "center_lat": center_lat if all_coordinates else 0,
            "center_lon": center_lon if all_coordinates else 20,
            "zoom": zoom_level if all_coordinates else 2.0,
            "num_tiles": len(mapbox_layers),
        }

    return fig, colorbar_info
