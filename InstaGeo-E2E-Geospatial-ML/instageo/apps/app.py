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

"""InstaGeo Serve Module.

InstaGeo Serve is a web application that enables the visualisation of GeoTIFF files in an
interactive map with heatmap visualization for inference predictions.
"""

import glob
import os
from pathlib import Path

import streamlit as st

from instageo.apps.viz import create_map_with_geotiff_tiles


def generate_map(
    directory: str,
    cmap_name: str = "viridis",
    alpha: int = 180,
    show_colorbar: bool = True,
    value_range: tuple | None = None,
    clip_values: bool = False,
    clip_range: tuple = (0, 1),
) -> None:
    """Generate the plotly map with heatmap visualization.

    Arguments:
        directory (str): Directory containing GeoTiff files.
        cmap_name (str): Name of the matplotlib colormap.
        alpha (int): Alpha transparency value (0-255).
        show_colorbar (bool): Whether to show colorbar.
        value_range (tuple): Min and max values for normalization.
        clip_values (bool): Whether to clip values to a specific range.
        clip_range (tuple): Range to clip values to if clip_values is True.

    Returns:
        None.
    """
    try:
        if not directory or not Path(directory).is_dir():
            raise ValueError("Invalid directory path.")

        tiles_to_consider = glob.glob(os.path.join(directory, "*.tif"))
        if not tiles_to_consider:
            raise FileNotFoundError("No GeoTIFF files found for the given directory.")

        # Debug information
        st.info(f"Found {len(tiles_to_consider)} GeoTIFF files to overlay")

        fig, colorbar_info = create_map_with_geotiff_tiles(
            tiles_to_consider,
            cmap_name=cmap_name,
            alpha=alpha,
            show_colorbar=show_colorbar,
            value_range=value_range,
            clip_values=clip_values,
            clip_range=clip_range,
        )

        # Display the map
        st.plotly_chart(fig, use_container_width=True)

        # Display colorbar and map info if available
        if colorbar_info and show_colorbar:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Colorbar")
                st.markdown(
                    f'<img src="{colorbar_info["image"]}" width="400">',
                    unsafe_allow_html=True,
                )
                st.write(
                    f"**Value Range:** {colorbar_info['vmin']:.3f} to "
                    f"{colorbar_info['vmax']:.3f}"
                )
                st.write(f"**Colormap:** {colorbar_info['cmap']}")

            with col2:
                st.subheader("Map Info")
                st.write(f"**Overlays:** {colorbar_info['num_tiles']} tiles")
                st.write(
                    f"**Center:** {colorbar_info['center_lat']:.3f}°, "
                    f"{colorbar_info['center_lon']:.3f}°"
                )
                st.write(f"**Zoom Level:** {colorbar_info['zoom']}")
                st.write(f"**Alpha:** {alpha}/255")

            # Visibility tips
            if colorbar_info["vmin"] == colorbar_info["vmax"]:
                st.warning(
                    "⚠️ **All values are the same!** The overlay might not be visible."
                )
            elif abs(colorbar_info["vmax"] - colorbar_info["vmin"]) < 0.001:
                st.warning(
                    "⚠️ **Very small value range!** Try adjusting the colormap or transparency."
                )

    except (ValueError, FileNotFoundError, Exception) as e:
        st.error(f"An error occurred: {str(e)}")


def main() -> None:
    """Instageo Serve Main Entry Point."""
    st.set_page_config(layout="wide")
    st.title("InstaGeo Serve - Inference Prediction Heatmap Visualization")

    st.sidebar.subheader(
        "This application enables the visualisation of GeoTIFF files as heatmaps "
        "on an interactive map.",
        divider="rainbow",
    )
    st.sidebar.header("Settings")

    with st.sidebar.container():
        directory = st.sidebar.text_input(
            "GeoTiff Directory:",
            help="Write the path to the directory containing your GeoTIFF files",
        )

        st.sidebar.subheader("Visualization Settings")

        # Quick visibility presets
        st.sidebar.write("**Quick Settings:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔥 High Contrast", help="Settings for maximum visibility"):
                st.session_state.preset_cmap = "jet"
                st.session_state.preset_alpha = 255
                st.session_state.preset_clip = False
        with col2:
            if st.button("🌈 Rainbow", help="Colorful rainbow visualization"):
                st.session_state.preset_cmap = "rainbow"
                st.session_state.preset_alpha = 255
                st.session_state.preset_clip = False

        # Colormap selection
        colormap_options = [
            "jet",
            "rainbow",
            "coolwarm",
            "bwr",
            "seismic",  # High contrast options first
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Reds",
            "Blues",
            "Greens",
            "Oranges",
            "Purples",
            "YlOrRd",
            "YlOrBr",
            "OrRd",
            "PuRd",
            "RdPu",
            "BuPu",
            "GnBu",
            "PuBu",
            "YlGnBu",
            "PuBuGn",
            "BuGn",
            "YlGn",
            "hsv",
        ]
        default_cmap = getattr(st.session_state, "preset_cmap", "jet")
        if default_cmap in colormap_options:
            default_index = colormap_options.index(default_cmap)
        else:
            default_index = 0

        cmap_name = st.sidebar.selectbox(
            "Colormap:",
            options=colormap_options,
            index=default_index,  # Default to jet for better visibility
            help="Choose the colormap for the heatmap visualization. "
            "Jet and rainbow provide high contrast.",
        )

        # Alpha (transparency) setting
        default_alpha = getattr(st.session_state, "preset_alpha", 255)
        alpha = st.sidebar.slider(
            "Transparency (Alpha):",
            min_value=50,
            max_value=255,
            value=default_alpha,  # Default to fully opaque for better visibility
            help="Adjust the transparency of the heatmap overlay "
            "(50=very transparent, 255=opaque)",
        )

        # Colorbar option
        show_colorbar = st.sidebar.checkbox(
            "Show Colorbar",
            value=True,
            help="Display a colorbar to interpret the heatmap values",
        )

        # Value range settings
        use_custom_range = st.sidebar.checkbox(
            "Use Custom Value Range",
            value=False,
            help="Specify custom min/max values for normalization",
        )

        value_range = None
        if use_custom_range:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                vmin = st.number_input("Min Value", value=0.0, step=0.1)
            with col2:
                vmax = st.number_input("Max Value", value=1.0, step=0.1)
            value_range = (vmin, vmax)

        # Value clipping settings
        default_clip = getattr(st.session_state, "preset_clip", False)
        clip_values = st.sidebar.checkbox(
            "Clip Values",
            value=default_clip,
            help="Clip values outside a specified range",
        )

        clip_range = (0, 1)
        if clip_values:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                clip_min = st.number_input("Clip Min", value=0.0, step=0.1)
            with col2:
                clip_max = st.number_input("Clip Max", value=1.0, step=0.1)
            clip_range = (clip_min, clip_max)

    if st.sidebar.button("Generate Map"):
        generate_map(
            directory,
            cmap_name=cmap_name,
            alpha=alpha,
            show_colorbar=show_colorbar,
            value_range=value_range,
            clip_values=clip_values,
            clip_range=clip_range,
        )
    else:
        # Display empty map
        fig, _ = create_map_with_geotiff_tiles(tiles_to_overlay=[], show_colorbar=False)
        st.plotly_chart(fig, use_container_width=True)

        # Show help information
        st.info(
            """
        **Welcome to InstaGeo Serve!**

        This tool helps you visualize geospatial inference predictions as interactive heatmaps.

        **Features:**
        - Multiple colormap options for different visualization needs
        - Adjustable transparency for overlay visualization
        - Colorbar for value interpretation
        - Custom value range normalization
        - Value clipping for focusing on specific ranges

        **Instructions:**
        1. Enter the path to your GeoTIFF files
        2. Select country, year, and month
        3. Customize visualization settings in the sidebar
        4. Click "Generate Map" to visualize your predictions
        """
        )


if __name__ == "__main__":
    main()
