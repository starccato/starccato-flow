"""Analysis and visualization functions for model evaluation."""

from pathlib import Path
from typing import List, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib import rcParams
import numpy as np
import pandas as pd
import torch
from PIL import Image
import io

from ..utils.defaults_general import TEN_KPC
from ..utils.defaults_plotting import (
    SIGNAL_COLOUR,
    GENERATED_SIGNAL_COLOUR,
    SIGNAL_LIM_UPPER,
    SIGNAL_LIM_LOWER,
    PARAMETER_LABELS,
    PARAMETER_RANGES,
    CM_TO_INCHES
)
from . import set_plot_style, get_time_axis
from .signals import plot_signal_grid

def _is_dark_color(color_str: str) -> bool:
    """Determine if a color (hex or named) is dark or light.
    
    Returns True if the color is dark (text should be white), False if light (text should be black).
    """
    # Handle named colors
    if color_str.lower() == "black":
        return True
    elif color_str.lower() == "white":
        return False
    elif color_str.lower() in ("navy", "darkblue", "darkred", "darkgreen"):
        return True
    
    # Handle hex colors - calculate luminance using sRGB formula
    if color_str.startswith("#"):
        hex_str = color_str.lstrip("#")
        if len(hex_str) == 6:
            try:
                r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
                # Normalize to 0-1
                r, g, b = r / 255.0, g / 255.0, b / 255.0
                # Apply gamma correction
                r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
                g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
                b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
                # Calculate relative luminance
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                # Consider dark if luminance < 0.5
                return luminance < 0.5
            except ValueError:
                pass
    
    # Default to light color (text should be black)
    return False

def plot_surface_density(fname=None, font_family=None, font_name=None, transparent=False, figsize: tuple[float, float] = (14.5, 6)):
    """Plot surface density of supernovae in the galactic plane."""
    set_plot_style(background="white", font_family=font_family, font_name=font_name)

    plt.rcParams["font.family"] = font_family
    if font_family == "sans-serif":
        plt.rcParams["font.sans-serif"] = [font_name]
    elif font_family == "serif":
        plt.rcParams["font.serif"] = [font_name]

    # Generate radius values
    r = np.linspace(0, 30, 1000)

    # Model parameters
    A = 1.96
    r_0 = 17.2
    theta_0 = 0.08
    beta = 0.13

    # Surface density
    surface_density = (
        A
        * np.sin((np.pi * r) / r_0 + theta_0)
        * np.exp(-beta * r)
    )

    # Create figure and axes
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    _, ax = plt.subplots(figsize=figsize, facecolor="white")

    # ax.plot(r, surface_density, color="lightblue", linewidth=2)
    ax.fill_between(r, surface_density, color="lightblue", alpha=0.3)

    # Put ticks only on visible axes
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.tick_params(labelsize=11)

    # Turn off grid
    ax.grid(False)

    ax.set_xlabel(r"$r\ (\mathrm{kpc})$", size=11)
    ax.set_ylabel("Surface Density", size=11)

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 1.0)

    if fname:
        plt.savefig(
            fname,
            dpi=300,
            bbox_inches="tight",
            transparent=transparent
        )

    plt.show()


def plot_galactic_distribution(
    galactic_coords: np.ndarray,
    sun_location: Optional[np.ndarray] = None,
    highlight_indices: Optional[np.ndarray] = None,
    fname_xy: Optional[str] = None,
    background: str = "white",
    transparent: Optional[bool] = None,
    light_year: bool = False,
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    scatter_size: float = 0.001,
    sun_marker_size: float = 100,
    show: bool = False,
    dpi: int = 300,
    figsize: tuple = (14.5, 14.5),
    rasterize_scatter: bool = True,
    line_weight: float = 1,
    fontsize_tick: int = 11,
    fontsize_title: int = 16,
    left_margin: float = 0.15,
) -> List[plt.Figure]:
    """Plot galactic supernova locations in the X-Y plane.

    Args:
        galactic_coords (np.ndarray): Cartesian galactic coordinates with shape (N, 3)
        sun_location (Optional[np.ndarray]): Sun position in galactic coordinates
        highlight_indices (Optional[np.ndarray]): Optional indices of supernovae to
            draw as highlighted points.
        fname_3d (Optional[str]): Deprecated. Unused.
        fname_xy (Optional[str]): Output path for the X-Y projection plot
        fname_xz (Optional[str]): Deprecated. Unused.
        fname_xy_closeup (Optional[str]): Deprecated. Unused.
        fname_yx_zx (Optional[str]): Deprecated. Unused.
        background (str): Plot theme, either "white" or "black"
        transparent (Optional[bool]): Override the saved figure transparency
        light_year (bool): If True, convert plot coordinates from kpc to light-years
        font_family (str): Font family to use
        font_name (str): Specific font name to use
        scatter_size (float): Marker size for supernova points
        sun_marker_size (float): Marker size for the sun marker
        show (bool): Whether to keep figures open and display them
        dpi (int): DPI used when saving output files
        legend_frameon (bool): Whether to display the legend box background
        figsize (tuple): Figure size in inches as (width, height). Default (16, 16) produces ~2400x2400 pixels at 150 dpi. For 2000x2000 pixels use ~(13.3, 13.3)

    Returns:
        List[plt.Figure]: A single-item list containing the X-Y figure.
    """
    galactic_coords = np.asarray(galactic_coords)
    if galactic_coords.ndim != 2 or galactic_coords.shape[1] != 3:
        raise ValueError("galactic_coords must have shape (N, 3).")

    if sun_location is None:
        sun_location = np.array([0.0, 8.178, 0.0208], dtype=float)
    else:
        sun_location = np.asarray(sun_location, dtype=float)
        if sun_location.shape != (3,):
            raise ValueError("sun_location must have shape (3,).")

    highlight_coords = None
    if highlight_indices is not None:
        highlight_indices = np.asarray(highlight_indices)
        if highlight_indices.ndim != 1:
            raise ValueError("highlight_indices must be a 1D array of indices.")
        highlight_coords = galactic_coords[highlight_indices]

    kpc_to_ly = 3261.56
    coord_scale = kpc_to_ly if light_year else 1.0
    galactic_coords = galactic_coords * coord_scale
    sun_location = sun_location * coord_scale

    x, y, _ = galactic_coords.T
    xy_radius = 33
    xy_radius *= 1.02
    if highlight_coords is not None and highlight_coords.size > 0:
        hx, hy, _ = highlight_coords.T
    else:
        hx = hy = None
    text_color = "white" if _is_dark_color(background) else "black"
    legend_facecolor = "black" if _is_dark_color(background) else "white"
    if transparent is None:
        transparent = _is_dark_color(background)
    facecolor = "none" if transparent else background

    def _prepare_output_path(path: Optional[str]) -> Optional[Path]:
        if path is None:
            return None
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _style_2d_axes(axes: plt.Axes) -> None:
        axes.tick_params(colors=text_color, labelsize=fontsize_tick, direction="inout", length=fontsize_tick, width=line_weight)
        for spine in axes.spines.values():
            spine.set_color(text_color)
            spine.set_linewidth(line_weight)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.set_aspect("equal", adjustable="box")
        if light_year:
            axes.xaxis.set_major_locator(mticker.MultipleLocator(20_000))
            axes.yaxis.set_major_locator(mticker.MultipleLocator(20_000))
            axes.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda val, pos: _light_year_tick_label(val))
            )
            axes.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda val, pos: _light_year_tick_label(val))
            )
        else:
            axes.xaxis.set_major_locator(mticker.MultipleLocator(5))
            axes.yaxis.set_major_locator(mticker.MultipleLocator(5))
            axes.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
            axes.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))

    def _light_year_tick_label(val: float) -> str:
        if np.isclose(val, 0.0):
            return "0"
        return f"{val:,.0f}\n${{}}_{{\\mathrm{{light\\ years}}}}$"

    def _tighten_light_year_tick_lines(axes: plt.Axes) -> None:
        for tick_label in list(axes.get_xticklabels()) + list(axes.get_yticklabels()):
            tick_label.set_linespacing(0.75)

    def _axis_label(base: str) -> str:
        return f"{base} (kpc)" if not light_year else base

    def _apply_xy_axis_line_window(axes: plt.Axes) -> None:
        if light_year:
            axes.spines["bottom"].set_bounds(-80_000, 80_000)
            axes.spines["left"].set_bounds(-80_000, 80_000)
            axes.spines["bottom"].set_linestyle("--")
            axes.spines["left"].set_linestyle("--")
        else:
            axes.spines["bottom"].set_bounds(-25, 25)
            axes.spines["left"].set_bounds(-25, 25)

    def _legend_with_supernova_marker(axes: plt.Axes) -> None:
        handles, labels = axes.get_legend_handles_labels()
        handle_map = dict(zip(labels, handles))

        preferred_order = [
            "Supernova",
            "Sampled Supernova",
            "Galactic Center: Sgr A*",
            "Sun",
        ]
        ordered_labels = [label for label in preferred_order if label in handle_map]
        ordered_labels += [label for label in labels if label not in ordered_labels]

        adjusted_handles = []
        for label in ordered_labels:
            if label == "Supernova":
                adjusted_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        linestyle="None",
                        marker="o",
                        markersize=9,
                        markerfacecolor="lightblue",
                        markeredgecolor="none",
                    )
                )
            elif label == "Sampled Supernova":
                adjusted_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        linestyle="None",
                        marker="o",
                        markersize=9,
                        markerfacecolor="orange",
                        markeredgecolor="none",
                    )
                )
            else:
                adjusted_handles.append(handle_map[label])

        legend_ncol = 2 if len(ordered_labels) == 4 else max(1, len(ordered_labels))
        axes.legend(
            adjusted_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=legend_ncol,
            facecolor=legend_facecolor,
            edgecolor="none",
            labelcolor=text_color,
            fontsize=fontsize_tick,
            frameon=False,
        )

    output_xy = _prepare_output_path(fname_xy)

    fig2 = plt.figure(figsize=(figsize[0]/CM_TO_INCHES, figsize[1]/CM_TO_INCHES), facecolor=facecolor)
    ax1 = fig2.add_subplot(111, facecolor=facecolor)
    ax1.scatter(x, y, s=scatter_size, alpha=1, c="lightblue", label="Supernova", rasterized=rasterize_scatter)
    ax1.scatter(
        0.0,
        0.0,
        s=sun_marker_size,
        c="black",
        edgecolors="orange",
        linewidths=1.8,
        marker="o",
        label="Galactic Center: Sgr A*",
    )
    # ax1.scatter(sun_location[0], sun_location[1], s=sun_marker_size, c="yellow", marker="*", label="Sun")
    ax1.scatter(
        sun_location[0],
        sun_location[1],
        s=sun_marker_size,
        c="yellow",
        marker="*",
        edgecolor="white" if background == "black" else "black",
        linewidths=0.75,
        label="Sun",
        zorder=20,
    )
    if hx is not None:
        ax1.scatter(
            hx,
            hy,
            s=max(sun_marker_size * 0.1, 10),
            c="orange",
            edgecolors="none",
            marker="o",
            label="Sampled Supernova",
            zorder=10,
        )
    ax1.set_xlabel(_axis_label("X"), color=text_color, fontsize=fontsize_title)
    ax1.set_ylabel(_axis_label("Y"), color=text_color, fontsize=fontsize_title)

    n_supernovae = galactic_coords.shape[0]
    ax1.text(
        0.95,
        0.02,
        f"n={n_supernovae:,}",
        transform=ax1.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize_tick,
        color=text_color,
    )

    _style_2d_axes(ax1)
    ax1.set_xlim(-xy_radius, xy_radius)
    ax1.set_ylim(-xy_radius, xy_radius)
    if light_year:
        tick_values = np.arange(-80_000, 80_001, 20_000)
        axis_padding = 5_000
        ax1.set_xlim(tick_values[0] - axis_padding, tick_values[-1] + axis_padding)
        ax1.set_ylim(tick_values[0] - axis_padding, tick_values[-1] + axis_padding)
        ax1.set_xticks(tick_values)
        ax1.set_yticks(tick_values)
        _tighten_light_year_tick_lines(ax1)
    else:
        # Set kpc limits to match light-year equivalent (-85000 to 85000 ly = -26.07 to 26.07 kpc)
        kpc_limit = 26.07
        kpc_padding = 2.07
        tick_values = np.arange(-25, 26, 5)
        ax1.set_xlim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
        ax1.set_ylim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
        ax1.set_xticks(tick_values)
        ax1.set_yticks(tick_values)
    _apply_xy_axis_line_window(ax1)
    _legend_with_supernova_marker(ax1)
    plt.tight_layout()
    if output_xy is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*'mode' parameter is deprecated and will be removed in Pillow 13.*",
                category=DeprecationWarning,
                module=r"matplotlib\.backends\.backend_pdf",
            )
            fig2.savefig(
                output_xy,
                dpi=dpi,
                bbox_inches="tight",
                transparent=transparent,
                facecolor=facecolor,
            )

    if show:
        plt.show()
    else:
        plt.close(fig2)

    return [fig2]

# def plot_galactic_distribution_with_posterior(
#     galactic_coords: np.ndarray,
#     posterior_ra: np.ndarray,
#     posterior_dec: np.ndarray,
#     posterior_distance: np.ndarray,
#     true_ra: Optional[float] = None,
#     true_dec: Optional[float] = None,
#     true_distance: Optional[float] = None,
#     sun_location: Optional[np.ndarray] = None,
#     fname: Optional[str] = None,
#     background: str = "white",
#     transparent: Optional[bool] = None,
#     font_family: str = "sans-serif",
#     font_name: str = "Avenir",
#     scatter_size: float = 0.001,
#     sun_marker_size: float = 100,
#     show: bool = False,
#     dpi: int = 300,
#     figsize: tuple = (30.5, 30.5),
# ) -> plt.Figure:
#     """Plot galactic supernova distribution in X-Y plane with posterior credible regions overlaid.

#     Args:
#         galactic_coords (np.ndarray): Cartesian galactic coordinates with shape (N, 3)
#         posterior_ra (np.ndarray): Posterior RA samples in radians
#         posterior_dec (np.ndarray): Posterior Dec samples in radians
#         posterior_distance (np.ndarray): Posterior distance samples in kpc
#         true_ra (Optional[float]): True RA in radians
#         true_dec (Optional[float]): True Dec in radians
#         true_distance (Optional[float]): True distance in kpc
#         sun_location (Optional[np.ndarray]): Sun position in galactic coordinates
#         fname (Optional[str]): Output path for the plot
#         background (str): Plot theme, either "white" or "black"
#         transparent (Optional[bool]): Override the saved figure transparency
#         font_family (str): Font family to use
#         font_name (str): Specific font name to use
#         scatter_size (float): Marker size for background supernova points
#         sun_marker_size (float): Marker size for the sun marker
#         show (bool): Whether to keep figure open and display it
#         dpi (int): DPI used when saving output files
#         figsize (tuple): Figure size in inches as (width, height)

#     Returns:
#         plt.Figure: The created matplotlib figure
#     """
#     from ..supernovae.supernovae import Supernovae
#     from matplotlib.patches import Patch
#     from matplotlib.colors import to_rgba

#     galactic_coords = np.asarray(galactic_coords)
#     if galactic_coords.ndim != 2 or galactic_coords.shape[1] != 3:
#         raise ValueError("galactic_coords must have shape (N, 3).")

#     if sun_location is None:
#         sun_location = np.array([0.0, 8.178, 0.0208], dtype=float)
#     else:
    #     sun_location = np.asarray(sun_location, dtype=float)
    #     if sun_location.shape != (3,):
    #         raise ValueError("sun_location must have shape (3,).")

    # # Set up plot styling
    # rcParams["font.family"] = font_family
    # rcParams["font.size"] = 22
    # if font_family == "sans-serif":
    #     rcParams["font.sans-serif"] = [font_name]
    # elif font_family == "serif":
    #     rcParams["font.serif"] = [font_name]

    # facecolor = background
    # text_color = "white" if _is_dark_color(background) else "black"
    # grid_color = "gray" if _is_dark_color(background) else "lightgray"
    # transparent = transparent if transparent is not None else _is_dark_color(background)
    # plot_facecolor = "none" if transparent else background

    # # Extract X-Y coordinates from background galactic distribution
    # x = galactic_coords[:, 0]
    # y = galactic_coords[:, 1]

    # # Transform posterior samples to galactic coordinates
    # sn_temp = Supernovae(complex=True)  # Temporary instance for coordinate transformation
    # post_x, post_y, post_z = sn_temp.equatorial_to_galactic(
    #     posterior_ra, posterior_dec, posterior_distance
    # )

    # # Apply the same 90° rotation used for highlighted points in the galactic plots
    # # so the posterior overlay uses the same visual orientation as the background.
    # # rotation_angle = np.deg2rad(-90.0)
    # # rotation_matrix = np.array([
    # #     [np.cos(rotation_angle), -np.sin(rotation_angle), 0.0],
    # #     [np.sin(rotation_angle), np.cos(rotation_angle), 0.0],
    # #     [0.0, 0.0, 1.0],
    # # ])
    # # posterior_coords = np.column_stack([post_x, post_y, post_z])
    # # posterior_coords = posterior_coords @ rotation_matrix.T
    # # post_x, post_y, post_z = posterior_coords.T
    
    # # # Convert posterior from heliocentric to galactocentric frame by adding sun location
    # # post_x += sun_location[0]
    # # post_y += sun_location[1]
    # # post_z += sun_location[2]

    # # Create figure with proper styling (matching plot_galactic_distribution)
    # figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    # fig = plt.figure(figsize=figsize, facecolor=plot_facecolor)
    # ax = fig.add_subplot(111, facecolor=facecolor)

    # # Plot background galactic distribution (exactly as in plot_galactic_distribution) - rasterized to reduce file size
    # ax.scatter(x, y, s=scatter_size, alpha=1, c="lightblue", label="Supernova", rasterized=True)
    # ax.scatter(
    #     0.0,
    #     0.0,
    #     s=sun_marker_size,
    #     c="black",
    #     edgecolors="white",
    #     linewidths=1.8,
    #     marker="o",
    #     label="Galactic Center: Sgr A*",
    # )
    # ax.scatter(sun_location[0], sun_location[1], s=sun_marker_size, c="yellow", marker="*", label="Sun", zorder=20)

    # # Add density contours from posterior samples in X-Y plane (ONLY DIFFERENCE: add this layer)
    # from scipy.stats import gaussian_kde
    # from matplotlib.colors import to_rgba
    
    # # Build KDE from posterior X-Y coordinates for credible contours
    # xy_data = np.vstack([post_x, post_y])
    # try:
    #     kde = gaussian_kde(xy_data)
        
    #     # Create grid for evaluating KDE
    #     x_min, x_max = post_x.min(), post_x.max()
    #     y_min, y_max = post_y.min(), post_y.max()
    #     x_grid = np.linspace(x_min - 2, x_max + 2, 200)
    #     y_grid = np.linspace(y_min - 2, y_max + 2, 200)
    #     X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
    #     positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
    #     density = kde(positions).reshape(X_mesh.shape)
        
    #     # Compute credible levels from density CDF
    #     sorted_density = np.sort(density.ravel())[::-1]
    #     cdf = np.cumsum(sorted_density) / np.sum(sorted_density)
        
    #     posterior_probs = [0.68, 0.90, 0.95]
    #     contour_levels = []
    #     for p in posterior_probs:
    #         idx = np.searchsorted(cdf, p, side="left")
    #         idx = min(idx, len(sorted_density) - 1)
    #         contour_levels.append(float(sorted_density[idx]))
        
    #     contour_levels = np.sort(contour_levels)
    #     contour_top = max(contour_levels[-1] * 1.001, np.max(sorted_density) * 1.001)
    #     contour_fill_levels = np.concatenate([contour_levels, [contour_top]])
        
    #     # Red fill colors matching celestial map (red with varying alphas: 0.40, 0.62, 0.88)
    #     red_fill_colors = [
    #         to_rgba("red", alpha=0.40),    # 68%
    #         to_rgba("red", alpha=0.62),    # 90%
    #         to_rgba("red", alpha=0.88),    # 95%
    #     ]
        
    #     # Plot filled contours as overlay (no label - not in legend)
    #     ax.contourf(
    #         X_mesh,
    #         Y_mesh,
    #         density,
    #         levels=contour_fill_levels,
    #         colors=red_fill_colors,
    #         antialiased=True,
    #     )
    # except Exception as e:
    #     # If KDE fails, just skip contours
    #     pass

    # # Plot true location if provided (matching celestial map marker: deepskyblue "x")
    # if true_ra is not None and true_dec is not None and true_distance is not None:
    #     true_x, true_y, true_z = sn_temp.equatorial_to_galactic(
    #         np.array([true_ra]), np.array([true_dec]), np.array([true_distance])
    #     )

    #     # rotation_matrix = np.array([
    #     # [np.cos(rotation_angle), -np.sin(rotation_angle), 0.0],
    #     # [np.sin(rotation_angle), np.cos(rotation_angle), 0.0],
    #     # [0.0, 0.0, 1.0],
    #     # ])
    #     # true_coords = np.column_stack([true_x, true_y, true_z])
    #     # true_coords = true_coords @ rotation_matrix.T
    #     # true_x, true_y, true_z = true_coords.T
    #     # Plot with same marker style as celestial map (deepskyblue "x")
    #     ax.scatter(
    #         true_x,
    #         true_y,
    #         s=72,
    #         marker="x",
    #         c="deepskyblue",
    #         linewidths=1.8,
    #         zorder=10,
    #         label="True Location",
    #     )

    # # Style axes exactly like plot_galactic_distribution
    # ax.set_xlabel("X (kpc)", color=text_color, fontsize=22)
    # ax.set_ylabel("Y (kpc)", color=text_color, fontsize=22)
    
    # # _style_2d_axes equivalent
    # ax.tick_params(colors=text_color, labelsize=18, direction="inout", length=12, width=1.4)
    # for spine in ax.spines.values():
    #     spine.set_color(text_color)
    # ax.spines["top"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    # ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    # ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
    # ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
    
    # # Set limits and ticks
    # xy_radius = 33
    # kpc_limit = 26.07
    # kpc_padding = 2.07
    # tick_values = np.arange(-25, 26, 5)
    # ax.set_xlim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
    # ax.set_ylim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
    # ax.set_xticks(tick_values)
    # ax.set_yticks(tick_values)
    
    # # _apply_xy_axis_line_window equivalent
    # ax.spines["bottom"].set_bounds(-25, 25)
    # ax.spines["left"].set_bounds(-25, 25)
    
    # ax.set_aspect("equal")
    # ax.grid(color=grid_color, alpha=0.2)
    
    # # Add legend (matching plot_galactic_distribution style)
    # legend_facecolor = "black" if _is_dark_color(background) else "white"
    # handles, labels = ax.get_legend_handles_labels()
    # adjusted_handles = []
    # for handle, label in zip(handles, labels):
    #     if label == "Supernova":
    #         adjusted_handles.append(
    #             mlines.Line2D(
    #                 [],
    #                 [],
    #                 linestyle="None",
    #                 marker="o",
    #                 markersize=9,
    #                 markerfacecolor="lightblue",
    #                 markeredgecolor="none",
    #             )
    #         )
    #     else:
    #         adjusted_handles.append(handle)

    # legend = ax.legend(
    #     adjusted_handles,
    #     labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.02),
    #     ncol=max(1, len(labels)),
    #     facecolor=legend_facecolor,
    #     edgecolor="none",
    #     framealpha=0.0,
    #     fontsize=14,
    #     labelcolor=text_color,
    # )

    # if fname is not None:
    #     # Determine format from filename extension
    #     fmt = fname.split('.')[-1].lower() if '.' in fname else 'png'
    #     # Use lower DPI for vector formats (SVG is vector-based, doesn't need 300 DPI)
    #     save_dpi = 100 if fmt == 'svg' else dpi
    #     fig.savefig(fname, dpi=save_dpi, bbox_inches="tight", transparent=transparent)

    # if show:
    #     plt.show()
    # else:
    #     plt.close(fig)

    # plt.rcdefaults()
    # return fig


def plot_galactic_distribution_with_posterior_zoom(
    galactic_coords: np.ndarray,
    posterior_ra: np.ndarray,
    posterior_dec: np.ndarray,
    posterior_distance: np.ndarray,
    true_ra: Optional[float] = None,
    true_dec: Optional[float] = None,
    true_distance: Optional[float] = None,
    sun_location: Optional[np.ndarray] = None,
    fname: Optional[str] = None,
    figsize: tuple[float, float] = (12.5, 12.5),
    scatter_size: float = 0.00005,
    sun_marker_size: float = 400,
    background: str = "white",
    dpi: int = 300,
    show: bool = False,
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    transparent: bool = False,
) -> plt.Figure:
    """Plot galactic distribution (X-Y plane) with posterior contours in 10 kpc zoom around sun.
    
    This is a zoomed version of plot_galactic_distribution_with_posterior that shows only
    the region within 10 kpc of the sun, with no legend, ticks, or axis markers.
    
    Args:
        galactic_coords: Background galactic coordinates (N, 3) in kpc
        posterior_ra: Posterior RA samples (radians)
        posterior_dec: Posterior Dec samples (radians)
        posterior_distance: Posterior distance samples (kpc)
        true_ra: True RA (radians)
        true_dec: True Dec (radians)
        true_distance: True distance (kpc)
        sun_location: Sun location in galactocentric frame (default [0.0, 8.178, 0.0208])
        fname: Output filename
        figsize: Figure size in cm as (width, height)
        scatter_size: Size of background scatter points
        sun_marker_size: Size of sun marker
        background: Background color
        dpi: Resolution for saving
        show: Whether to display plot
        font_family: Font family
        font_name: Font name
        transparent: Whether to save transparent
    
    Returns:
        Matplotlib figure
    """
    # Convert cm to inches
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    if sun_location is None:
        sun_location = np.array([0.0, 8.178, 0.0208])
    
    # Set up plot styling
    rcParams["font.family"] = font_family
    rcParams["font.size"] = 18
    if font_family == "sans-serif":
        rcParams["font.sans-serif"] = [font_name]
    elif font_family == "serif":
        rcParams["font.serif"] = [font_name]

    facecolor = background
    text_color = "white" if _is_dark_color(background) else "black"
    grid_color = "gray" if _is_dark_color(background) else "lightgray"
    transparent = transparent if transparent is not None else _is_dark_color(background)
    plot_facecolor = "none"  # Always transparent
    ax_facecolor = "none"  # Always transparent so clip path creates circular boundary

    # Extract X-Y coordinates from background galactic distribution
    x = galactic_coords[:, 0]
    y = galactic_coords[:, 1]
    
    # Define zoom radius for filtering
    zoom_radius = 10.0
    
    # Filter background stars to only those within zoom_radius of sun location (X-Y plane)
    distances_from_sun = np.sqrt((x - sun_location[0])**2 + (y - sun_location[1])**2)
    within_radius = distances_from_sun <= zoom_radius
    x_filtered = x[within_radius]
    y_filtered = y[within_radius]

    # Transform posterior samples to galactic coordinates
    from ..supernovae.supernovae import Supernovae
    sn_temp = Supernovae(complex=True)  # Temporary instance for coordinate transformation
    post_x, post_y, post_z = sn_temp.equatorial_to_galactic(
        posterior_ra, posterior_dec, posterior_distance
    )

    # Create figure with proper styling (matching plot_galactic_distribution)
    fig = plt.figure(figsize=figsize, facecolor=plot_facecolor)
    ax = fig.add_subplot(111, facecolor=ax_facecolor)
    
    # Black hole at galactic center: two circles (accretion disk outer + event horizon interior)
    from matplotlib.patches import Circle
    bh_disk_outer = Circle(
        (0.0, 0.0), 0.5, color="orange", alpha=0.8, zorder=8
    )
    ax.add_patch(bh_disk_outer)
    bh_interior = Circle(
        (0.0, 0.0), 0.35, color="black", alpha=0.95, zorder=9
    )
    ax.add_patch(bh_interior)
    
    ax.scatter(sun_location[0], sun_location[1], s=sun_marker_size, c="yellow", marker="*", zorder=20)

    # Add density contours from posterior samples in X-Y plane
    from scipy.stats import gaussian_kde
    from matplotlib.colors import to_rgba
    
    # Build KDE from posterior X-Y coordinates for credible contours
    xy_data = np.vstack([post_x, post_y])
    try:
        kde = gaussian_kde(xy_data)
        
        # Create grid for evaluating KDE (restricted to zoom region)
        x_min = sun_location[0] - zoom_radius
        x_max = sun_location[0] + zoom_radius
        y_min = sun_location[1] - zoom_radius
        y_max = sun_location[1] + zoom_radius
        
        x_grid = np.linspace(x_min, x_max, 200)
        y_grid = np.linspace(y_min, y_max, 200)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X_mesh.ravel(), Y_mesh.ravel()])
        density = kde(positions).reshape(X_mesh.shape)
        
        # Compute credible levels from density CDF
        sorted_density = np.sort(density.ravel())[::-1]
        cdf = np.cumsum(sorted_density) / np.sum(sorted_density)
        
        posterior_probs = [0.68, 0.90, 0.95]
        contour_levels = []
        for p in posterior_probs:
            idx = np.searchsorted(cdf, p, side="left")
            idx = min(idx, len(sorted_density) - 1)
            contour_levels.append(float(sorted_density[idx]))
        
        contour_levels = np.sort(contour_levels)
        contour_top = max(contour_levels[-1] * 1.001, np.max(sorted_density) * 1.001)
        contour_fill_levels = np.concatenate([contour_levels, [contour_top]])
        
        # Red fill colors matching celestial map
        red_fill_colors = [
            to_rgba("red", alpha=0.40),    # 68%
            to_rgba("red", alpha=0.62),    # 90%
            to_rgba("red", alpha=0.88),    # 95%
        ]
        
        # Plot filled contours as overlay (no label - not in legend)
        ax.contourf(
            X_mesh,
            Y_mesh,
            density,
            levels=contour_fill_levels,
            colors=red_fill_colors,
            antialiased=True,
        )
    except Exception as e:
        # If KDE fails, just skip contours
        pass

    # Plot true location if provided
    if true_ra is not None and true_dec is not None and true_distance is not None:
        true_x, true_y, true_z = sn_temp.equatorial_to_galactic(
            np.array([true_ra]), np.array([true_dec]), np.array([true_distance])
        )
        
        # true_coords = np.column_stack([true_x, true_y, true_z])
        # true_coords = true_coords @ rotation_matrix.T
        # true_x, true_y, true_z = true_coords.T

        # Plot with same marker style as celestial map (deepskyblue "x" with size matching sky map)
        ax.scatter(
            true_x,
            true_y,
            s=72,
            marker="x",
            c="deepskyblue",
            linewidths=1.8,
            zorder=10,
        )

    # Style axes exactly like plot_galactic_distribution but zoomed
    ax.tick_params(colors=text_color, labelsize=18, direction="inout", length=12, width=1.4)
    for spine in ax.spines.values():
        spine.set_color(text_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Set zoom limits around sun (10 kpc radius) with small buffer to prevent border clipping
    buffer = 0.3  # Small buffer in kpc to prevent clipping at edges
    ax.set_xlim(sun_location[0] - zoom_radius - buffer, sun_location[0] + zoom_radius + buffer)
    ax.set_ylim(sun_location[1] - zoom_radius - buffer, sun_location[1] + zoom_radius + buffer)
    
    # No clip path - circles will define the boundary
    
    # Remove ticks and axis markers
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Remove spine bounds
    ax.spines["bottom"].set_bounds(0, 0)
    ax.spines["left"].set_bounds(0, 0)
    
    ax.set_aspect("equal")
    
    # Add filled circle patch with background color underneath everything (zorder=0)
    # Only add when NOT transparent
    if not transparent:
        circle_fill = mpatches.Circle(
            (sun_location[0], sun_location[1]),
            zoom_radius,
            fill=True,
            facecolor=background,
            edgecolor="none",
            zorder=0
        )
        ax.add_patch(circle_fill)
    
    # Add white dashed border around the circle
    circle_border = mpatches.Circle(
        (sun_location[0], sun_location[1]),
        zoom_radius,
        fill=False,
        edgecolor="white",
        linestyle="--",
        linewidth=1.5,
        zorder=15
    )
    ax.add_patch(circle_border)
    
    # Add arrow and text showing radius from sun to top of circle
    arrow_start_y = sun_location[1]
    arrow_end_y = sun_location[1] + zoom_radius
    arrow_x = sun_location[0]
    
    # Draw arrow from sun to top of circle
    ax.annotate(
        "",
        xy=(arrow_x, arrow_end_y),
        xytext=(arrow_x, arrow_start_y),
        arrowprops=dict(
            arrowstyle="->",
            color="white",
            lw=1.5,
            zorder=16
        )
    )
    
    # Add text label for radius
    text_y = sun_location[1] + zoom_radius / 2
    ax.text(
        arrow_x + 1.0,
        text_y,
        "10 kpc",
        fontsize=16,
        color="white",
        verticalalignment="center",
        zorder=16
    )
    
    ax.grid(color=grid_color, alpha=0.2)

    if fname is not None:
        # Determine format from filename extension
        fmt = fname.split('.')[-1].lower() if '.' in fname else 'png'
        # Use lower DPI for vector formats (SVG is vector-based, doesn't need 300 DPI)
        save_dpi = 100 if fmt == 'svg' else dpi
        fig.savefig(fname, dpi=save_dpi, bbox_inches="tight", transparent=transparent, format=fmt)

    if show:
        plt.show()
    else:
        plt.close(fig)

    plt.rcdefaults()
    return fig


def plot_reconstruction_distribution(
    reconstructed_signals: List[np.ndarray],
    noisy_signal: torch.Tensor,
    true_signal: torch.Tensor,
    max_value: float,
    num_samples: int = 1000,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    figsize: tuple[float, float] = (20, 15)
):
    """Plot distribution of multiple reconstructions of a single signal.
    
    Args:
        reconstructed_signals (List[np.ndarray]): List of reconstructed signals
        noisy_signal (torch.Tensor): Noisy version of signal
        true_signal (torch.Tensor): True clean signal
        max_value (float): Maximum value for scaling
        num_samples (int): Number of reconstructions
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    """
    set_plot_style(background, font_family, font_name)
    vline_color = "white" if _is_dark_color(background) else "black"

    # Prepare data
    reconstructed_signals = np.array(reconstructed_signals)
    true_signal_np = true_signal.squeeze().cpu().numpy() * max_value
    noisy_signal_np = noisy_signal.squeeze().cpu().numpy() * max_value
    reconstructed_signals_df = pd.DataFrame(reconstructed_signals.T)
    d = get_time_axis()

    # Create figure
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Plot percentiles
    p2_5 = reconstructed_signals_df.quantile(0.025, axis=1)
    p97_5 = reconstructed_signals_df.quantile(0.975, axis=1)
    p25 = reconstructed_signals_df.quantile(0.25, axis=1)
    p75 = reconstructed_signals_df.quantile(0.75, axis=1)

    ax.fill_between(d, p2_5, p97_5, color="white", alpha=0.2)
    ax.fill_between(d, p2_5, p97_5, color=GENERATED_SIGNAL_COLOUR, alpha=0.4)
    ax.fill_between(d, p25, p75, color="white", alpha=0.4)
    ax.fill_between(d, p25, p75, color=GENERATED_SIGNAL_COLOUR, alpha=0.6)

    # Plot original signal
    ax.plot(d, true_signal_np, color="black", 
            linewidth=1, alpha=0.75, zorder=3)
    # Plot noisy signal
    ax.plot(d, noisy_signal_np, color="deepskyblue", 
            linewidth=1, alpha=0.5, zorder=4)

    # Style the plot
    ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    ax.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax.set_xlim(min(d), max(d))
    ax.grid(True, alpha=0.3)
    
    # Style axes and labels
    ax.tick_params(axis="both", colors=vline_color, labelsize=12)
    ax.set_xlabel("time (s)", fontsize=16, color=vline_color)
    ax.set_ylabel("h", fontsize=16, color=vline_color)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(vline_color)

    # Add sample size note
    plt.text(
        0.98, 0.02, f"n = {num_samples}",
        ha="right", va="bottom",
        transform=ax.transAxes,
        fontsize=12, color=vline_color,
        alpha=0.8
    )

    # Add legend
    legend_handles = [
        mpatches.Patch(color=GENERATED_SIGNAL_COLOUR, alpha=0.6, 
                      label="Central 95%"),
        mpatches.Patch(color=GENERATED_SIGNAL_COLOUR, alpha=1.0, 
                      label="Central 50%"),
        mlines.Line2D([], [], color="deepskyblue", linewidth=2, 
                     label="Original Signal")
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=12,
        facecolor="none",
        edgecolor=vline_color,
        labelcolor=vline_color,
        framealpha=0.0
    )

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight",
                   transparent=(background=="black"))
    
    plt.show()
    plt.rcdefaults()


def p_p_plot(
    true_params: np.ndarray,
    inferred_params: np.ndarray,
    fname: str = "plots/pp_plot.png"
): 
    """Create a P-P plot comparing true and inferred parameters.
    
    Args:
        true_params (np.ndarray): True parameter values, shape (num_samples, num_params)
        inferred_params (np.ndarray): Inferred parameter values, shape (num_samples, num_params)
        fname (str): Filename to save plot
    """
    # TODO: Implement P-P plot
    pass


def create_signal_grid_gif(
    dataset,
    num_frames: int = 20,
    num_signals_per_frame: int = 8,
    num_cols: int = 4,
    num_rows: int = 2,
    fname: str = "plots/signal_grid_animation.gif",
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    duration: int = 1000,
    seed: Optional[int] = None
) -> None:
    """Create an animated GIF of signal grids with randomly sampled signals.
    
    Args:
        dataset: Dataset object with signals (e.g., CCSNData)
        num_frames (int): Number of frames in the GIF
        num_signals_per_frame (int): Number of signals to display per frame
        num_cols (int): Number of columns in grid
        num_rows (int): Number of rows in grid
        fname (str): Filename to save the GIF
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
        duration (int): Duration of each frame in milliseconds
        seed (Optional[int]): Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    frames = []
    total_signals = len(dataset)
    
    print(f"Creating {num_frames} frames for GIF animation...")
    
    for frame_idx in range(num_frames):
        # Randomly sample signal indices
        signal_indices = np.random.choice(total_signals, size=num_signals_per_frame, replace=False)
        
        # Collect signals
        selected_signals = []
        for idx in signal_indices:
            signal = dataset[idx][0].cpu().numpy().flatten()
            selected_signals.append(signal)
        
        selected_signals = np.array(selected_signals)
        
        # Use plot_signal_grid to create the plot
        # Temporarily disable plt.show() by using non-interactive backend
        plt.ioff()
        fig, _ = plot_signal_grid(
            signals=selected_signals/TEN_KPC,
            noisy_signals=None,
            max_value=dataset.max_strain,
            num_cols=num_cols,
            num_rows=num_rows,
            fname=None,
            background=background,
            generated=False,
            font_family=font_family,
            font_name=font_name
        )
        
        # Save frame to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                   facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(Image.open(buf).copy())  # Copy to avoid buffer issues
        buf.close()
        
        plt.close(fig)
        plt.ion()  # Re-enable interactive mode
        
        if (frame_idx + 1) % 5 == 0:
            print(f"  Generated {frame_idx + 1}/{num_frames} frames")
    
    # Save as GIF
    print(f"Saving GIF to {fname}...")
    frames[0].save(
        fname,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF created successfully with {num_frames} frames!")


def plot_sky_localisation(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    true_ra: Optional[float] = None,
    true_dec: Optional[float] = None,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    figsize: tuple[float, float] = (30, 18)
) -> plt.Figure:
    """Plot sky location distribution from RA and Dec samples.
    
    Args:
        ra_samples (np.ndarray): Right Ascension samples in radians
        dec_samples (np.ndarray): Declination samples in radians
        true_ra (Optional[float]): True Right Ascension in radians
        true_dec (Optional[float]): True Declination in radians
        fname (Optional[str]): Filename to save the plot
        background (str): Background color ("white" or "black")
        font_family (str): Font family for labels
        font_name (str): Specific font name
        
    Returns:
        plt.Figure: The matplotlib figure object
    """
    # Set up colors based on background
    if _is_dark_color(background):
        text_color = "white"
        grid_color = "black"
        grid_alpha = 0.5
    else:
        text_color = "black"
        grid_color = "black"
        grid_alpha = 0.5
    
    # Create figure with robust projection fallback.
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    fig = plt.figure(figsize=figsize)
    try:
        ax = plt.axes(projection='geo aitoff')
    except Exception:
        try:
            ax = plt.axes(projection='aitoff')
        except Exception:
            # Last-resort fallback to regular Cartesian axes.
            ax = plt.axes()
    
    # Set background color
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)
    
    # Make the plot outline solid white
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2)
        spine.set_linestyle('-')
    
    # Add grid with dotted lines
    ax.grid(linestyle=':', linewidth=0.8)
    
    # Plot the samples as a contour/density plot
    from scipy.stats import gaussian_kde
    
    # Convert samples to the correct coordinate system for plotting
    ra_plot = ra_samples
    dec_plot = dec_samples
    
    # Print sample statistics for debugging
    print(f"RA range: [{np.min(ra_plot):.3f}, {np.max(ra_plot):.3f}] rad")
    print(f"Dec range: [{np.min(dec_plot):.3f}, {np.max(dec_plot):.3f}] rad")
    print(f"Number of samples: {len(ra_plot)}")
    
    # Create density estimate
    try:
        kde = gaussian_kde(np.vstack([ra_plot, dec_plot]))
        
        # Create grid for contour plot
        ra_grid = np.linspace(-np.pi, np.pi, 200)
        dec_grid = np.linspace(-np.pi/2, np.pi/2, 100)
        ra_mesh, dec_mesh = np.meshgrid(ra_grid, dec_grid)
        positions = np.vstack([ra_mesh.ravel(), dec_mesh.ravel()])
        
        # Evaluate KDE on grid
        density = kde(positions).reshape(ra_mesh.shape)
        
        # Plot filled contours for 68%, 95%, 99.7% credible regions
        # Calculate levels corresponding to these percentiles
        sorted_density = np.sort(density.ravel())[::-1]
        cumsum = np.cumsum(sorted_density)
        cumsum /= cumsum[-1]
        
        level_68 = sorted_density[np.argmin(np.abs(cumsum - 0.68))]
        level_95 = sorted_density[np.argmin(np.abs(cumsum - 0.95))]
        level_997 = sorted_density[np.argmin(np.abs(cumsum - 0.997))]
        
        # Use brighter colors for better visibility
        contour_color = '#FF6B6B' if background == "white" else '#FF4444'
        
        # Plot filled contours - need 3 colors/alphas for 4 levels (creates 3 regions)
        contours = ax.contourf(ra_mesh, dec_mesh, density, 
                              levels=[level_997, level_95, level_68, density.max()],
                              colors=[contour_color, contour_color, contour_color],
                              alpha=[0.3, 0.5, 0.7],
                              extend='neither')
        
        # Add contour lines with higher visibility
        line_color = 'black' if background == "white" else 'white'
        ax.contour(ra_mesh, dec_mesh, density,
                  levels=[level_68, level_95, level_997],
                  colors=line_color, linewidths=2, alpha=0.9)
        
    except Exception as e:
        print(f"KDE failed: {e}")
        # If KDE fails, just plot scatter
        scatter_color = '#FF6B6B' if background == "white" else '#FF4444'
        ax.scatter(ra_plot, dec_plot, c=scatter_color, s=5, alpha=0.5, edgecolors='none')
    
    # Plot median position as a star
    ra_median = np.median(ra_samples)
    dec_median = np.median(dec_samples)
    star_color = '#FF6B6B' if background == "white" else '#FF4444'
    star_edge = 'black' if background == "white" else 'white'
    ax.plot(ra_median, dec_median, marker='*', markersize=30,
            color=star_color, markeredgecolor=star_edge,
            markeredgewidth=2, zorder=5)
    print(f"Median position: RA={ra_median:.3f} rad, Dec={dec_median:.3f} rad")

    # Plot true location if provided.
    if true_ra is not None and true_dec is not None:
        true_color = '#00BCD4' if background == "white" else '#00E5FF'
        ax.plot(
            float(true_ra),
            float(true_dec),
            marker='x',
            markersize=14,
            color=true_color,
            markeredgewidth=3,
            zorder=6,
        )
    
    # Add detector locations for reference
    detector_coords = [
        ("LIGO Hanford", np.deg2rad(240), np.deg2rad(46.5)),
        ("LIGO Livingston", np.deg2rad(268), np.deg2rad(30.5)),
        ("Virgo", np.deg2rad(10), np.deg2rad(43.6))
    ]
    
    for name, ra_det, dec_det in detector_coords:
        # Convert to -pi to pi range
        ra_det_plot = ra_det - np.pi
        ax.plot(ra_det_plot, dec_det, marker='v', markersize=8,
                color='#FFD93D', markeredgecolor=text_color,
                markeredgewidth=0.5, zorder=4)
    
    # Set tick colors
    ax.tick_params(colors=text_color)
    
    plt.tight_layout()
    
    if fname:
        plt.savefig(fname, dpi=300, facecolor=background,
                   edgecolor='none', bbox_inches='tight')
        print(f"Saved sky localization plot to {fname}")
    
    plt.show()
    return fig
