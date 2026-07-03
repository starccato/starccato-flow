"""Analysis and visualization functions for model evaluation."""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker
from matplotlib import rcParams
import numpy as np
import pandas as pd
import torch
import corner
from PIL import Image
import io

from ..utils.defaults import TEN_KPC
from ..utils.plotting_defaults import (
    SIGNAL_COLOUR,
    GENERATED_SIGNAL_COLOUR,
    SIGNAL_LIM_UPPER,
    SIGNAL_LIM_LOWER,
    PARAMETER_LABELS,
    PARAMETER_RANGES
)
from . import set_plot_style, get_time_axis
from .signals import plot_signal_grid, plot_candidate_signal

def plot_surface_density(fname=None, font_family=None, font_name=None, transparent=False):
    """Plot surface density of supernovae in the galactic plane."""

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = 18
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
    _, ax = plt.subplots(figsize=(8, 6), facecolor="white")

    ax.plot(r, surface_density, color="lightblue", linewidth=2)

    # Put ticks only on visible axes
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Turn off grid
    ax.grid(False)

    ax.set_xlabel("r (kpc)")
    ax.set_ylabel("Surface Density")

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
    fname_3d: Optional[str] = None,
    fname_xy: Optional[str] = None,
    fname_xz: Optional[str] = None,
    fname_xy_closeup: Optional[str] = "plots/galactic_supernovae_xy_closeup.png",
    fname_yx_zx: Optional[str] = "plots/galactic_supernovae_yx_zx.png",
    background: str = "white",
    transparent: Optional[bool] = None,
    light_year: bool = False,
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    scatter_size: float = 0.001,
    sun_marker_size: float = 100,
    show: bool = False,
    dpi: int = 300,
    legend_frameon: bool = False,
    figsize: tuple = (16, 16),
) -> List[plt.Figure]:
    """Plot galactic supernova locations in 3D, X-Y, and X-Z views.

    Args:
        galactic_coords (np.ndarray): Cartesian galactic coordinates with shape (N, 3)
        sun_location (Optional[np.ndarray]): Sun position in galactic coordinates
        highlight_indices (Optional[np.ndarray]): Optional indices of supernovae to
            draw as highlighted yellow points.
        fname_3d (Optional[str]): Output path for the 3D plot
        fname_xy (Optional[str]): Output path for the X-Y projection plot
        fname_xz (Optional[str]): Output path for the X-Z projection plot
        fname_xy_closeup (Optional[str]): Output path for the X-Y closeup projection plot
        fname_yx_zx (Optional[str]): Output path for the stacked Y-X (top) and Z-X (bottom) plot
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
        List[plt.Figure]: The created matplotlib figures in [3D, X-Y, X-Z] order
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
    if highlight_coords is not None:
        highlight_coords = highlight_coords * coord_scale

    x, y, z = galactic_coords.T
    # xy_radius = max(np.max(np.abs(x)), np.max(np.abs(y)), abs(sun_location[0]), abs(sun_location[1]))
    # print(xy_radius)
    xy_radius = 33
    xz_radius = max(np.max(np.abs(x)), np.max(np.abs(z)), abs(sun_location[0]), abs(sun_location[2]))
    xy_radius *= 1.02
    xz_radius *= 1.02
    if highlight_coords is not None and highlight_coords.size > 0:
        hx, hy, hz = highlight_coords.T
    else:
        hx = hy = hz = None
    text_color = "white" if background == "black" else "black"
    legend_facecolor = "black" if background == "black" else "white"
    grid_color = "gray"
    if transparent is None:
        transparent = background == "black"
    facecolor = "none" if transparent else background

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = 18
    if font_family == "sans-serif":
        plt.rcParams["font.sans-serif"] = [font_name]
    elif font_family == "serif":
        plt.rcParams["font.serif"] = [font_name]

    def _prepare_output_path(path: Optional[str]) -> Optional[Path]:
        if path is None:
            return None
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path

    def _style_2d_axes(axes: plt.Axes) -> None:
        axes.tick_params(colors=text_color, labelsize=18, direction="inout", length=12, width=1.4)
        for spine in axes.spines.values():
            spine.set_color(text_color)
        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        # axes.set_aspect("equal")
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
        adjusted_handles = []
        for handle, label in zip(handles, labels):
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
            elif label == "Sampled Supernovae":
                adjusted_handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        linestyle="None",
                        marker="o",
                        markersize=9,
                        markerfacecolor="yellow",
                        markeredgecolor="none",
                    )
                )
            else:
                adjusted_handles.append(handle)

        axes.legend(
            adjusted_handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=max(1, len(labels)),
            facecolor=legend_facecolor,
            edgecolor="none",
            labelcolor=text_color,
            fontsize=20,
            frameon=legend_frameon,
        )

    output_3d = _prepare_output_path(fname_3d)
    output_xy = _prepare_output_path(fname_xy)
    output_xz = _prepare_output_path(fname_xz)
    # output_xy_closeup = _prepare_output_path(fname_xy_closeup)
    # output_yx_zx = _prepare_output_path(fname_yx_zx)

    figures: List[plt.Figure] = []

    fig1 = plt.figure(figsize=figsize, facecolor=facecolor)
    ax1 = fig1.add_subplot(111, projection="3d", facecolor=facecolor)
    ax1.scatter(x, y, z, s=scatter_size, alpha=1, c="lightblue", label="Supernova")
    ax1.scatter(
        0.0,
        0.0,
        0.0,
        s=sun_marker_size,
        c="black",
        edgecolors="white",
        linewidths=1.8,
        marker="o",
        label="Galactic Center: Sgr A*",
    )
    ax1.scatter(
        sun_location[0],
        sun_location[1],
        sun_location[2],
        s=sun_marker_size,
        c="yellow",
        marker="*",
        label="Sun",
    )
    if hx is not None:
        ax1.scatter(
            hx,
            hy,
            hz,
            s=max(sun_marker_size * 0.1, 10),
            c="yellow",
            edgecolors="none",
            marker="o",
            label="Sampled Supernovae",
            zorder=10,
        )
    ax1.set_xlabel(_axis_label("X"), color=text_color, fontsize=22)
    ax1.set_ylabel(_axis_label("Y"), color=text_color, fontsize=22)
    ax1.set_zlabel(_axis_label("Z"), color=text_color, fontsize=22)
    ax1.tick_params(colors=text_color, labelsize=18)
    ax1.set_aspect("equal")
    ax1.set_zticks([])
    ax1.xaxis.pane.set_facecolor("none")
    ax1.xaxis.pane.set_alpha(0)
    ax1.yaxis.pane.set_facecolor("none")
    ax1.yaxis.pane.set_alpha(0)
    ax1.zaxis.pane.set_facecolor("none")
    ax1.zaxis.pane.set_alpha(0)
    ax1.xaxis.pane.set_edgecolor(text_color)
    ax1.yaxis.pane.set_edgecolor(text_color)
    ax1.zaxis.pane.set_edgecolor(text_color)
    ax1.grid(color=grid_color, alpha=0.2)
    ax1.zaxis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax1.set_xlim(-xy_radius, xy_radius)
    ax1.set_ylim(-xy_radius, xy_radius)
    z_max = max(abs(z.min()), abs(z.max()))
    ax1.set_zlim(-z_max, z_max)
    if light_year:
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(20_000))
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(20_000))
        ax1.set_zlim(-20_000, 20_000)
        ax1.zaxis.set_major_locator(mticker.MultipleLocator(10_000))
        ax1.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda val, pos: _light_year_tick_label(val))
        )
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda val, pos: _light_year_tick_label(val))
        )
        ax1.zaxis.set_major_formatter(
            mticker.FuncFormatter(lambda val, pos: _light_year_tick_label(val))
        )
        _tighten_light_year_tick_lines(ax1)
        for tick_label in ax1.get_zticklabels():
            tick_label.set_linespacing(0.75)
    else:
        # Set nice round tick values for kpc
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(10))
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(10))
        ax1.set_zlim(-10, 10)
        ax1.zaxis.set_major_locator(mticker.MultipleLocator(5))
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
        ax1.zaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
    _legend_with_supernova_marker(ax1)
    if output_3d is not None:
        fig1.savefig(output_3d, dpi=dpi, bbox_inches="tight", transparent=transparent)
    figures.append(fig1)

    fig2 = plt.figure(figsize=figsize, facecolor=facecolor)
    ax2 = fig2.add_subplot(111, facecolor=facecolor)
    ax2.scatter(x, y, s=scatter_size, alpha=1, c="lightblue", label="Supernova")
    ax2.scatter(
        0.0,
        0.0,
        s=sun_marker_size,
        c="black",
        edgecolors="white",
        linewidths=1.8,
        marker="o",
        label="Galactic Center: Sgr A*",
    )
    ax2.scatter(sun_location[0], sun_location[1], s=sun_marker_size, c="yellow", marker="*", label="Sun")
    if hx is not None:
        ax2.scatter(
            hx,
            hy,
            s=max(sun_marker_size * 0.1, 10),
            c="yellow",
            edgecolors="none",
            marker="o",
            label="Sampled Supernovae",
            zorder=10,
        )
    ax2.set_xlabel(_axis_label("X"), color=text_color, fontsize=22)
    ax2.set_ylabel(_axis_label("Y"), color=text_color, fontsize=22)
    # ax2.set_title(
    #     "Simulated Galactic Supernova Distribution in X-Y Plane",
    #     color=text_color,
    #     fontsize=24,
    #     pad=20,
    #     fontweight="bold",
    # )
    _style_2d_axes(ax2)
    ax2.set_xlim(-xy_radius, xy_radius)
    ax2.set_ylim(-xy_radius, xy_radius)
    if light_year:
        tick_values = np.arange(-80_000, 80_001, 20_000)
        axis_padding = 5_000
        ax2.set_xlim(tick_values[0] - axis_padding, tick_values[-1] + axis_padding)
        ax2.set_ylim(tick_values[0] - axis_padding, tick_values[-1] + axis_padding)
        ax2.set_xticks(tick_values)
        ax2.set_yticks(tick_values)
        _tighten_light_year_tick_lines(ax2)
    else:
        # Set kpc limits to match light-year equivalent (-85000 to 85000 ly = -26.07 to 26.07 kpc)
        kpc_limit = 26.07
        kpc_padding = 2.07
        tick_values = np.arange(-25, 26, 5)
        ax2.set_xlim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
        ax2.set_ylim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
        ax2.set_xticks(tick_values)
        ax2.set_yticks(tick_values)
    _apply_xy_axis_line_window(ax2)
    _legend_with_supernova_marker(ax2)
    if output_xy is not None:
        fig2.savefig(output_xy, dpi=dpi, bbox_inches="tight", transparent=transparent)
    figures.append(fig2)

    fig3 = plt.figure(figsize=figsize, facecolor=facecolor)
    ax3 = fig3.add_subplot(111, facecolor=facecolor)
    ax3.scatter(x, z, s=scatter_size, alpha=1, c="lightblue", label="Supernova")
    ax3.scatter(
        0.0,
        0.0,
        s=sun_marker_size,
        c="black",
        edgecolors="white",
        linewidths=1.8,
        marker="o",
        label="Galactic Center: Sgr A*",
    )
    ax3.scatter(sun_location[0], sun_location[2], s=sun_marker_size, c="yellow", marker="*", label="Sun")
    if hx is not None:
        ax3.scatter(
            hx,
            hz,
            s=max(sun_marker_size * 0.1, 18),
            c="yellow",
            edgecolors="none",
            marker="o",
            label="Sampled Supernovae",
            zorder=10,
        )
    ax3.set_xlabel(_axis_label("X"), color=text_color, fontsize=22)
    ax3.set_ylabel(_axis_label("Z"), color=text_color, fontsize=22)
    _style_2d_axes(ax3)
    ax3.set_xlim(-xz_radius, xz_radius)
    ax3.set_ylim(-xz_radius, xz_radius)
    if light_year:
        ax3.set_ylim(-20_000, 20_000)
        ax3.yaxis.set_major_locator(mticker.MultipleLocator(10_000))
        ax3.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda val, pos: _light_year_tick_label(val))
        )
    else:
        # Set nice round tick values for kpc on Z axis
        ax3.set_ylim(-10, 10)
        ax3.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
    _legend_with_supernova_marker(ax3)
    if output_xz is not None:
        fig3.savefig(output_xz, dpi=dpi, bbox_inches="tight", transparent=transparent)
    figures.append(fig3)

    # fig4 = plt.figure(figsize=(16, 16), facecolor=facecolor)
    # ax4 = fig4.add_subplot(111, facecolor=facecolor)
    # ax4.scatter(x, y, s=scatter_size, alpha=1, c="lightblue", label="Supernova")
    # ax4.scatter(
    #     0.0,
    #     0.0,
    #     s=sun_marker_size,
    #     c="black",
    #     edgecolors="white",
    #     linewidths=1.8,
    #     marker="o",
    #     label="Galactic Center: Sgr A*",
    # )
    # ax4.scatter(sun_location[0], sun_location[1], s=sun_marker_size, c="yellow", marker="*", label="Sun")
    # ax4.set_xlabel(_axis_label("X"), color=text_color, fontsize=22)
    # ax4.set_ylabel(_axis_label("Y"), color=text_color, fontsize=22)
    # _style_2d_axes(ax4)

    # # Closeup bounds requested by user in light-years.
    # closeup_x_ly = (-60_000.0, 60_000.0)
    # closeup_y_ly = (-10_000.0, 80_000.0)
    # closeup_padding_ly = 5_000.0
    # if light_year:
    #     ax4.set_xlim(closeup_x_ly[0] - closeup_padding_ly, closeup_x_ly[1] + closeup_padding_ly)
    #     ax4.set_ylim(closeup_y_ly[0] - closeup_padding_ly, closeup_y_ly[1] + closeup_padding_ly)
    #     _tighten_light_year_tick_lines(ax4)
    # else:
    #     ly_to_kpc = 1.0 / 3261.56
    #     padding_kpc = closeup_padding_ly * ly_to_kpc
    #     ax4.set_xlim(closeup_x_ly[0] * ly_to_kpc - padding_kpc, closeup_x_ly[1] * ly_to_kpc + padding_kpc)
    #     ax4.set_ylim(closeup_y_ly[0] * ly_to_kpc - padding_kpc, closeup_y_ly[1] * ly_to_kpc + padding_kpc)
    # _apply_xy_axis_line_window(ax4)

    # _legend_with_supernova_marker(ax4)
    # if output_xy_closeup is not None:
    #     fig4.savefig(output_xy_closeup, dpi=dpi, bbox_inches="tight", transparent=transparent)
    # figures.append(fig4)

    # fig5, (ax5_top, ax5_bottom) = plt.subplots(
    #     2,
    #     1,
    #     sharex=True,
    #     figsize=(16, 20),
    #     facecolor=facecolor,
    # )
    # ax5_top.set_facecolor(facecolor)
    # ax5_bottom.set_facecolor(facecolor)

    # ax5_top.scatter(x, y, s=scatter_size, alpha=1, c="lightblue", label="Supernova")
    # ax5_top.scatter(
    #     0.0,
    #     0.0,
    #     s=sun_marker_size,
    #     c="black",
    #     edgecolors="white",
    #     linewidths=1.8,
    #     marker="o",
    #     label="Galactic Center: Sgr A*",
    # )
    # ax5_top.scatter(sun_location[0], sun_location[1], s=sun_marker_size, c="yellow", marker="*", label="Sun")
    # ax5_top.set_ylabel(_axis_label("Y"), color=text_color, fontsize=22)
    # _style_2d_axes(ax5_top)

    # if light_year:
    #     tick_values = np.arange(-80_000, 80_001, 20_000)
    #     axis_padding = 5_000
    #     x_min = tick_values[0] - axis_padding
    #     x_max = tick_values[-1] + axis_padding
    #     y_min = tick_values[0] - axis_padding
    #     y_max = tick_values[-1] + axis_padding
    #     ax5_top.set_xlim(x_min, x_max)
    #     ax5_top.set_ylim(y_min, y_max)
    #     ax5_top.set_xticks(tick_values)
    #     ax5_top.set_yticks(tick_values)
    #     _tighten_light_year_tick_lines(ax5_top)
    # else:
    #     ax5_top.set_xlim(-xy_radius, xy_radius)
    #     ax5_top.set_ylim(-xy_radius, xy_radius)
    # _apply_xy_axis_line_window(ax5_top)
    # _legend_with_supernova_marker(ax5_top)

    # ax5_bottom.scatter(x, z, s=scatter_size, alpha=1, c="lightblue", label="Supernova")
    # ax5_bottom.scatter(
    #     0.0,
    #     0.0,
    #     s=sun_marker_size,
    #     c="black",
    #     edgecolors="white",
    #     linewidths=1.8,
    #     marker="o",
    #     label="Galactic Center: Sgr A*",
    # )
    # ax5_bottom.scatter(sun_location[0], sun_location[2], s=sun_marker_size, c="yellow", marker="*", label="Sun")
    # ax5_bottom.set_xlabel(_axis_label("X"), color=text_color, fontsize=22)
    # ax5_bottom.set_ylabel(_axis_label("Z"), color=text_color, fontsize=22)
    # _style_2d_axes(ax5_bottom)

    # if light_year:
    #     ax5_bottom.set_xlim(x_min, x_max)
    #     ax5_bottom.set_xticks(tick_values)
    #     ax5_bottom.set_ylim(-10_000, 10_000)
    #     ax5_bottom.yaxis.set_major_locator(mticker.MultipleLocator(5_000))
    #     ax5_bottom.yaxis.set_major_formatter(
    #         mticker.FuncFormatter(lambda val, pos: _light_year_tick_label(val))
    #     )
    #     _tighten_light_year_tick_lines(ax5_bottom)
    # else:
    #     ly_to_kpc = 1.0 / 3261.56
    #     ax5_bottom.set_xlim(-xy_radius, xy_radius)
    #     ax5_bottom.set_ylim(-10_000 * ly_to_kpc, 10_000 * ly_to_kpc)

    # fig5.subplots_adjust(hspace=0.08, top=0.92)
    # if output_yx_zx is not None:
    #     fig5.savefig(output_yx_zx, dpi=dpi, bbox_inches="tight", transparent=transparent)
    # figures.append(fig5)

    if show:
        plt.show()
    else:
        for figure in figures:
            plt.close(figure)

    plt.rcdefaults()
    return figures


def plot_galactic_distribution_with_posterior(
    galactic_coords: np.ndarray,
    posterior_ra: np.ndarray,
    posterior_dec: np.ndarray,
    posterior_distance: np.ndarray,
    true_ra: Optional[float] = None,
    true_dec: Optional[float] = None,
    true_distance: Optional[float] = None,
    sun_location: Optional[np.ndarray] = None,
    fname: Optional[str] = None,
    background: str = "white",
    transparent: Optional[bool] = None,
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    scatter_size: float = 0.001,
    sun_marker_size: float = 100,
    show: bool = False,
    dpi: int = 300,
    figsize: tuple = (12, 12),
) -> plt.Figure:
    """Plot galactic supernova distribution in X-Y plane with posterior credible regions overlaid.

    Args:
        galactic_coords (np.ndarray): Cartesian galactic coordinates with shape (N, 3)
        posterior_ra (np.ndarray): Posterior RA samples in radians
        posterior_dec (np.ndarray): Posterior Dec samples in radians
        posterior_distance (np.ndarray): Posterior distance samples in kpc
        true_ra (Optional[float]): True RA in radians
        true_dec (Optional[float]): True Dec in radians
        true_distance (Optional[float]): True distance in kpc
        sun_location (Optional[np.ndarray]): Sun position in galactic coordinates
        fname (Optional[str]): Output path for the plot
        background (str): Plot theme, either "white" or "black"
        transparent (Optional[bool]): Override the saved figure transparency
        font_family (str): Font family to use
        font_name (str): Specific font name to use
        scatter_size (float): Marker size for background supernova points
        sun_marker_size (float): Marker size for the sun marker
        show (bool): Whether to keep figure open and display it
        dpi (int): DPI used when saving output files
        figsize (tuple): Figure size in inches as (width, height)

    Returns:
        plt.Figure: The created matplotlib figure
    """
    from ..supernovae.supernovae import Supernovae
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgba

    galactic_coords = np.asarray(galactic_coords)
    if galactic_coords.ndim != 2 or galactic_coords.shape[1] != 3:
        raise ValueError("galactic_coords must have shape (N, 3).")

    if sun_location is None:
        sun_location = np.array([0.0, 8.178, 0.0208], dtype=float)
    else:
        sun_location = np.asarray(sun_location, dtype=float)
        if sun_location.shape != (3,):
            raise ValueError("sun_location must have shape (3,).")

    # Set up plot styling
    rcParams["font.family"] = font_family
    rcParams["font.size"] = 18
    if font_family == "sans-serif":
        rcParams["font.sans-serif"] = [font_name]
    elif font_family == "serif":
        rcParams["font.serif"] = [font_name]

    facecolor = background if background in ("white", "black") else "white"
    text_color = "white" if background == "black" else "black"
    grid_color = "gray" if background == "black" else "lightgray"
    transparent = transparent if transparent is not None else (background == "black")
    plot_facecolor = "none" if transparent else background

    # Extract X-Y coordinates from background galactic distribution
    x = galactic_coords[:, 0]
    y = galactic_coords[:, 1]

    # Transform posterior samples to galactic coordinates
    sn_temp = Supernovae()  # Temporary instance for coordinate transformation
    post_x, post_y, post_z = sn_temp.equatorial_to_galactic(
        posterior_ra, posterior_dec, posterior_distance
    )
    
    # Convert posterior from heliocentric to galactocentric frame by adding sun location
    post_x += sun_location[0]
    post_y += sun_location[1]
    post_z += sun_location[2]

    # Create figure with proper styling (matching plot_galactic_distribution)
    fig = plt.figure(figsize=figsize, facecolor=plot_facecolor)
    ax = fig.add_subplot(111, facecolor=facecolor)

    # Plot background galactic distribution (exactly as in plot_galactic_distribution)
    ax.scatter(x, y, s=scatter_size, alpha=1, c="lightblue", label="Supernova")
    ax.scatter(
        0.0,
        0.0,
        s=sun_marker_size,
        c="black",
        edgecolors="white",
        linewidths=1.8,
        marker="o",
        label="Galactic Center: Sgr A*",
    )
    ax.scatter(sun_location[0], sun_location[1], s=sun_marker_size, c="yellow", marker="*", label="Sun", zorder=20)

    # Add density contours from posterior samples in X-Y plane (ONLY DIFFERENCE: add this layer)
    from scipy.stats import gaussian_kde
    from matplotlib.colors import to_rgba
    
    # Build KDE from posterior X-Y coordinates for credible contours
    xy_data = np.vstack([post_x, post_y])
    try:
        kde = gaussian_kde(xy_data)
        
        # Create grid for evaluating KDE
        x_min, x_max = post_x.min(), post_x.max()
        y_min, y_max = post_y.min(), post_y.max()
        x_grid = np.linspace(x_min - 2, x_max + 2, 200)
        y_grid = np.linspace(y_min - 2, y_max + 2, 200)
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
        
        # Red fill colors matching celestial map (red with varying alphas: 0.40, 0.62, 0.88)
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

    # Plot true location if provided (matching celestial map marker: deepskyblue "x")
    if true_ra is not None and true_dec is not None and true_distance is not None:
        true_x, true_y, true_z = sn_temp.equatorial_to_galactic(
            np.array([true_ra]), np.array([true_dec]), np.array([true_distance])
        )
        # Convert to galactocentric frame by adding sun location
        true_x += sun_location[0]
        true_y += sun_location[1]
        true_z += sun_location[2]
        # Plot with same marker style as celestial map (deepskyblue "x")
        ax.scatter(
            true_x,
            true_y,
            s=72,
            marker="x",
            c="deepskyblue",
            linewidths=1.8,
            zorder=10,
            label="True Location",
        )

    # Style axes exactly like plot_galactic_distribution
    ax.set_xlabel("X (kpc)", color=text_color, fontsize=22)
    ax.set_ylabel("Y (kpc)", color=text_color, fontsize=22)
    
    # _style_2d_axes equivalent
    ax.tick_params(colors=text_color, labelsize=18, direction="inout", length=12, width=1.4)
    for spine in ax.spines.values():
        spine.set_color(text_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, pos: f"{val:.0f}"))
    
    # Set limits and ticks
    xy_radius = 33
    kpc_limit = 26.07
    kpc_padding = 2.07
    tick_values = np.arange(-25, 26, 5)
    ax.set_xlim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
    ax.set_ylim(-kpc_limit - kpc_padding, kpc_limit + kpc_padding)
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)
    
    # _apply_xy_axis_line_window equivalent
    ax.spines["bottom"].set_bounds(-25, 25)
    ax.spines["left"].set_bounds(-25, 25)
    
    ax.set_aspect("equal")
    ax.grid(color=grid_color, alpha=0.2)
    
    # Add legend (matching plot_galactic_distribution style)
    legend_facecolor = "black" if background == "black" else "white"
    handles, labels = ax.get_legend_handles_labels()
    adjusted_handles = []
    for handle, label in zip(handles, labels):
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
        else:
            adjusted_handles.append(handle)

    legend = ax.legend(
        adjusted_handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=max(1, len(labels)),
        facecolor=legend_facecolor,
        edgecolor="none",
        framealpha=0.0,
        fontsize=14,
        labelcolor=text_color,
    )

    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_inches="tight", transparent=transparent)

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
    font_name: str = "Times New Roman"
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
    vline_color = "white" if background == "black" else "black"

    # Prepare data
    reconstructed_signals = np.array(reconstructed_signals)
    true_signal_np = true_signal.squeeze().cpu().numpy() * max_value
    noisy_signal_np = noisy_signal.squeeze().cpu().numpy() * max_value
    reconstructed_signals_df = pd.DataFrame(reconstructed_signals.T)
    d = get_time_axis()

    # Create figure
    fig = plt.figure(figsize=(8, 6))
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


def create_snr_variation_gif(
    dataset,
    signal_index: int = 0,
    snr_start: int = 200,
    snr_end: int = 10,
    num_frames: int = 20,
    fname: str = "plots/snr_variation.gif",
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    duration: int = 500
) -> None:
    """Create an animated GIF showing how a signal changes with varying SNR.
    
    Args:
        dataset: Dataset object (e.g., CCSNData) with calculate_snr and aLIGO_noise methods
        signal_index (int): Index of the signal to use from the dataset
        snr_start (int): Starting SNR value (higher, less noise)
        snr_end (int): Ending SNR value (lower, more noise)
        num_frames (int): Number of frames in the animation
        fname (str): Filename to save the GIF
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
        duration (int): Duration of each frame in milliseconds
    """
    print(f"Creating SNR variation GIF from SNR={snr_start} to SNR={snr_end}...")
    
    # Get the clean signal
    clean_signal = dataset.signals[:, signal_index].reshape(1, -1)
    
    # Calculate SNR range
    snr_values = np.linspace(snr_start, snr_end, num_frames)
    
    frames = []
    
    # Import required utilities
    from ..utils.defaults import SAMPLING_FREQ, Y_LENGTH
    
    is_even = (Y_LENGTH % 2 == 0)
    half_N = Y_LENGTH // 2 if is_even else (Y_LENGTH - 1) // 2
    delta_f = 1 / (Y_LENGTH * SAMPLING_FREQ)
    fourier_freq = np.arange(half_N + 1) * delta_f
    
    Sn = dataset.AdvLIGOPsd(fourier_freq)
    
    # Turn off interactive plotting to avoid showing intermediate plots
    plt.ioff()
    
    for frame_idx, target_snr in enumerate(snr_values):
        # Scale signal properly
        s = clean_signal / 3.086e+22
        s_array = np.asarray(s).flatten()
        rho = dataset.calculate_snr(s_array, Sn)
        
        # Generate noise
        n = dataset.aLIGO_noise(seed_offset=frame_idx)
        
        # Add noise with target SNR
        d_noisy = s + n * (rho / target_snr) * 100
        
        # Scale back
        s_scaled = s * 3.086e+22
        d_noisy_scaled = d_noisy * 3.086e+22
        
        # Normalize
        s_normalized = s_scaled / dataset.max_strain
        d_noisy_normalized = d_noisy_scaled / dataset.max_strain
        
        # Use plot_candidate_signal to create the frame
        fig = plot_candidate_signal(
            signal=s_normalized/TEN_KPC,
            noisy_signal=d_noisy_normalized/TEN_KPC,
            max_value=dataset.max_strain,
            fname=None,
            generated=False,
            background=background,
            font_family=font_family,
            font_name=font_name
        )
        
        # Add SNR text annotation to the figure
        ax = fig.gca()
        text_color = "white" if background == "black" else "black"
        ax.text(0.98, 0.98, f'SNR = {target_snr:.1f}',
                transform=ax.transAxes,
                fontsize=16, color=text_color,
                verticalalignment='top',
                horizontalalignment='right')
        
        # Save frame to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                   facecolor=fig.get_facecolor())
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        
        plt.close(fig)
        
        if (frame_idx + 1) % 5 == 0:
            print(f"  Generated {frame_idx + 1}/{num_frames} frames")
    
    # Re-enable interactive plotting
    plt.ion()
    
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
    font_name: str = "Times New Roman"
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
    if background == "black":
        text_color = "white"
        grid_color = "black"
        grid_alpha = 0.5
    else:
        text_color = "black"
        grid_color = "black"
        grid_alpha = 0.5
    
    # Create figure with robust projection fallback.
    fig = plt.figure(figsize=(12, 7))
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
