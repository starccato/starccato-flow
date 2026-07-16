"""Signal plotting functions for waveform visualization."""

from typing import Optional, Tuple, Sequence
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import torch
from . import set_plot_style, get_time_axis
from ..utils.defaults_plotting import (
    SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR, DEFAULT_FONT_FAMILY, 
    DEFAULT_FONT, SIGNAL_LIM_UPPER, SIGNAL_LIM_LOWER
)


def _map_detector_labels(labels: Sequence[str]) -> Tuple[str, ...]:
    """Map detector short codes to full names for display.
    
    Args:
        labels: Sequence of detector labels (e.g., ["H1", "L1", "V1"])
        
    Returns:
        Tuple of full detector names for display
    """
    detector_map = {
        "H1": "LIGO Hanford",
        "L1": "LIGO Livingston",
        "V1": "Virgo",
    }
    return tuple(detector_map.get(label, label) for label in labels)


def plot_signal_grid(
    signals: np.ndarray,
    noisy_signals: np.ndarray,
    max_value: int,
    num_cols: int = 2,
    num_rows: int = 4,
    fname: Optional[str] = None,
    generated: bool = False,
    background: str = "white",
    font_family: str = DEFAULT_FONT_FAMILY,
    font_name: str = DEFAULT_FONT,
    param_values: Optional[np.ndarray] = None,
    param_label: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a grid of waveform signals.
    
    Args:
        signals (np.ndarray): Array of signals to plot
        noisy_signals (np.ndarray): Noisy version of signals (currently unused)
        num_cols (int): Number of columns in grid
        num_rows (int): Number of rows in grid
        fname (Optional[str]): Filename to save plot
        generated (bool): Whether signals are generated (affects color)
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
        param_values (Optional[np.ndarray]): Array of parameter values to display above each plot
        param_label (Optional[str]): Label for the parameter (e.g., "Ye")
    
    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes objects
    """
    set_plot_style(background, font_family, font_name)
    
    signal_colour = GENERATED_SIGNAL_COLOUR if generated else SIGNAL_COLOUR
    vline_color = "white" if background == "black" else "black"
    text_color = "white" if background == "black" else "black"

    # Adjust figsize if we have parameter labels
    figsize = (15, 10) if param_values is not None and param_label is not None else (15, 8)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    d = get_time_axis()

    for i, ax in enumerate(axes):
        if i >= len(signals):
            ax.axis('off')
            continue
            
        y = signals[i].flatten()
        y = y * max_value
        ax.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
        ax.set_xlim(min(d), max(d))
        ax.plot(d, y, color=signal_colour)
        
        ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
        ax.grid(False)
        
        # Display parameter value above each subplot if provided
        if param_values is not None and param_label is not None and i < len(param_values):
            param_text = f"{param_label} = {param_values[i]:.3f}"
            ax.set_title(param_text, fontsize=11, color=text_color, pad=8)
        
        if i % num_cols != 0:
            ax.yaxis.set_ticklabels([])
        if i < num_cols * (num_rows - 1):
            ax.xaxis.set_ticklabels([])

    fig.supxlabel('time (s)', fontsize=20)
    fig.supylabel('h', fontsize=20)

    plt.tight_layout()
    if fname:
        if fname.endswith('.svg'):
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(fname, format='svg', transparent=(background=="black"))
        else:
            plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.show()
    plt.rcdefaults()
    return fig, axes

def plot_detector_signal_channels(
    signals: np.ndarray,
    noisy_signals: Optional[np.ndarray] = None,
    max_value: float = 1.0,
    detector_labels: Sequence[str] = ("H1", "L1", "V1"),
    fname: Optional[str] = None,
    generated: bool = False,
    background: str = "black",
    font_family: str = DEFAULT_FONT_FAMILY,
    font_name: str = DEFAULT_FONT,
    transparent: bool = False,
    figsize_mm: Tuple[float, float] = (165, 190),
    fontsize_tick: int = 12,
    fontsize_text: int = 18,
    line_weight: float = 1.4,
    left_margin: float = 0.05,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot 3 detector-channel signals with clean and noisy overlay styling.

    Notes:
        - If noisy_signals is provided, both clean and noisy signals are plotted.
        - Expected signal shape is either ``(3, signal_length)`` or
          ``(signal_length, 3)``.
    """
    set_plot_style(background, font_family, font_name)

    signal_colour = SIGNAL_COLOUR
    vline_color = "white" if background == "black" else "black"

    sig = np.asarray(signals)
    if sig.ndim != 2:
        raise ValueError(f"signals must be 2D, got shape {sig.shape}")

    if sig.shape[0] == 3:
        channel_signals = sig
    elif sig.shape[1] == 3:
        channel_signals = sig.T
    else:
        raise ValueError(
            "signals must have 3 detector channels in axis 0 or axis 1; "
            f"got shape {sig.shape}"
        )

    # Handle noisy signals if provided
    if noisy_signals is not None:
        noisy_sig = np.asarray(noisy_signals)
        if noisy_sig.ndim != 2:
            raise ValueError(f"noisy_signals must be 2D, got shape {noisy_sig.shape}")
        
        if noisy_sig.shape[0] == 3:
            channel_noisy = noisy_sig
        elif noisy_sig.shape[1] == 3:
            channel_noisy = noisy_sig.T
        else:
            raise ValueError(
                "noisy_signals must have 3 detector channels in axis 0 or axis 1; "
                f"got shape {noisy_sig.shape}"
            )
    else:
        channel_noisy = None

    # Convert mm to inches (1 inch = 25.4 mm)
    figsize_inches = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
    
    fig = plt.figure(figsize=figsize_inches)
    gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], 
                          hspace=0.4, left=0.15, right=0.95, top=0.88, bottom=0.12)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    # Share x-axis among all subplots
    axes[1].sharex(axes[0])
    axes[2].sharex(axes[0])
    d = get_time_axis()
    x_min, x_max = -0.01, 0.05
    tick_step = 0.01
    xticks = np.arange(x_min, x_max + (0.5 * tick_step), tick_step)
    y_expand = 1.25
    # Use noisy signal for y-limits if available, otherwise use clean signal
    signal_for_ylim = channel_noisy if channel_noisy is not None else channel_signals
    max_absolute_value = np.max(np.abs(signal_for_ylim))
    y_min = -max_absolute_value * y_expand
    y_max = max_absolute_value * y_expand

    # Map detector labels to full names for display
    display_labels = _map_detector_labels(detector_labels)

    for i, ax in enumerate(axes):
        y_clean = channel_signals[i].flatten()
        
        # Plot noisy signal first (if provided) with lower opacity
        if channel_noisy is not None:
            y_noisy = channel_noisy[i].flatten()
            ax.plot(d, y_noisy, color=signal_colour, linewidth=line_weight, alpha=0.5, label="Signal + Noise")
        
        # Plot clean signal on top with full opacity
        ax.plot(d, y_clean, color=signal_colour, linewidth=line_weight, alpha=1.0, label="Signal")
        
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.margins(x=0.0)
        ax.set_xmargin(0)
        ax.autoscale(enable=False, axis='x')

        ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5, linewidth=line_weight)
        ax.grid(False)
        ax.set_title(display_labels[i], fontsize=fontsize_text)
        ax.tick_params(axis='x', colors=vline_color, labelsize=fontsize_tick)
        ax.tick_params(axis='y', colors=vline_color, labelsize=fontsize_tick)

        for spine in ax.spines.values():
            spine.set_color(vline_color)
            spine.set_linewidth(line_weight)

        if i < 2:
            ax.tick_params(axis='x', which='both', labelbottom=False, bottom=False)

    axes[-1].set_xticks(xticks)
    axes[-1].tick_params(axis='x', which='both', labelbottom=True, bottom=True, colors=vline_color)
    axes[-1].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    
    # Set axis labels with proper size and color
    axes[-1].set_xlabel('time (s)', fontsize=fontsize_text, color=vline_color)
    axes[1].set_ylabel('h', fontsize=fontsize_text, color=vline_color)

    # Add legend outside/on top if we have both signals
    # if channel_noisy is not None:
    #     handles, labels = axes[0].get_legend_handles_labels()
    #     fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
    #               facecolor="none", edgecolor=vline_color, labelcolor=vline_color, 
    #               fontsize=12, framealpha=0.0, ncol=2)

    if fname:
        if fname.endswith('.svg'):
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(fname, format='svg', transparent=transparent)
        else:
            plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=transparent)

    plt.show()
    plt.rcdefaults()
    return fig, np.array(axes)


def plot_candidate_signal(
    signal: np.ndarray,
    noisy_signal: np.ndarray,
    max_value: float,
    fname: Optional[str] = None,
    generated: bool = False,
    background: str = "white",
    font_family: str = DEFAULT_FONT_FAMILY,
    font_name: str = DEFAULT_FONT
) -> plt.Figure:
    """Plot clean and noisy signals overlaid with consistent styling.
    
    Args:
        signal (np.ndarray): Clean signal to plot
        noisy_signal (np.ndarray): Noisy signal to plot
        max_value (float): Maximum value for scaling
        fname (Optional[str]): Filename to save plot
        generated (bool): Whether signals are generated (affects color)
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    
    clean_color = SIGNAL_COLOUR
    noisy_color = SIGNAL_COLOUR
    vline_color = "white" if background == "black" else "black"
    text_color = vline_color

    fig = plt.figure(figsize=(6, 6))
    d = get_time_axis()

    if torch.is_tensor(signal):
        y_clean = signal.cpu().numpy().flatten() * max_value
    else:
        y_clean = signal.flatten() * max_value
    
    if torch.is_tensor(noisy_signal):
        y_noisy = noisy_signal.cpu().numpy().flatten() * max_value
    else:
        y_noisy = noisy_signal.flatten() * max_value

    plt.plot(d, y_noisy, color=noisy_color, linewidth=1.5, alpha=0.5, label="Signal + Noise")
    plt.plot(d, y_clean, color=clean_color, linewidth=2, alpha=1.0, label="Signal")
    
    plt.axvline(x=0, color=vline_color, linestyle='--', alpha=0.5)
    plt.ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    plt.xlim(min(d), max(d))
    plt.xlabel('time (s)', size=16, color=text_color)
    plt.ylabel('h', size=16, color=text_color)
    plt.grid(False)
    
    plt.legend(loc='lower right', facecolor="none", edgecolor=text_color, 
               labelcolor=text_color, fontsize=12, framealpha=0.0)

    plt.tight_layout()
    if fname:
        if fname.endswith('.svg'):
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(fname, format='svg', transparent=(background=="black"))
        else:
            plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
        plt.show()
        plt.rcdefaults()
    
    return fig


def plot_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = DEFAULT_FONT_FAMILY,
    font_name: str = DEFAULT_FONT
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot original and reconstructed signals for comparison.
    
    Single-channel version of detector signal plotting with clean and noisy overlay.
    
    Args:
        original (torch.Tensor): Original clean signal
        reconstructed (torch.Tensor): Reconstructed signal (analog to noisy)
        max_value (float): Maximum value for scaling
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes objects
    """
    set_plot_style(background, font_family, font_name)
    vline_color = "white" if background == "black" else "black"

    fig, ax = plt.subplots(figsize=(12, 5))
    d = get_time_axis()
    x_min, x_max = -0.01, 0.05
    tick_step = 0.01
    xticks = np.arange(x_min, x_max + (0.5 * tick_step), tick_step)
    
    y_original = original.flatten() * max_value
    y_reconstructed = reconstructed.flatten() * max_value
    
    # Calculate y limits
    y_expand = 1.5
    max_absolute_value = max(np.abs(y_original).max(), np.abs(y_reconstructed).max())
    y_min = -max_absolute_value * y_expand
    y_max = max_absolute_value * y_expand
    
    # Plot reconstructed signal first (if provided) with lower opacity
    ax.plot(d, y_reconstructed, color=SIGNAL_COLOUR, linewidth=1.5, 
            alpha=0.5, label="Reconstructed")
    
    # Plot original signal on top with full opacity
    ax.plot(d, y_original, color=SIGNAL_COLOUR, linewidth=2, 
            alpha=1.0, label="Original")
    
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.margins(x=0.0)
    ax.set_xmargin(0)
    ax.autoscale(enable=False, axis='x')

    ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    ax.grid(False)
    ax.tick_params(axis='x', colors=vline_color)
    ax.tick_params(axis='y', colors=vline_color)

    for spine in ax.spines.values():
        spine.set_color(vline_color)
    
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_xlabel("time (s)", fontsize=16, color=vline_color)
    ax.set_ylabel("h", fontsize=16, color=vline_color)
    
    ax.legend(loc='upper right', facecolor="none", edgecolor=vline_color,
             labelcolor=vline_color, fontsize=11, framealpha=0.0)

    plt.tight_layout()
    if fname:
        if fname.endswith('.svg'):
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(fname, format='svg', transparent=False)
        else:
            plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=False)

    plt.show()
    plt.rcdefaults()
    return fig, ax


def plot_single_signal(
    signal: np.ndarray,
    max_value: float,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    generated: bool = False
) -> plt.Figure:
    """Plot a single waveform signal.
    
    Args:
        signal (np.ndarray): Signal to plot
        max_value (float): Maximum value for scaling
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
        generated (bool): Whether signal is generated (affects color)
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    vline_color = "white" if background == "black" else "black"
    signal_color = GENERATED_SIGNAL_COLOUR if generated else "deepskyblue"

    fig = plt.figure(figsize=(8, 6))
    d = get_time_axis()
    y = signal.flatten() * max_value

    plt.plot(d, y, color=signal_color)
    plt.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    plt.ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    plt.xlabel('time (s)', size=16)
    plt.ylabel('h', size=16)
    plt.grid(True)

    if fname:
        if fname.endswith('.svg'):
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(fname, format='svg', transparent=(background=="black"))
        else:
            plt.savefig(fname, dpi=300, bbox_inches="tight", 
                       transparent=(background=="black"))
    
    plt.show()
    plt.rcdefaults()
    return fig


def plot_signal_distribution(
    signals: np.ndarray,
    generated: bool = False,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    fname: Optional[str] = None
) -> plt.Figure:
    """Plot distribution of signals with percentiles and median.
    
    Args:
        signals (np.ndarray): Array of signals (y_length, num_signals)
        generated (bool): Whether signals are generated
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
        fname (Optional[str]): Filename to save plot
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    vline_color = "white" if background == "black" else "black"
    median_color = vline_color
    text_color = vline_color
    
    fig = plt.figure(figsize=(6, 6))
    distribution_color = GENERATED_SIGNAL_COLOUR if generated else SIGNAL_COLOUR

    signals_df = pd.DataFrame(signals)
    median_line = signals_df.median(axis=1)
    p2_5 = signals_df.quantile(0.025, axis=1)
    p97_5 = signals_df.quantile(0.975, axis=1)
    p25 = signals_df.quantile(0.25, axis=1)
    p75 = signals_df.quantile(0.75, axis=1)

    d = get_time_axis()

    plt.fill_between(d, p2_5, p97_5, color="white", alpha=0.2)
    plt.fill_between(d, p2_5, p97_5, color=distribution_color, alpha=0.4)
    plt.fill_between(d, p25, p75, color="white", alpha=0.4)
    plt.fill_between(d, p25, p75, color=distribution_color, alpha=0.6)

    plt.plot(d, median_line.values, color=median_color,
             linestyle=(0, (1, 1)), linewidth=1.5, alpha=1.0)

    plt.axvline(x=0, color=vline_color, linestyle='--', alpha=0.5)
    plt.ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    plt.xlim(min(d), max(d))
    plt.xlabel('time (s)', size=16, color=text_color)
    plt.ylabel('h', size=16, color=text_color)
    plt.grid(False)

    n = signals.shape[1] if signals.ndim > 1 else len(signals)
    plt.text(
        0.98, 0.02, f"n = {n}",
        ha='right', va='bottom',
        transform=plt.gca().transAxes,
        fontsize=12, color=text_color,
        alpha=0.8
    )

    legend_handles = [
        mpatches.Patch(color=distribution_color, alpha=0.6, 
                      label="Central 95%"),
        mpatches.Patch(color=distribution_color, alpha=1.00, 
                      label="Central 50%"),
        mlines.Line2D([], [], color=median_color, linestyle=(0, (1, 1)), 
                     linewidth=1.5, label="Median")
    ]
    plt.legend(handles=legend_handles, loc='upper right',
              facecolor="none", edgecolor=text_color, 
              labelcolor=text_color, fontsize=12, framealpha=0.0)

    plt.tight_layout()
    if fname:
        if fname.endswith('.svg'):
            plt.rcParams['svg.fonttype'] = 'none'
            plt.savefig(fname, format='svg', 
                       transparent=(background=="black"))
        else:
            plt.savefig(fname, dpi=300, bbox_inches="tight", 
                       transparent=(background=="black"))
