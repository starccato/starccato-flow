"""Signal plotting functions for waveform visualization."""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import torch
from . import set_plot_style, get_time_axis
from ..utils.plotting_defaults import (
    SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR, DEFAULT_FONT_FAMILY, 
    DEFAULT_FONT, SIGNAL_LIM_UPPER, SIGNAL_LIM_LOWER
)


def plot_signal_grid(
    signals: np.ndarray,
    noisy_signals: np.ndarray,
    num_cols: int = 2,
    num_rows: int = 4,
    fname: Optional[str] = None,
    generated: bool = False,
    background: str = "white",
    font_family: str = DEFAULT_FONT_FAMILY,
    font_name: str = DEFAULT_FONT
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
    
    Returns:
        Tuple[plt.Figure, plt.Axes]: Figure and axes objects
    """
    set_plot_style(background, font_family, font_name)
    
    signal_colour = GENERATED_SIGNAL_COLOUR if generated else SIGNAL_COLOUR
    vline_color = "white" if background == "black" else "black"

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    axes = axes.flatten()

    d = get_time_axis()

    for i, ax in enumerate(axes):
        if i >= len(signals):
            ax.axis('off')
            continue
            
        y = signals[i].flatten()
        ax.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
        ax.set_xlim(min(d), max(d))
        ax.plot(d, y, color=signal_colour)
        
        ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
        ax.grid(False)
        
        if i % num_cols != 0:
            ax.yaxis.set_ticklabels([])
        if i < num_cols * (num_rows - 1):
            ax.xaxis.set_ticklabels([])

    fig.supxlabel('time (s)', fontsize=20)
    fig.supylabel('h', fontsize=20)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.show()
    plt.rcdefaults()
    return fig, axes


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
    
    Args:
        original (torch.Tensor): Original signal
        reconstructed (torch.Tensor): Reconstructed signal
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

    fig, ax = plt.subplots(figsize=(15, 4))
    d = get_time_axis()

    y_original = original.flatten() * max_value
    y_reconstructed = reconstructed.flatten() * max_value
    
    ax.plot(d, y_original, color="deepskyblue", 
            label="Original Signal", linewidth=2)
    ax.plot(d, y_reconstructed, color=GENERATED_SIGNAL_COLOUR, 
            label="Reconstructed Signal", linewidth=2)

    ax.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", colors=vline_color, labelsize=12)
    
    for spine in ax.spines.values():
        spine.set_color(vline_color)
    
    ax.set_xlabel("time (s)", fontsize=16, color=vline_color)
    ax.set_ylabel("h", fontsize=16, color=vline_color)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.0, 
             labelcolor=vline_color)

    n = len(y_original)
    plt.text(
        0.98, 0.02, f"n = {n}",
        ha='right', va='bottom',
        transform=ax.transAxes,
        fontsize=12, color=vline_color,
        alpha=0.8
    )

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", 
                   transparent=(background=="black"))

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
        plt.savefig(fname, dpi=300, bbox_inches="tight", 
                   transparent=(background=="black"))

    plt.show()
    plt.rcdefaults()
    return fig
