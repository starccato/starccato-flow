from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import torch
from ..nn.vae import VAE
from ..utils.defaults import DEVICE, TEN_KPC
from .plotting_defaults import SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR, LATENT_SPACE_COLOUR, DEFAULT_FONT_SIZE, DEFAULT_FONT_FAMILY, DEFAULT_FONT, SIGNAL_LIM_UPPER, SIGNAL_LIM_LOWER
import corner
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, ReversePermutation, MaskedAffineAutoregressiveTransform
from nflows.flows import Flow
from PIL import Image
import io

import math

def set_plot_style(background: str = "white", font_family: str = "serif", font_name: str = "Times New Roman") -> None:
    """Set consistent matplotlib plot styling.
    
    Args:
        background (str): Background color, either "white" or "black"
        font_family (str): Font family to use
        font_name (str): Specific font name to use
    """
    if background == "black":
        plt.style.use('dark_background')
        text_color = 'white'
        background_color = 'black'
    else:
        plt.style.use('default')
        text_color = 'black'
        background_color = 'white'
    
    plt.rcParams.update({
        'axes.facecolor': background_color,
        'figure.facecolor': background_color,
        'savefig.facecolor': background_color,
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        'font.family': font_family,
        f'font.{font_family}': [font_name],
        'font.size': DEFAULT_FONT_SIZE
    })

def get_time_axis(length: int = 256) -> np.ndarray:
    """Generate consistent time axis values.
    
    Args:
        length (int): Number of time points
    
    Returns:
        np.ndarray: Array of time values
    """
    return np.linspace(-53 / 4096, (length - 53) / 4096, length)

def plot_signal_grid(
    signals: np.ndarray,
    noisy_signals: np.ndarray,
    max_value: float,
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
        max_value (float): Maximum value for scaling
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
    # Set consistent styling
    set_plot_style(background, font_family, font_name)
    
    # Set colors based on background and generated status
    signal_colour = GENERATED_SIGNAL_COLOUR if generated else SIGNAL_COLOUR
    vline_color = "white" if background == "black" else "black"

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    axes = axes.flatten()

    # Get time axis
    d = get_time_axis()

    # Plot each signal
    for i, ax in enumerate(axes):
        if i >= len(signals):  # Handle case where fewer signals than slots
            ax.axis('off')
            continue
            
        y = signals[i].flatten() * max_value
        ax.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
        ax.set_xlim(min(d), max(d))
        ax.plot(d, y, color=signal_colour)
        
        ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
        ax.grid(False)
        
        # Handle axis labels
        # Only show y-axis ticks on leftmost plots (first column)
        if i % num_cols != 0:
            ax.yaxis.set_ticklabels([])
        # Only show x-axis ticks on bottom row
        if i < num_cols * (num_rows - 1):
            ax.xaxis.set_ticklabels([])

    # Add overall labels
    fig.supxlabel('time (s)', fontsize=20)
    fig.supylabel('h', fontsize=20)

    # Finalize and save
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.show()
    plt.rcdefaults()  # Reset to default style
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
    # Set consistent styling
    set_plot_style(background, font_family, font_name)
    
    # Set colors based on background and generated status
    clean_color = SIGNAL_COLOUR
    noisy_color = SIGNAL_COLOUR
    vline_color = "white" if background == "black" else "black"
    text_color = vline_color

    # Set up figure
    fig = plt.figure(figsize=(6, 6))

    # Get time axis
    d = get_time_axis()

    # Convert tensors to numpy if needed and scale signals
    if torch.is_tensor(signal):
        y_clean = signal.cpu().numpy().flatten() * max_value
    else:
        y_clean = signal.flatten() * max_value
    
    if torch.is_tensor(noisy_signal):
        y_noisy = noisy_signal.cpu().numpy().flatten() * max_value
    else:
        y_noisy = noisy_signal.flatten() * max_value

    # Plot signals
    plt.plot(d, y_noisy, color=noisy_color, linewidth=1.5, alpha=0.5, label="Signal + Noise")
    plt.plot(d, y_clean, color=clean_color, linewidth=2, alpha=1.0, label="Signal")

    
    # Add reference line and styling
    plt.axvline(x=0, color=vline_color, linestyle='--', alpha=0.5)
    plt.ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    plt.xlim(min(d), max(d))
    plt.xlabel('time (s)', size=16, color=text_color)
    plt.ylabel('h', size=16, color=text_color)
    plt.grid(False)
    
    # Add legend
    plt.legend(loc='lower right', facecolor="none", edgecolor=text_color, 
               labelcolor=text_color, fontsize=12, framealpha=0.0)

    # Finalize and save
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
        plt.show()
        plt.rcdefaults()  # Reset to default style
    
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

    # Create figure and get time axis
    fig, ax = plt.subplots(figsize=(15, 4))
    d = get_time_axis()

    # Plot signals
    y_original = original.flatten() * max_value
    y_reconstructed = reconstructed.flatten() * max_value
    
    ax.plot(d, y_original, color="deepskyblue", 
            label="Original Signal", linewidth=2)
    ax.plot(d, y_reconstructed, color=GENERATED_SIGNAL_COLOUR, 
            label="Reconstructed Signal", linewidth=2)

    # Style the plot
    ax.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", colors=vline_color, labelsize=12)
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_color(vline_color)
    
    # Labels
    ax.set_xlabel("time (s)", fontsize=16, color=vline_color)
    ax.set_ylabel("h", fontsize=16, color=vline_color)
    ax.legend(fontsize=12, loc='upper right', framealpha=0.0, 
             labelcolor=vline_color)

    # Add sample size note
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

def plot_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    fname: Optional[str] = None,
    axes: Optional[plt.Axes] = None,
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir"
):
    """Plot training and validation loss curves.
    
    Args:
        train_losses (List[float]): List of training loss values
        val_losses (Optional[List[float]]): List of validation loss values
        fname (Optional[str]): Filename to save plot
        axes (Optional[plt.Axes]): Existing axes to plot on
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        plt.Axes: The plot axes
    """
    set_plot_style(background, font_family, font_name)
    
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()
    
    # Plot training losses - solid line
    axes.plot(train_losses, label="Training Loss", color=SIGNAL_COLOUR, 
              linewidth=3, alpha=1.0, linestyle='-')
    
    # Plot validation losses if provided - solid line
    if val_losses is not None:
        axes.plot(val_losses, label="Validation Loss", color=GENERATED_SIGNAL_COLOUR, 
                  linewidth=3, alpha=1.0, linestyle='-')
    
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel("Negative Log Likelihood", size=20)
    axes.legend(fontsize=20, framealpha=0.0)
    axes.tick_params(labelsize=18)
    axes.grid(False)
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.rcdefaults()

def plot_individual_loss(
    total_losses: List[float],
    reconstruction_losses: List[float],
    kld_losses: List[float],
    fname: Optional[str] = None,
    axes: Optional[plt.Axes] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> Union[plt.Figure, plt.Axes]:
    """Plot individual components of the loss.
    
    Args:
        total_losses (List[float]): Total loss values
        reconstruction_losses (List[float]): Reconstruction loss values
        kld_losses (List[float]): KLD loss values
        fname (Optional[str]): Filename to save plot
        axes (Optional[plt.Axes]): Existing axes to plot on
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        Union[plt.Figure, plt.Axes]: The figure or axes object depending on input
    """
    set_plot_style(background, font_family, font_name)
    
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()
        return_fig = True
    else:
        return_fig = False

    # Use consistent color scheme with distinct line styles
    axes.plot(total_losses, label="Total Loss", color=SIGNAL_COLOUR, 
              linewidth=3, alpha=1.0, linestyle='-')
    axes.plot(reconstruction_losses, label="Reconstruction Loss", color=GENERATED_SIGNAL_COLOUR, 
              linewidth=3, alpha=1.0, linestyle='--', dashes=(5, 3))
    axes.plot(kld_losses, label="KLD Loss", color=LATENT_SPACE_COLOUR, 
              linewidth=3.5, alpha=1.0, linestyle=':')
    
    axes.set_xlabel("Epoch", size=16)
    axes.set_ylabel("Loss", size=16)
    axes.legend(fontsize=12, framealpha=0.0)
    axes.grid(False)
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.rcdefaults()
    return fig if return_fig else axes

def plot_training_validation_loss(
    losses: List[float],
    validation_losses: List[float],
    fname: Optional[str] = None,
    axes: Optional[plt.Axes] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> plt.Axes:
    """Plot training and validation loss curves.
    
    Args:
        losses (List[float]): Training loss values
        validation_losses (List[float]): Validation loss values
        fname (Optional[str]): Filename to save plot
        axes (Optional[plt.Axes]): Existing axes to plot on
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        plt.Axes: The plot axes
    """
    set_plot_style(background, font_family, font_name)
    
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()

    axes.plot(losses, label="Training Loss", color='orange')
    axes.plot(validation_losses, label="Validation Loss", color='grey')
    axes.set_xlabel("Epoch", size=16)
    axes.set_ylabel("Loss", size=16)
    axes.legend(fontsize=12)
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.rcdefaults()
    return axes
    
    # return axes.get_figure()

def plot_parameter_distribution(
    values: Union[List[float], np.ndarray],
    param_name: str,
    param_label: Optional[str] = None,
    bins: int = 25,
    fname: Optional[str] = None,
    axes: Optional[plt.Axes] = None,
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    color: Optional[str] = None,
    alpha: float = 0.8,
    show_stats: bool = True,
    param_range: Optional[Tuple[float, float]] = None
) -> Union[plt.Figure, plt.Axes]:
    """Plot the distribution of a single parameter.
    
    Args:
        values (Union[List[float], np.ndarray]): Parameter values to plot
        param_name (str): Name of the parameter
        param_label (Optional[str]): Label for the parameter (LaTeX supported). If None, uses param_name
        bins (int): Number of histogram bins
        fname (Optional[str]): Filename to save plot
        axes (Optional[plt.Axes]): Existing axes to plot on
        background (str): Background color theme ("white" or "black")
        font_family (str): Font family to use
        font_name (str): Specific font name
        color (Optional[str]): Color for the histogram. If None, uses SIGNAL_COLOUR
        alpha (float): Transparency of the histogram bars
        show_stats (bool): Whether to display mean and std on the plot
        param_range (Optional[Tuple[float, float]]): Fixed range for x-axis (min, max). If None, uses data range
    
    Returns:
        Union[plt.Figure, plt.Axes]: The figure or axes object depending on input
    """
    set_plot_style(background, font_family, font_name)
    
    if axes is None:
        fig = plt.figure(figsize=(6, 6))
        axes = fig.gca()
        return_fig = True
    else:
        return_fig = False
    
    # Convert to numpy array if needed
    if isinstance(values, list):
        values = np.array(values)
    
    # Use default color if not specified
    if color is None:
        color = SIGNAL_COLOUR
    
    # Create histogram
    n, bins_edges, patches = axes.hist(
        values, 
        bins=bins, 
        color=color, 
        alpha=alpha, 
        edgecolor='none'
    )
    
    # Add mean line
    mean_val = np.mean(values)
    std_val = np.std(values)
    axes.axvline(mean_val, color=GENERATED_SIGNAL_COLOUR, linewidth=2.5, linestyle='--')    

    # Set labels
    if param_label is None:
        param_label = param_name
    
    axes.set_xlabel(param_label, size=20)
    axes.set_ylabel("Count", size=20)
    # axes.set_title(param_label, size=22, pad=15)
    
    # Set x-axis limits
    if param_range is not None:
        axes.set_xlim(param_range[0], param_range[1])
    else:
        axes.set_xlim(min(values), max(values))
    
    # # Add statistics text box if requested
    # if show_stats:
    #     stats_text = f'μ = {mean_val:.3f}\nσ = {std_val:.3f}\nN = {len(values)}'
    #     text_color = 'white' if background == "black" else 'black'
    #     axes.text(0.97, 0.97, stats_text,
    #              transform=axes.transAxes,
    #              fontsize=14,
    #              verticalalignment='top',
    #              horizontalalignment='right',
    #              bbox=dict(boxstyle='round', facecolor=background, 
    #                       alpha=0.8, edgecolor=text_color, linewidth=1.5))
    
    axes.tick_params(labelsize=18)
    # Rotate x-axis tick labels diagonally like corner plots
    axes.tick_params(axis='x', rotation=45)
    axes.grid(False)
    axes.legend(fontsize=16, framealpha=0.0)
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.rcdefaults()
    return fig if return_fig else axes

def plot_parameter_distribution_grid(
    parameters_dict: dict,
    labels_dict: Optional[dict] = None,
    ranges_dict: Optional[dict] = None,
    bins: int = 25,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    color: Optional[str] = None,
    alpha: float = 0.8,
    figsize: Tuple[float, float] = (20, 5)
) -> plt.Figure:
    """Plot distributions for multiple parameters in a 1x4 grid (one row).
    
    Args:
        parameters_dict (dict): Dictionary mapping parameter names to value arrays
        labels_dict (Optional[dict]): Dictionary mapping parameter names to LaTeX labels
        ranges_dict (Optional[dict]): Dictionary mapping parameter names to (min, max) tuples
        bins (int): Number of histogram bins
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme ("white" or "black")
        font_family (str): Font family to use
        font_name (str): Specific font name
        color (Optional[str]): Color for the histogram. If None, uses SIGNAL_COLOUR
        alpha (float): Transparency of the histogram bars
        figsize (Tuple[float, float]): Figure size in inches
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    
    # Create 1x4 subplot grid (one row, four columns)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    axes = axes.flatten()
    
    # Use default color if not specified
    if color is None:
        color = SIGNAL_COLOUR
    
    # Plot each parameter
    for idx, (param_name, values) in enumerate(parameters_dict.items()):
        if idx >= 4:  # Only plot first 4 parameters
            break
            
        ax = axes[idx]
        
        # Convert to numpy array if needed
        if isinstance(values, list):
            values = np.array(values)
        
        # Create histogram
        n, bins_edges, patches = ax.hist(
            values, 
            bins=bins, 
            color=color, 
            alpha=alpha, 
            edgecolor='none'
        )
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(mean_val, color=GENERATED_SIGNAL_COLOUR, linewidth=2.5, linestyle='--')
        
        # Get label
        if labels_dict and param_name in labels_dict:
            param_label = labels_dict[param_name]
        else:
            param_label = param_name
        
        # Set labels and title
        ax.set_xlabel(param_label, size=16)
        ax.set_ylabel("Count", size=16)
        ax.set_title(param_label, size=18, pad=10)
        
        # Set x-axis limits
        if ranges_dict and param_name in ranges_dict:
            ax.set_xlim(ranges_dict[param_name][0], ranges_dict[param_name][1])
        else:
            ax.set_xlim(min(values), max(values))
        
        # Style ticks
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
    
    # Hide any unused subplots
    for idx in range(len(parameters_dict), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.rcdefaults()
    return fig

def plot_latent_morphs(
    model: VAE, 
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float,
    steps: int = 10,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot a sequence of signals showing latent space morphing between two signals.
    
    Args:
        model (VAE): VAE model for encoding/decoding
        signal_1 (torch.Tensor): Starting signal
        signal_2 (torch.Tensor): Ending signal
        max_value (float): Maximum value for scaling
        steps (int): Number of interpolation steps
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: Figure and list of axes objects
    """
    set_plot_style(background, font_family, font_name)
    vline_color = "white" if background == "black" else "black"
    
    model.eval()
    with torch.no_grad():
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)
        
        # Generate interpolated signals
        interpolated_latents = [mean_1 * (1 - alpha) + mean_2 * alpha 
                              for alpha in np.linspace(0, 1, steps)]
        morphed_signals = [model.decoder(latent).cpu().detach().numpy() 
                          for latent in interpolated_latents]

    # Setup figure
    num_plots = steps + 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    axes = axes.flatten()
    
    # Get time axis
    d = get_time_axis()

    # Plot original signal 1
    y1 = signal_1.cpu().detach().numpy().flatten() * max_value
    axes[0].plot(d, y1, color="deepskyblue")
    axes[0].set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    axes[0].axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    axes[0].grid(True)
    axes[0].set_title("Original Signal 1 (Start)")

    # Plot interpolated signals
    for i, signal in enumerate(morphed_signals):
        y_interp = signal.flatten() * max_value
        axes[i + 1].plot(d, y_interp, color=GENERATED_SIGNAL_COLOUR)
        axes[i + 1].set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
        axes[i + 1].axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
        axes[i + 1].grid(True)
        axes[i + 1].set_title(f"Interpolated Signal {i + 1}")

    # Plot original signal 2
    y2 = signal_2.cpu().detach().numpy().flatten() * max_value
    axes[-1].plot(d, y2, color="deepskyblue")
    axes[-1].set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    axes[-1].axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    axes[-1].grid(True)
    axes[-1].set_title("Original Signal 2 (End)")

    # Add labels
    fig.supxlabel('time (s)', fontsize=16)
    fig.supylabel('h', fontsize=16)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.show()
    plt.rcdefaults()
    return fig, axes

def plot_latent_morph_grid(
    model: VAE,
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float,
    train_dataset: torch.utils.data.Dataset,
    steps: int = 10,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> plt.Figure:
    """Plot a grid showing latent space morphing between two signals with 3D latent visualization.
    
    Args:
        model (VAE): VAE model for encoding/decoding
        signal_1 (torch.Tensor): Starting signal
        signal_2 (torch.Tensor): Ending signal
        max_value (float): Maximum value for scaling
        train_dataset (Dataset): Dataset for latent space visualization
        steps (int): Number of interpolation steps
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    vline_color = "white" if background == "black" else "black"
    latent_scatter_color = LATENT_SPACE_COLOUR
    signal_color = "deepskyblue"
    
    model.eval()
    with torch.no_grad():
        # Encode signals
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)

        # Get middle latent point
        alpha = 0.5
        mean_mid = mean_1 * (1 - alpha) + mean_2 * alpha
        signal_mid = model.decoder(mean_mid).cpu().detach().numpy().flatten() * max_value

        # Reconstruct signals
        signal_1_np = signal_1.cpu().detach().numpy().flatten() * max_value
        signal_2_np = signal_2.cpu().detach().numpy().flatten() * max_value

        # Get time axis
        d = get_time_axis()

        # Posterior means for background latent scatter
        all_means = []
        for x, _ in train_dataset:
            x = torch.tensor(x).to(DEVICE)
            mean, _ = model.encoder(x)
            all_means.append(mean.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)

    # Set up figure
    fig = plt.figure(figsize=(15, 10))
    
    # ----- Row 1: Signals -----
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(d, signal_1_np, color=signal_color)
    ax1.axvline(x=0, linestyle="--", color=vline_color, alpha=0.5)
    ax1.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax1.set_title("Start Signal")
    ax1.set_ylabel("h", fontsize=16)
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(d, signal_mid, color=GENERATED_SIGNAL_COLOUR)
    ax2.axvline(x=0, linestyle="--", color=vline_color, alpha=0.5)
    ax2.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax2.set_title("Interpolated Signal")
    ax2.set_xlabel("time (s)", fontsize=16)
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(d, signal_2_np, color=signal_color)
    ax3.axvline(x=0, linestyle="--", color=vline_color, alpha=0.5)
    ax3.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax3.set_title("End Signal")
    ax3.set_xlabel("time (s)", fontsize=16)
    ax3.grid(True)

    # ----- Row 2: Latent space -----
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], 
               alpha=0.1, color=latent_scatter_color)
    ax4.scatter(mean_1[0].cpu(), mean_1[1].cpu(), mean_1[2].cpu(), 
               color=signal_color, label="Start")
    ax4.set_title("Start in Latent Space")
    ax4.set_xlabel('Latent Dim 1')
    ax4.set_ylabel('Latent Dim 2')
    ax4.set_zlabel('Latent Dim 3')

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], 
               alpha=0.1, color=latent_scatter_color)
    ax5.plot(
        [mean_1[0].cpu(), mean_2[0].cpu()],
        [mean_1[1].cpu(), mean_2[1].cpu()],
        [mean_1[2].cpu(), mean_2[2].cpu()],
        color=GENERATED_SIGNAL_COLOUR, linestyle='--'
    )
    ax5.scatter(mean_1[0].cpu(), mean_1[1].cpu(), mean_1[2].cpu(), 
               color=signal_color, label="Start", alpha=0.5)
    ax5.scatter(mean_mid[0].cpu(), mean_mid[1].cpu(), mean_mid[2].cpu(), 
               color=GENERATED_SIGNAL_COLOUR, label="Midpoint")
    ax5.scatter(mean_2[0].cpu(), mean_2[1].cpu(), mean_2[2].cpu(), 
               color=signal_color, label="End", alpha=0.5)
    ax5.set_title("Latent Space Path")
    ax5.set_xlabel('Latent Dim 1')
    ax5.set_ylabel('Latent Dim 2')
    ax5.set_zlabel('Latent Dim 3')
    ax5.legend()

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], 
               alpha=0.1, color=latent_scatter_color)
    ax6.scatter(mean_2[0].cpu(), mean_2[1].cpu(), mean_2[2].cpu(), 
               color=signal_color, label="End")
    ax6.set_title("End in Latent Space")
    ax6.set_xlabel('Latent Dim 1')
    ax6.set_ylabel('Latent Dim 2')
    ax6.set_zlabel('Latent Dim 3')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Add vertical space between rows
    
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.show()
    plt.rcdefaults()
    return fig

def animate_latent_morphs(
    model: VAE,
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float, 
    train_dataset: torch.utils.data.Dataset,
    steps: int = 10,
    interval: int = 200,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> animation.Animation:
    """Create an animation of latent space morphing between two signals.
    
    Args:
        model (VAE): VAE model for encoding/decoding
        signal_1 (torch.Tensor): Starting signal
        signal_2 (torch.Tensor): Ending signal
        max_value (float): Maximum value for scaling
        train_dataset (Dataset): Dataset for latent space visualization
        steps (int): Number of interpolation steps
        interval (int): Animation interval in milliseconds
        fname (Optional[str]): Filename to save animation
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        animation.Animation: The animation object
    """
    set_plot_style(background, font_family, font_name)
    vline_color = "white" if background == "black" else "black"
    signal_color = "deepskyblue"
    
    model.eval()
    with torch.no_grad():
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)

        # Forward and backward interpolation
        forward_interpolated = [mean_1 * (1 - alpha) + mean_2 * alpha 
                              for alpha in np.linspace(0, 1, steps)]
        backward_interpolated = [mean_2 * (1 - alpha) + mean_1 * alpha 
                               for alpha in np.linspace(0, 1, steps)]
        interpolated_latents = forward_interpolated + backward_interpolated
        morphed_signals = [model.decoder(latent).cpu().detach().numpy() 
                          for latent in interpolated_latents]

        # Compute the posterior distribution
        all_means = []
        for x, y in train_dataset:
            x = torch.tensor(x).to(DEVICE)
            mean, _ = model.encoder(x)
            all_means.append(mean.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)

    # Setup figure
    fig = plt.figure(figsize=(10, 17))

    # Create 3D plot for latent space
    ax_latent = fig.add_subplot(211, projection='3d')
    ax_latent.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], 
                     color=LATENT_SPACE_COLOUR, alpha=0.2, label='Posterior Distribution')
    ax_latent.scatter(mean_1[0].cpu().numpy(), mean_1[1].cpu().numpy(), mean_1[2].cpu().numpy(), 
                     color=signal_color, s=50, label='Signal 1')
    ax_latent.scatter(mean_2[0].cpu().numpy(), mean_2[1].cpu().numpy(), mean_2[2].cpu().numpy(), 
                     color=signal_color, s=50, label='Signal 2')
    ax_latent.plot([mean_1[0].cpu().numpy(), mean_2[0].cpu().numpy()],
                   [mean_1[1].cpu().numpy(), mean_2[1].cpu().numpy()],
                   [mean_1[2].cpu().numpy(), mean_2[2].cpu().numpy()], 
                   color=GENERATED_SIGNAL_COLOUR, linestyle='--', 
                   label='Interpolation Path', linewidth=2)
    moving_point, = ax_latent.plot([], [], [], 'ro', markersize=7, 
                                 label='Interpolated Point')
    ax_latent.set_xlabel('Latent Dim 1')
    ax_latent.set_ylabel('Latent Dim 2')
    ax_latent.set_zlabel('Latent Dim 3')

    # Create plot for signal morphing
    ax_signal = fig.add_subplot(212)
    d = get_time_axis()

    # Initialize the plot
    line, = ax_signal.plot([], [], color=GENERATED_SIGNAL_COLOUR)
    ax_signal.set_xlim(min(d), max(d))
    ax_signal.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax_signal.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)
    ax_signal.grid(True)
    ax_signal.set_xlabel('time (s)', fontsize=16)
    ax_signal.set_ylabel('h', fontsize=16)

    def init():
        line.set_data([], [])
        moving_point.set_data([], [])
        moving_point.set_3d_properties([])
        return line, moving_point

    def update(frame):
        y_interp = morphed_signals[frame].flatten() * max_value
        line.set_data(d, y_interp)

        latent_point = interpolated_latents[frame].cpu().numpy()
        moving_point.set_data(latent_point[0], latent_point[1])
        moving_point.set_3d_properties(latent_point[2])
        return line, moving_point

    ani = animation.FuncAnimation(
        fig, update, frames=len(interpolated_latents),
        init_func=init, blit=True, interval=interval, repeat=True
    )

    if fname:
        ani.save(fname, writer='imagemagick', fps=30)

    plt.show()
    plt.rcdefaults()
    return ani

def plot_latent_morph_up_and_down(
    model,
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float,
    train_dataset,
    steps=10,
    background="white",
    font_family="sans-serif",
    font_name="Avenir",
    fname="plots/latent_morph.svg"
):
    model.eval()
    with torch.no_grad():
        # Encode signals
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)

        # Get middle latent point
        alpha = 0.5
        mean_mid = mean_1 * (1 - alpha) + mean_2 * alpha
        signal_mid = model.decoder(mean_mid).cpu().detach().numpy().flatten() * max_value

        # Reconstruct signals
        signal_1_np = signal_1.cpu().detach().numpy().flatten() * max_value
        signal_2_np = signal_2.cpu().detach().numpy().flatten() * max_value

        # X-axis
        x_vals = [i / 4096 for i in range(0, 256)]
        x_vals = [x - (53 / 4096) for x in x_vals]

        # Posterior means for background latent scatter
        all_means = []
        for x, _ in train_dataset:
            x = torch.tensor(x).to(DEVICE)
            mean, _ = model.encoder(x)
            all_means.append(mean.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)


    if background == "black":
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['figure.facecolor'] = 'black'
        plt.rcParams['savefig.facecolor'] = 'black'
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'
        text_colour = 'white'
        grid_color = 'white'
        vline_color = 'white'
    else:
        plt.style.use('default')
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['text.color'] = 'black'
        plt.rcParams['axes.labelcolor'] = 'black'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        text_colour = 'black'
        grid_color = 'black'
        vline_color = 'black'
    
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.sans-serif'] = font_name
    plt.rcParams['font.size'] = 12

    mean_1 = mean_1.squeeze()
    mean_2 = mean_2.squeeze()
    mean_mid = mean_mid.squeeze()

    # Set up figure
    px = 1/plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(600*px, 900*px))

    ax1 = fig.add_subplot(2, 1, 1)
    # plot all the signals on the plot
    # To ensure the red "Interpolant" plot is on top, set a higher zorder for it:
    ax1.plot(x_vals, signal_1_np, color='deepskyblue', label='Signal 1', alpha=0.5, linewidth=2, zorder=1)
    ax1.plot(x_vals, signal_mid, color='red', label='Interpolant', linewidth=2, zorder=3)  # Highest zorder
    ax1.plot(x_vals, signal_2_np, color='deepskyblue', label='Signal 2', alpha=0.75, linewidth=2, zorder=2)
    plt.axvline(x=0, color=vline_color, linestyle='dashed', alpha=0.5)
    ax1.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    ax1.set_xlim(left=x_vals[0], right=x_vals[-1])
    ax1.set_xlabel("time (s)", fontsize=16)
    ax1.set_ylabel("h", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)    

    ax1.legend(loc='upper center', fontsize=12, facecolor='none', bbox_to_anchor=(0.5, 1.125), ncol=3, frameon=False)

    # create the latent plot, but only use the first 2 dimensions
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.scatter(all_means[:, 0], all_means[:, 1], alpha=0.2, color='gray', edgecolors='none', s=50)
    ax2.plot(
        [mean_1[0].cpu(), mean_2[0].cpu()],
        [mean_1[1].cpu(), mean_2[1].cpu()],
        color='red', linestyle='--', linewidth=3
    )
    ax2.scatter(mean_1[0].cpu(), mean_1[1].cpu(), color='deepskyblue', label="Signal 1", alpha=0.5, edgecolors='none', s=100)
    ax2.scatter(mean_mid[0].cpu(), mean_mid[1].cpu(), color='red', label="Interpolant", edgecolors='none', s=100)
    ax2.scatter(mean_2[0].cpu(), mean_2[1].cpu(), color='deepskyblue', label="Signal 2", alpha=0.75, edgecolors='none', s=100)
    ax2.set_xlabel('Latent Dimension 1', fontsize=16)
    ax2.set_ylabel('Latent Dimension 2', fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(loc='upper center', fontsize=12, facecolor='none', bbox_to_anchor=(0.5, 1.125), ncol=3, frameon=False)

    # Sample size note
    n = 1684
    plt.text(
        0.98, 0.02, f"n = {n}",
        ha='right', va='bottom',
        transform=plt.gca().transAxes,
        fontsize=12, color=text_colour,
        alpha=0.8
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(fname=fname, dpi=300, bbox_inches='tight', transparent=True)

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
    
    # Set up figure
    fig = plt.figure(figsize=(6, 6))
    distribution_color = GENERATED_SIGNAL_COLOUR if generated else SIGNAL_COLOUR

    # Calculate statistics
    signals_df = pd.DataFrame(signals)
    median_line = signals_df.median(axis=1)
    p2_5 = signals_df.quantile(0.025, axis=1)
    p97_5 = signals_df.quantile(0.975, axis=1)
    p25 = signals_df.quantile(0.25, axis=1)
    p75 = signals_df.quantile(0.75, axis=1)

    # Get time axis
    d = get_time_axis()

    # Plot percentiles with white base + overlay
    plt.fill_between(d, p2_5, p97_5, color="white", alpha=0.2)
    plt.fill_between(d, p2_5, p97_5, color=distribution_color, alpha=0.4)
    plt.fill_between(d, p25, p75, color="white", alpha=0.4)
    plt.fill_between(d, p25, p75, color=distribution_color, alpha=0.6)

    # Plot median line
    plt.plot(d, median_line.values, color=median_color,
             linestyle=(0, (1, 1)), linewidth=1.5, alpha=1.0)

    # Add reference line and styling
    plt.axvline(x=0, color=vline_color, linestyle='--', alpha=0.5)
    plt.ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
    plt.xlim(min(d), max(d))
    plt.xlabel('time (s)', size=16, color=text_color)
    plt.ylabel('h', size=16, color=text_color)
    plt.grid(False)

    # Add sample size note
    n = signals.shape[1] if signals.ndim > 1 else len(signals)
    plt.text(
        0.98, 0.02, f"n = {n}",
        ha='right', va='bottom',
        transform=plt.gca().transAxes,
        fontsize=12, color=text_color,
        alpha=0.8
    )

    # Add legend
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

def plot_gradients(
    encoder_gradients: List[float],
    decoder_gradients: List[float],
    q_gradients: List[float],
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Plot encoder, decoder, and Q network gradient norms.
    
    Args:
        encoder_gradients (List[float]): Encoder gradient norms
        decoder_gradients (List[float]): Decoder gradient norms
        q_gradients (List[float]): Q network gradient norms
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: Figure and list of axes
    """
    set_plot_style(background, font_family, font_name)

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    colors = ["deepskyblue", GENERATED_SIGNAL_COLOUR, "green"]
    
    # Plot gradients
    for ax, grads, title, color in zip(
        axes,
        [encoder_gradients, decoder_gradients, q_gradients],
        ["Encoder", "Decoder", "Q Network"],
        colors
    ):
        ax.plot(grads, label=f'{title} Gradients', color=color)
        ax.set_title(f'{title} Gradient Norms During Training', fontsize=14)
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Gradient Norm', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", 
                   transparent=(background=="black"))

    plt.show()
    plt.rcdefaults()
    return fig, axes

def plot_latent_space_3d(
    model: VAE,
    dataloader: torch.utils.data.DataLoader,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> plt.Figure:
    """Plot 3D visualization of the latent space.
    
    Args:
        model (VAE): VAE model for encoding
        dataloader (DataLoader): Dataloader for getting samples
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    
    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for y, y_noisy, d in dataloader:
            y = y.to(DEVICE)
            d = d.to(DEVICE)
            d = d.view(d.size(0), -1)
            y = y.view(y.size(0), -1)

            mean, _ = model.encoder(y)
            latent_vectors.append(mean.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    latent_3d = latent_vectors[:, :3]  # First 3 dimensions

    # Plot in 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2],
                        color=LATENT_SPACE_COLOUR, s=40, alpha=0.7)

    ax.set_title('Latent Space Representation (3D)', fontsize=16)
    ax.set_xlabel('Latent Dimension 1', fontsize=12)
    ax.set_ylabel('Latent Dimension 2', fontsize=12)
    ax.set_zlabel('Latent Dimension 3', fontsize=12)
    ax.grid(True)
    
    n = latent_vectors.shape[0]
    ax.text2D(
        0.98, 0.02,
        s=f"n = {n}",
        ha='right',
        va='bottom',
        transform=ax.transAxes,
        fontsize=12,
        alpha=0.8
    )

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", 
                   transparent=(background=="black"))

    plt.show()
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
        vae (VAE): VAE model for generating reconstructions
        signal (torch.Tensor): Signal to reconstruct
        max_value (float): Maximum value for scaling
        num_samples (int): Number of reconstructions to generate
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        plt.Figure: The figure object
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


def plot_corner(vae: VAE, flow: Flow, signal, noisy_signal, params, fname="plots/corner_plot.png"):
    """Plot corner plot of parameter posterior distribution.
    
    Args:
        noisy_signal (torch.Tensor): Noisy input signal
        params (torch.Tensor): True parameter values
        fname (Optional[str]): Filename to save plot
    """
    vae.eval()
    flow.eval()

    with torch.no_grad():
        noisy_signal = noisy_signal.to(DEVICE).float()
        if noisy_signal.dim() == 2:
            noisy_signal = noisy_signal.unsqueeze(0)
        
        _, mean, log_var = vae(noisy_signal)

        # Sample from flow conditioned on z
        num_draws = 5000

        context = mean.view(1, -1)
        samples = flow.sample(num_samples=num_draws, context=context)
        samples = samples.reshape(num_draws, -1)  # -> [num_draws, 2]

        samples_cpu = samples.detach().cpu()
        samples_cpu[:, [0, 1, 3]] = torch.exp(samples_cpu[:, [0, 1, 3]])  # Transform back to positive space
        samples_cpu = samples_cpu.numpy()
        true_params = params.detach().cpu() if torch.is_tensor(params) else params
        true_params = true_params.flatten()  # Flatten to [2] from [1, 2]
        true_params[2] = torch.log(true_params[2] + 1e-8)  # log-transform
        
        print("True params:", true_params)

    plt.rcParams['figure.facecolor'] = 'none' # Transparent figure background
    plt.rcParams['axes.facecolor'] = 'black' # Black subplot backgrounds
    plt.rcParams['savefig.facecolor'] = 'none' # Also transparent when saving
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    print(true_params.detach().cpu().numpy())

    figure = corner.corner(
        samples_cpu,
        labels=[
            r"$\beta_{IC,b}$",
            r"$\omega_0$",
            r"$\log(A)$",
            r"$Y_{e,b,c}$",
        ],
        range=[(0, 0.25), (0, 16), (0, math.log(10000)), (0, 0.3)],
        truths=true_params[:4].numpy(),
        truth_color=SIGNAL_COLOUR,
        show_titles=True,
        title_quantiles=[0.16, 0.5, 0.84],
        title_fmt='.4f',
        title_kwargs={'fontsize': 12},  # Summary text on histograms
        label_kwargs={'fontsize': 24},  # Parameter names
        bins=100,
        smooth=3,
        color=GENERATED_SIGNAL_COLOUR,
        hist_kwargs={'density': False, 'alpha': 1.0},
        levels=(0.68, 0.95),
        fill_contours=True,
        plot_datapoints=False
    )

    # Fill hist patches
    for ax in figure.get_axes():
        for patch in ax.patches:
            patch.set_facecolor("white")
            patch.set_alpha(1.0)

    # **Make axis lines white and adjust tick labels**
    for ax in figure.get_axes():
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        # Axis tick numbers
        ax.tick_params(labelsize=12)
        # Reduce label padding to save space
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2

    # Transparent canvas
    figure.patch.set_alpha(1.0)

    # Reduce spacing between subplots to make plots bigger
    figure.subplots_adjust(hspace=0.05, wspace=0.05)
    
    plt.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

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
        dataset: Dataset object with signals (e.g., CCSNSNRData)
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
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
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
        dataset: Dataset object (e.g., CCSNSNRData) with calculate_snr and aLIGO_noise methods
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
    from ..utils.defaults import SAMPLING_RATE, Y_LENGTH
    
    is_even = (Y_LENGTH % 2 == 0)
    half_N = Y_LENGTH // 2 if is_even else (Y_LENGTH - 1) // 2
    delta_f = 1 / (Y_LENGTH * SAMPLING_RATE)
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
        # We need to temporarily disable plt.show() and handle the figure ourselves
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
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
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
