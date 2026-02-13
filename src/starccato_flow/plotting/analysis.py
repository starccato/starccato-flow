"""Analysis and visualization functions for model evaluation."""

from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
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
    SIGNAL_LIM_LOWER
)
from . import set_plot_style, get_time_axis
from .signals import plot_signal_grid, plot_candidate_signal


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


def plot_corner(samples_cpu, true_params, fname="plots/corner_plot.png", dataset=None, 
                labels=None, ranges=None):
    """Plot corner plot of parameter posterior distribution.
    
    Args:
        samples_cpu (np.ndarray): Posterior samples as numpy array, shape (num_samples, num_params)
        true_params (np.ndarray): True parameter values as numpy array, shape (num_params,)
        fname (str): Filename to save plot
        dataset: Optional dataset object (CCSNData or ToyData) to extract parameter metadata
        labels (list): Optional custom labels for parameters. If None, will be inferred from dataset or num_params
        ranges (list): Optional custom ranges for parameters as list of tuples [(min, max), ...]
    """
    # Detect number of parameters
    num_params = samples_cpu.shape[1]
    
    # If dataset is provided, extract parameter names and labels
    if dataset is not None and labels is None:
        if hasattr(dataset, 'parameter_names') and hasattr(dataset, 'PARAMETER_LABELS'):
            # CCSN data with parameter metadata
            labels = [dataset.PARAMETER_LABELS.get(name, name) for name in dataset.parameter_names]
        elif hasattr(dataset, 'parameter_names'):
            # Has parameter names but no labels
            labels = [name.replace('_', ' ').title() for name in dataset.parameter_names]
    
    # If dataset is provided and ranges not specified, try to extract them
    if dataset is not None and ranges is None:
        if hasattr(dataset, 'parameter_names') and hasattr(dataset, 'PARAMETER_RANGES'):
            # Build ranges list from parameter names
            ranges = [dataset.PARAMETER_RANGES.get(name, None) for name in dataset.parameter_names]
            # If any range is None, set entire ranges to None (let corner auto-determine)
            if None in ranges:
                ranges = None
    
    # If labels still not set, use default logic based on number of parameters
    if labels is None:
        if num_params == 2:
            # Toy data with 2 parameters
            labels = [r"Parameter 1", r"Parameter 2"]
        elif num_params == 4:
            # CCSN data with 4 parameters (legacy fallback)
            labels = [
                r"$\beta_{IC,b}$",
                r"$\omega_0$",
                r"$A$",
                r"$Y_{e,c,b}$",
            ]
        else:
            # Generic labels for other cases
            labels = [f"Parameter {i+1}" for i in range(num_params)]
    
    # Set default ranges if not provided
    if ranges is None:
        if num_params == 2:
            ranges = [(-3, 3), (-3, 3)]
        elif num_params == 4:
            # Legacy fallback for 4 parameters
            ranges = [(0, 0.25), (0, 16), (0, 10000), (0.2, 0.3)]
        # Otherwise ranges will be None and corner will auto-determine
    
    plt.rcParams['figure.facecolor'] = 'none'  # Transparent figure background
    plt.rcParams['axes.facecolor'] = 'black'  # Black subplot backgrounds
    plt.rcParams['savefig.facecolor'] = 'none'  # Also transparent when saving
    plt.rcParams['text.color'] = 'white'
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'

    # Special case for single parameter - corner library has issues with this
    if num_params == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(samples_cpu.flatten(), bins=100, color=GENERATED_SIGNAL_COLOUR, 
                alpha=0.7, density=True, edgecolor='none')
        if true_params is not None and len(true_params) > 0:
            ax.axvline(true_params[0], color=SIGNAL_COLOUR, linewidth=2, label='True value')
        ax.set_xlabel(labels[0] if labels else 'Parameter', fontsize=24, color='white')
        ax.set_ylabel('Density', fontsize=24, color='white')
        if ranges is not None and ranges[0] is not None:
            ax.set_xlim(ranges[0])
        ax.tick_params(labelsize=12, colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        ax.set_facecolor('black')
        fig.patch.set_alpha(1.0)
        
        # Add title with quantiles
        q = np.percentile(samples_cpu.flatten(), [16, 50, 84])
        title = f"{q[1]:.4f}$_{{-{q[1]-q[0]:.4f}}}^{{+{q[2]-q[1]:.4f}}}$"
        ax.set_title(title, fontsize=12, color='white')
        
        plt.savefig(fname, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()
        return

    corner_kwargs = {
        'labels': labels,
        'truths': true_params[:num_params],
        'truth_color': SIGNAL_COLOUR,
        'show_titles': True,
        'title_quantiles': [0.16, 0.5, 0.84],
        'title_fmt': '.4f',
        'title_kwargs': {'fontsize': 12},
        'label_kwargs': {'fontsize': 24},
        'bins': 100,
        'smooth': 3,
        'color': GENERATED_SIGNAL_COLOUR,
        'hist_kwargs': {'density': False, 'alpha': 1.0},
        'levels': (0.68, 0.95),
        'fill_contours': True,
        'plot_datapoints': False
    }
    
    # Add range only if specified
    if ranges is not None:
        corner_kwargs['range'] = ranges
    
    figure = corner.corner(samples_cpu, **corner_kwargs)

    # Fill hist patches
    for ax in figure.get_axes():
        for patch in ax.patches:
            patch.set_facecolor("white")
            patch.set_alpha(1.0)

    # Make axis lines white and adjust tick labels
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


def plot_sky_localisation(
    ra_samples: np.ndarray,
    dec_samples: np.ndarray,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> plt.Figure:
    """Plot sky location distribution from RA and Dec samples.
    
    Args:
        ra_samples (np.ndarray): Right Ascension samples in radians
        dec_samples (np.ndarray): Declination samples in radians
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
    
    # Create figure with ligo.skymap projection
    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection='geo aitoff')
    
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
