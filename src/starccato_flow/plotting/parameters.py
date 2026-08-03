"""Parameter distribution plotting functions."""

from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import corner
from scipy import stats
from . import set_plot_style
from ..utils.defaults_plotting import (
    SIGNAL_COLOUR,
    GENERATED_SIGNAL_COLOUR,
    PARAMETER_LABELS,
    PARAMETER_COLOURS,
    PARAMETER_RANGES,
    CM_TO_INCHES
)

from pypalettes import load_cmap

def plot_parameter_distribution(
    values: Union[list, np.ndarray],
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
    param_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (14.5, 14.5)
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
        figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
        fig = plt.figure(figsize=figsize)
        axes = fig.gca()
        return_fig = True
    else:
        return_fig = False
    
    if isinstance(values, list):
        values = np.array(values)
    
    if color is None:
        color = SIGNAL_COLOUR
    
    n, bins_edges, patches = axes.hist(
        values, 
        bins=bins, 
        color=color, 
        alpha=alpha, 
        edgecolor='none'
    )
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    axes.axvline(mean_val, color=GENERATED_SIGNAL_COLOUR, linewidth=2.5, linestyle='--')    

    if param_label is None:
        param_label = param_name
    
    axes.set_xlabel(param_label, size=20)
    axes.set_ylabel("Count", size=20)
    
    if param_range is not None:
        axes.set_xlim(param_range[0], param_range[1])
    else:
        axes.set_xlim(min(values), max(values))
    
    axes.tick_params(labelsize=18)
    axes.tick_params(axis='x', rotation=45)
    axes.grid(False)
    axes.legend(fontsize=16, framealpha=0.0)
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.rcdefaults()
    return fig if return_fig else axes


def plot_parameter_distributions(
    parameters_dict: dict,
    labels_dict: Optional[dict] = None,
    ranges_dict: Optional[dict] = None,
    bins: int = 200,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    color: Optional[str] = None,
    alpha: float = 0.8,
    figsize: Tuple[float, float] = (14.5, 12)
) -> plt.Figure:
    """Plot distributions for multiple parameters in a 2x2 grid.
    Args:
        parameters_dict (dict): Dictionary mapping parameter names to value arrays
        labels_dict (Optional[dict]): Dictionary mapping parameter names to LaTeX labels. If None, uses PARAMETER_LABELS.
        ranges_dict (Optional[dict]): Dictionary mapping parameter names to (min, max) tuples
        bins (int): Number of histogram bins
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme ("white" or "black")
        font_family (str): Font family to use
        font_name (str): Specific font name
        color (Optional[str]): Color for the histogram. If None, uses SIGNAL_COLOUR
        alpha (float): Transparency of the histogram dots
        figsize (Tuple[float, float]): Figure size in inches
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    if color is None:
        color = SIGNAL_COLOUR
    axis_color = "white" if background == "black" else "black"

    for idx, (param_name, values) in enumerate(parameters_dict.items()):
        if idx >= 4:
            break
        ax = axes[idx]
        if isinstance(values, list):
            values = np.array(values)

        if ranges_dict and param_name in ranges_dict:
            hist_range = ranges_dict[param_name]
        else:
            hist_range = (min(values), max(values))

        counts, bin_edges = np.histogram(values, bins=bins, range=hist_range)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Build stacked dot coordinates: for each bin, one dot per count at y = 0, 1, 2, ...
        x_dots = np.repeat(bin_centers, counts)
        y_dots = np.concatenate([np.arange(c) for c in counts]) if counts.sum() > 0 else np.array([])

        ax.scatter(
            x_dots, y_dots,
            color=color,
            alpha=alpha,
            s=0.25,
            edgecolor='none'
        )

        if param_name == "beta1_IC_b":
            ax.axvline(0.06, color='grey', linestyle='--', linewidth=1)
            ax.axvline(0.17, color='grey', linestyle='--', linewidth=1)
            ax.text(
                0.03, 0.98, "Slow rotation",
                rotation=90, ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9, color="#6baed6", weight="bold"
            )
            ax.text(
                0.115, 0.98, "Rapid rotation",
                rotation=90, ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9, color="#fdae6b", weight="bold"
            )
            ax.text(
                0.20, 0.98, "Extreme rotation",
                rotation=90, ha="center", va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9, color="#de2d26", weight="bold"
            )

        if labels_dict and param_name in labels_dict:
            param_label = labels_dict[param_name]
        elif param_name in PARAMETER_LABELS:
            param_label = PARAMETER_LABELS[param_name]
        else:
            param_label = param_name
        ax.set_xlabel(param_label, size=11, weight="bold")
        ax.set_xlim(hist_range[0], hist_range[1])
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=11)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.grid(False)

        # Remove the box border, keep only left/bottom axes lines, disconnected at the corner
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(axis_color)
        ax.spines['bottom'].set_color(axis_color)
        ax.spines['left'].set_position(('outward', 8))
        ax.spines['bottom'].set_position(('outward', 8))

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background == "black"))
    plt.rcdefaults()
    return fig


def plot_pp_coverage(
    posterior_samples_list: list,
    true_params_list: list,
    param_names: list,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    figsize: Tuple[float, float] = (12, 12),
    transparent: bool = False,
    fontsize_title: float = 16,
    fontsize_tick: float = 11
) -> plt.Figure:
    """Plot credible interval coverage (p-p plot) for multiple parameters.
    
    For each credible interval level (e.g., 68%, 95%), this plot shows the fraction of
    true parameter values that fall within that interval (empirical) vs the theoretical
    expectation. Each parameter is represented as a line.
    
    Args:
        posterior_samples_list (list): List of posterior sample arrays, each shape (num_samples, num_params)
        true_params_list (list): List of true parameter values, each shape (num_params,)
        param_names (list): List of parameter names
        fname (Optional[str]): Filename to save plot
        background (str): Background color theme ("white" or "black")
        font_family (str): Font family to use
        font_name (str): Specific font name
        figsize (Tuple[float, float]): Figure size in inches
        n_credible_levels (int): Number of credible interval levels to evaluate
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)

    palette = load_cmap("Fun", cmap_type="continuous")
    
    fig, ax = plt.subplots(figsize=(figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES))
    
    # Credible interval levels to evaluate (0-100%)
    n_credible_levels = 100
    credible_levels = np.linspace(0.01, 0.99, n_credible_levels)
    
    # For each parameter, calculate empirical coverage
    num_params = len(param_names)

    # Keep colors stable by parameter name; only unknown parameters use a fallback palette.
    unknown_params = [name for name in param_names if name not in PARAMETER_COLOURS]
    if len(unknown_params) == 1:
        fallback_colors = [palette(0.5)]
    elif len(unknown_params) > 1:
        fallback_colors = [palette(x) for x in np.linspace(0.05, 0.95, len(unknown_params))]
    else:
        fallback_colors = []
    fallback_color_map = dict(zip(unknown_params, fallback_colors))

    for param_idx in range(num_params):
        empirical_coverage = []
        
        # For each credible level
        for level in credible_levels:
            # Calculate the quantiles for this credible level
            lower_quantile = (1 - level) / 2
            upper_quantile = 1 - lower_quantile
            
            n_in_interval = 0
            total = 0
            
            # Check how many true values fall within their credible intervals
            for posterior_samples, true_params in zip(posterior_samples_list, true_params_list):
                if isinstance(posterior_samples, torch.Tensor):
                    posterior_samples = posterior_samples.cpu().numpy()
                if isinstance(true_params, torch.Tensor):
                    true_params = true_params.cpu().numpy()
                
                # Get the posterior samples for this parameter
                param_posterior = posterior_samples[:, param_idx]
                true_value = true_params[param_idx]
                
                # Calculate credible interval
                lower = np.quantile(param_posterior, lower_quantile)
                upper = np.quantile(param_posterior, upper_quantile)
                
                # Check if true value is within interval
                if lower <= true_value <= upper:
                    n_in_interval += 1
                total += 1
            
            # Empirical fraction
            empirical_coverage.append(n_in_interval / total if total > 0 else 0)
        
        # Plot line for this parameter
        param_label = PARAMETER_LABELS.get(param_names[param_idx], param_names[param_idx])
        param_name = param_names[param_idx]
        if param_name in PARAMETER_COLOURS:
            line_color = PARAMETER_COLOURS[param_name]
        else:
            line_color = fallback_color_map[param_name]
        ax.plot(credible_levels, np.array(empirical_coverage), 
            color=line_color, linewidth=2.5, label=param_label, alpha=0.8)
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], color='gray', linewidth=2, linestyle='--', label='Perfect Calibration', alpha=0.6)
    
    # Formatting
    ax.set_xlabel('Probability within the Credible Interval', size=fontsize_title)
    ax.set_ylabel(r'Fraction of events within the Credible Interval', size=fontsize_title)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    ax.tick_params(labelsize=fontsize_tick)
    ax.grid(True, alpha=0.3)
    legend_alpha = 0 if transparent else 0.95
    ax.legend(fontsize=fontsize_tick, loc='lower right', framealpha=legend_alpha)
    
    plt.tight_layout()
    if fname:
        savefig_kwargs = {"dpi": 300, "bbox_inches": "tight", "transparent": transparent}
        if not transparent:
            savefig_kwargs["facecolor"] = background
        plt.savefig(fname, **savefig_kwargs)
    
    plt.rcdefaults()
    return fig


def plot_epoch_sky_parameters(
    dataset,
    sky_params: list,
    fname: str,
    background: str = "black",
    color: str = "#3498db",
    bins: int = 40,
    figsize: Tuple[float, float] = (30, 25)
) -> None:
    """Plot sky parameter distributions from a dataset in a 2x2 grid.
    
    Args:
        dataset: Dataset with .parameters attribute (e.g., hThetaMulti)
        sky_params (list): List of sky parameter names to plot (e.g., ["ra", "dec", "d", "psi"])
        fname (str): Filename to save plot
        background (str): Background color ("black" or "white")
        color (str): Color for histogram bars
        bins (int): Number of histogram bins
    """
    num_sky_params = len(sky_params)
    if num_sky_params == 0:
        return
    
    # Create 2x2 grid
    n_rows, n_cols = 2, 2
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.patch.set_facecolor(background)
    axes = axes.flatten()
    
    # Plot each sky parameter
    for i, param_name in enumerate(sky_params):
        if i >= 4:  # Only 2x2 grid
            break
        
        # Sky parameters are at the end: [intrinsic_params..., ra, dec, d, psi]
        param_idx = dataset.parameters.shape[1] - len(sky_params) + i
        if param_idx < dataset.parameters.shape[1]:
            values = dataset.parameters[:, param_idx]
            ax = axes[i]
            ax.hist(values, bins=bins, color=color, alpha=0.7, edgecolor='white')
            ax.set_xlabel(param_name, fontsize=12, color='white', fontfamily='sans-serif')
            ax.set_ylabel('Count', fontsize=12, color='white', fontfamily='sans-serif')
            ax.set_facecolor(background)
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for i in range(num_sky_params, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(fname, facecolor=background, edgecolor='none', dpi=300)
    plt.close()


def plot_corner(samples_cpu, true_param, background="black", fname="plots/corner_plot.png", dataset=None, 
                labels=None, ranges=None, font_family="sans-serif", font_name="Avenir", figsize=(14.5, 14.5), fontsize_title=16, fontsize_tick=12):
    """Plot corner plot of parameter posterior distribution.
    
    Args:
        samples_cpu (np.ndarray): Posterior samples as numpy array, shape (num_samples, num_params)
        true_param (np.ndarray): True parameter values as numpy array, shape (num_params,)
        background (str): Background color ("black" or "white")
        fname (str): Filename to save plot
        dataset: Optional dataset object
        labels (list): Optional custom labels for parameters. If None, will be inferred from dataset or num_params
        ranges (list): Optional custom ranges for parameters as list of tuples [(min, max), ...]
        font_family (str): Font family to use
        font_name (str): Specific font name
    """
    set_plot_style(background, font_family, font_name)
    
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
            # CCSN data with 4 parameters (use unified PARAMETER_LABELS)
            labels = [
                PARAMETER_LABELS['beta1_IC_b'],
                PARAMETER_LABELS['omega_0(rad|s)'],
                PARAMETER_LABELS['A(km)'],
                PARAMETER_LABELS['Ye_c_b'],
            ]
        elif num_params == 5:
            # CCSN data with 5 parameters (intrinsic + 1 extrinsic like psi)
            labels = [
                PARAMETER_LABELS['beta1_IC_b'],
                PARAMETER_LABELS['omega_0(rad|s)'],
                PARAMETER_LABELS['A(km)'],
                PARAMETER_LABELS['Ye_c_b'],
                PARAMETER_LABELS['psi'],
            ]
        elif num_params == 8:
            # Full dataset with all parameters
            param_names = ['beta1_IC_b', 'omega_0(rad|s)', 'A(km)', 'Ye_c_b', 'ra', 'dec', 'd', 'psi']
            labels = [PARAMETER_LABELS.get(name, name) for name in param_names]
        else:
            # Generic labels for other cases
            labels = [f"Parameter {i+1}" for i in range(num_params)]
    
    # Set default ranges if not provided
    if ranges is None:
        if num_params == 2:
            ranges = [(-3, 3), (-3, 3)]
        elif num_params == 4:
            # Use unified PARAMETER_RANGES for 4 parameters
            param_names = ['beta1_IC_b', 'omega_0(rad|s)', 'A(km)', 'Ye_c_b']
            ranges = [PARAMETER_RANGES.get(name) for name in param_names]
        elif num_params == 5:
            # 5 parameters (intrinsic + psi)
            param_names = ['beta1_IC_b', 'omega_0(rad|s)', 'A(km)', 'Ye_c_b', 'psi']
            ranges = [PARAMETER_RANGES.get(name) for name in param_names]
        elif num_params == 8:
            # Full dataset with all parameters
            param_names = ['beta1_IC_b', 'omega_0(rad|s)', 'A(km)', 'Ye_c_b', 'ra', 'dec', 'd', 'psi']
            ranges = [PARAMETER_RANGES.get(name) for name in param_names]
        # Otherwise ranges will be None and corner will auto-determine
    
    # Set rcParams based on background color
    if background == "black":
        text_color = 'white'
        axes_color = 'black'
        patch_color = 'white'
        spine_color = 'white'
        transparent = True
    else:  # white background
        text_color = 'black'
        axes_color = 'white'
        patch_color = 'black'
        spine_color = 'black'
        transparent = False
    
    plt.rcParams['figure.facecolor'] = axes_color
    plt.rcParams['axes.facecolor'] = axes_color
    plt.rcParams['savefig.facecolor'] = axes_color
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color

    # Special case for single parameter - corner library has issues with this
    if num_params == 1:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES))
        ax.hist(samples_cpu.flatten(), bins=100, color=GENERATED_SIGNAL_COLOUR, 
                alpha=0.7, density=True, edgecolor='none')
        if true_param is not None and len(true_param) > 0:
            ax.axvline(true_param[0], color=SIGNAL_COLOUR, linewidth=2, label='True value')
        ax.set_xlabel(labels[0] if labels else 'Parameter', fontsize=fontsize_title, color=text_color)
        ax.set_ylabel('Count', fontsize=fontsize_title, color=text_color)
        if ranges is not None and ranges[0] is not None:
            ax.set_xlim(ranges[0])
        ax.tick_params(labelsize=fontsize_tick, colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
        ax.set_facecolor(axes_color)
        fig.patch.set_facecolor(axes_color)
        
        # Disable rasterization on all artist objects
        for line in ax.lines:
            line.set_rasterized(False)
        for patch in ax.patches:
            patch.set_rasterized(False)
        for collection in ax.collections:
            collection.set_rasterized(False)
        for image in ax.images:
            image.set_rasterized(False)
        
        # Add title with quantiles
        q = np.percentile(samples_cpu.flatten(), [16, 50, 84])
        title = f"{q[1]:.4f}$_{{-{q[1]-q[0]:.4f}}}^{{+{q[2]-q[1]:.4f}}}$"
        ax.set_title(title, fontsize=fontsize_tick, color=text_color)
        
        # Determine format from filename extension and configure for SVG vector output
        file_format = None
        if fname.lower().endswith('.svg'):
            file_format = 'svg'
            # Temporarily set rcParams to prevent rasterization in SVG
            old_rasterized = plt.rcParams.get('image.composite_image', None)
            plt.rcParams['image.composite_image'] = False
        
        save_kwargs = {
            "dpi": 300,
            "bbox_inches": 'tight',
            "transparent": transparent,
        }
        if file_format:
            save_kwargs["format"] = file_format
        
        plt.savefig(fname, **save_kwargs)
        
        # Restore old rcParams if they were changed
        if file_format == 'svg' and old_rasterized is not None:
            plt.rcParams['image.composite_image'] = old_rasterized
        
        plt.show()
        return

    corner_kwargs = {
        'labels': labels,
        'truths': true_param[:num_params],
        'truth_color': SIGNAL_COLOUR,
        'show_titles': True,
        'title_quantiles': [0.16, 0.5, 0.84],
        'title_fmt': '.2f',
        'title_kwargs': {'fontsize': fontsize_tick},
        'label_kwargs': {'fontsize': fontsize_title},
        'bins': 100,
        'smooth': 3,
        'color': GENERATED_SIGNAL_COLOUR,
        'hist_kwargs': {'density': False, 'alpha': 1.0},
        'levels': (0.68, 0.95),
        'fill_contours': True,
        'plot_datapoints': False,
        'fig': plt.figure(figsize=(figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES))
    }
    
    # Add range only if specified
    if ranges is not None:
        corner_kwargs['range'] = ranges
    
    figure = corner.corner(samples_cpu, **corner_kwargs)

    # Manually set font size and family on the per-panel titles (mean ± sd text)
    for ax in figure.get_axes():
        if ax.get_title():  # only diagonal panels have titles
            ax.title.set_fontsize(fontsize_tick)
            ax.title.set_fontfamily(font_name)
            ax.title.set_color(text_color)

    # Fill hist patches with appropriate color
    for ax in figure.get_axes():
        for patch in ax.patches:
            patch.set_facecolor(patch_color)
            patch.set_alpha(1.0)

    # Make axis lines and adjust tick labels with appropriate colors
    for ax in figure.get_axes():
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
        # Axis tick numbers
        ax.tick_params(labelsize=12)
        # Reduce label padding to save space
        ax.xaxis.labelpad = 2
        ax.yaxis.labelpad = 2
        
        # Fix rasterization: disable rasterization for ALL artist objects
        # This includes collections (contourf), lines, patches, and images
        for collection in ax.collections:
            collection.set_rasterized(False)
        for line in ax.lines:
            line.set_rasterized(False)
        for patch in ax.patches:
            patch.set_rasterized(False)
        for image in ax.images:
            image.set_rasterized(False)

    # Set figure background
    figure.patch.set_facecolor(axes_color)

    # Reduce spacing between subplots to make plots bigger
    figure.subplots_adjust(hspace=0.05, wspace=0.05)
    
    # Determine format from filename extension and configure for SVG vector output
    file_format = None
    if fname.lower().endswith('.svg'):
        file_format = 'svg'
        # Temporarily set rcParams to prevent rasterization in SVG
        old_rasterized = plt.rcParams.get('image.composite_image', None)
        plt.rcParams['image.composite_image'] = False
    
    save_kwargs = {
        "dpi": 300,
        "bbox_inches": 'tight',
        "transparent": transparent,
    }
    if file_format:
        save_kwargs["format"] = file_format
    
    plt.savefig(fname, **save_kwargs)
    
    # Restore old rcParams if they were changed
    if file_format == 'svg' and old_rasterized is not None:
        plt.rcParams['image.composite_image'] = old_rasterized
    plt.show()


def plot_eos_ye_distribution(
    eos_values: np.ndarray,
    ye_values: np.ndarray,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    alpha: float = 0.7,
    point_size: float = 50,
    figsize: Tuple[float, float] = (14.5, 8)
) -> plt.Figure:
    """Create a violin plot of Ye values across different EOS types.
    
    Shows the full distribution of electron fraction (Ye) for each Equation of State
    using violin plots with individual points overlaid.
    
    Args:
        eos_values (np.ndarray): Array of EOS values (categorical, strings)
        ye_values (np.ndarray): Array of Ye values (continuous)
        fname (Optional[str]): Filename to save plot
        background (str): Background color ("white" or "black")
        font_family (str): Font family to use
        font_name (str): Specific font name
        figsize (Tuple[float, float]): Figure size in inches
        jitter_strength (float): Amount of horizontal jitter for points
        alpha (float): Transparency of violin fill
        point_size (float): Size of individual points
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    
    # Prepare data for plotting
    df_plot = pd.DataFrame({
        'EOS': eos_values.astype(str),
        'Ye': ye_values
    })
    
    # Sort EOS by mean Ye for better visualization
    eos_order = df_plot.groupby('EOS')['Ye'].mean().sort_values().index.tolist()
    
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    fig, ax = plt.subplots(figsize=figsize)
    
    # Overlay individual points (one per EOS, no distribution to show)
    sns.stripplot(
        data=df_plot,
        x='EOS',
        y='Ye',
        order=eos_order,
        ax=ax,
        size=point_size / 10,  # Scale down seaborn point size
        color=SIGNAL_COLOUR,
        alpha=alpha,
        jitter=False
    )
    
    # Formatting
    ax.set_xlabel('Equation of State (EOS)', fontsize=16, fontfamily=font_name)
    ax.set_ylabel(PARAMETER_LABELS['Ye_c_b'], fontsize=16, fontfamily=font_name)
    ax.tick_params(labelsize=7, axis='x')
    ax.tick_params(labelsize=11, axis='y')
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight', transparent=(background == "black"))
    
    plt.rcdefaults()
    return fig


def plot_eos_ye_posterior_distribution(
    samples_ye: np.ndarray,
    true_ye: float,
    true_eos: str,
    dataset_ye: list,
    dataset_eos: list,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    figsize: Tuple[float, float] = (14.5, 8),
    alpha: float = 0.7,
    point_size: float = 50
) -> plt.Figure:
    """Plot EOS-Ye distribution with posterior overlay and marginal posterior.
    
    Creates a violin plot of Ye across EOS types from the dataset, overlays posterior
    samples, and shows the marginal posterior distribution of Ye on the left side.
    
    Args:
        samples_ye (np.ndarray): Posterior samples for Ye
        true_ye (float): True Ye value
        true_eos (str): True EOS type
        dataset_ye (list): List of [training_ye, validation_ye] arrays
        dataset_eos (list): List of [training_eos, validation_eos] arrays
        fname (Optional[str]): Filename to save plot
        background (str): Background color ("white" or "black")
        font_family (str): Font family to use
        font_name (str): Specific font name
        figsize (Tuple[float, float]): Figure size in inches
        alpha (float): Transparency of violin fill
        point_size (float): Size of individual points
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    
    # Combine training and validation dataset
    combined_ye = np.concatenate(dataset_ye)
    combined_eos = np.concatenate([eos.astype(str) for eos in dataset_eos])
    
    # Prepare data for plotting
    df_plot = pd.DataFrame({
        'EOS': combined_eos,
        'Ye': combined_ye,
        'Source': 'Dataset'
    })
    
    # Prepare posterior samples
    df_posterior = pd.DataFrame({
        'EOS': [str(true_eos)] * len(samples_ye),
        'Ye': samples_ye,
        'Source': 'Posterior'
    })
    
    # Sort EOS by mean Ye for better visualization
    eos_order = df_plot.groupby('EOS')['Ye'].mean().sort_values().index.tolist()
    
    # Create figure with GridSpec for marginal plot (swapped: marginal on left 1/4, main on right 3/4)
    from matplotlib.gridspec import GridSpec
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, width_ratios=[1, 3], wspace=0.3)
    ax_marginal = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    
    # Marginal plot: KDE of posterior Ye with true value marked
    from scipy.stats import gaussian_kde
    
    # Create KDE of posterior samples
    kde = gaussian_kde(samples_ye)
    ye_range = np.linspace(np.min(samples_ye) - 0.005, np.max(samples_ye) + 0.005, 200)
    kde_values = kde(ye_range)
    
    # Plot KDE as line only (no fill)
    ax_marginal.plot(kde_values, ye_range, color='red', linewidth=2)

    # Mark true Ye value with a horizontal line
    ax_marginal.axhline(true_ye, color=SIGNAL_COLOUR, linewidth=2.5)
    
    # Marginal plot formatting
    ax_marginal.set_xlabel('Count', fontsize=16)
    ax_marginal.set_ylabel(PARAMETER_LABELS['Ye_c_b'], fontsize=16)
    ax_marginal.tick_params(labelsize=7, axis='x')
    # ax_marginal.tick_params(labelsize=11, axis='y')    
    ax_marginal.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax_marginal.set_axisbelow(True)
    
    # Main plot: dataset points (no violin plot density)
    # Overlay dataset points with jitter
    sns.stripplot(
        data=df_plot,
        x='EOS',
        y='Ye',
        order=eos_order,
        ax=ax_main,
        size=point_size / 10,
        color='black',
        alpha=0.2,
        jitter=False
    )
    
    # Highlight true EOS with background box
    true_eos_idx = eos_order.index(str(true_eos))
    ax_main.axvspan(true_eos_idx - 0.45, true_eos_idx + 0.45, 
                    alpha=0.15, color=SIGNAL_COLOUR, zorder=0)
    
    # Calculate and plot credible interval bands for posterior
    if len(samples_ye) > 0:
        # Sort samples in descending order and compute CDF
        sorted_samples = np.sort(samples_ye)[::-1]
        cdf = np.cumsum(sorted_samples) / np.sum(sorted_samples)
        
        # Define credible levels and corresponding alpha values
        credible_levels = [0.50, 0.68, 0.95]
        band_alphas = [0.3, 0.2, 0.1]
        
        # Find thresholds for each credible level
        thresholds = []
        for level in credible_levels:
            idx = np.searchsorted(cdf, level, side="left")
            idx = min(idx, len(sorted_samples) - 1)
            thresholds.append(float(sorted_samples[idx]))
        
        # Sort thresholds and plot bands from widest to narrowest
        thresholds_sorted = np.sort(thresholds)
        
        # Plot bands: widest (95%) first, then 68%, then 50%
        for i, (level, threshold) in enumerate(zip([0.95, 0.68, 0.50], thresholds_sorted)):
            # Find corresponding quantiles
            lower_q = (1 - level) / 2
            upper_q = 1 - lower_q
            lower_ye = np.quantile(samples_ye, lower_q)
            upper_ye = np.quantile(samples_ye, upper_q)
            
            ax_main.axhspan(lower_ye, upper_ye, alpha=0.15 - i*0.04, color='red', zorder=1)
    
    # Add true Ye line to main plot
    ax_main.axhline(true_ye, color=SIGNAL_COLOUR, linewidth=2.5)

    # Add a dot marking the true value at the true EOS location
    ax_main.scatter(true_eos_idx, true_ye, color=SIGNAL_COLOUR, s=80, zorder=5, edgecolor='none')
    
    # Main plot formatting
    ax_main.set_xlabel('Equation of State (EOS)', fontsize=16)
    ax_main.tick_params(labelsize=7, axis='x')

    # Remove y-axis label and ticks on the main plot (already shown on the marginal plot)
    ax_main.set_ylabel('')
    ax_main.tick_params(axis='y', left=False, labelleft=False)
    
    # Highlight true EOS on x-axis with SIGNAL_COLOUR
    ax_main_xticklabels = ax_main.get_xticklabels()
    for i, label in enumerate(ax_main_xticklabels):
        if label.get_text() == str(true_eos):
            label.set_color(SIGNAL_COLOUR)
            label.set_weight('bold')
    
    plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right')
    ax_main.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax_main.set_axisbelow(True)
    
    # Sync y-axis limits between marginal and main
    ax_marginal.set_ylim(ax_main.get_ylim())
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches='tight', transparent=(background == "black"))
    
    plt.rcdefaults()
    return fig