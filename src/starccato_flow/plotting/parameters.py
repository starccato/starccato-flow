"""Parameter distribution plotting functions."""

from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import corner
from scipy import stats
from . import set_plot_style
from ..utils.plotting_defaults import (
    SIGNAL_COLOUR,
    GENERATED_SIGNAL_COLOUR,
    PARAMETER_LABELS,
    PARAMETER_RANGES
)


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
    figsize: Tuple[float, float] = (12, 10)
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
        alpha (float): Transparency of the histogram bars
        figsize (Tuple[float, float]): Figure size in inches
    
    Returns:
        plt.Figure: The figure object
    """
    set_plot_style(background, font_family, font_name)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    if color is None:
        color = SIGNAL_COLOUR
    
    for idx, (param_name, values) in enumerate(parameters_dict.items()):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        if isinstance(values, list):
            values = np.array(values)
        
        n, bins_edges, patches = ax.hist(
            values, 
            bins=bins, 
            color=color, 
            alpha=alpha, 
            edgecolor='none'
        )
        
        if labels_dict and param_name in labels_dict:
            param_label = labels_dict[param_name]
        elif param_name in PARAMETER_LABELS:
            param_label = PARAMETER_LABELS[param_name]
        else:
            param_label = param_name
        
        ax.set_xlabel(param_label, size=16)
        ax.set_ylabel("Count", size=16)
        ax.set_title(param_label, size=18, pad=10)
        
        if ranges_dict and param_name in ranges_dict:
            ax.set_xlim(ranges_dict[param_name][0], ranges_dict[param_name][1])
        else:
            ax.set_xlim(min(values), max(values))
        
        ax.tick_params(labelsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)
    
    for idx in range(len(parameters_dict), 4):
        axes[idx].axis('off')
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
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
    figsize: Tuple[float, float] = (10, 8),
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Credible interval levels to evaluate (0-100%)
    n_credible_levels = 100
    credible_levels = np.linspace(0.01, 0.99, n_credible_levels)
    
    # Define colors for each parameter
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # For each parameter, calculate empirical coverage
    num_params = len(param_names)
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
        color = colors[param_idx % len(colors)]
        ax.plot(credible_levels, np.array(empirical_coverage), 
                color=color, linewidth=2.5, label=param_label, alpha=0.8)
    
    # Plot diagonal (perfect calibration)
    ax.plot([0, 1], [0, 1], color='gray', linewidth=2, linestyle='--', label='Perfect Calibration', alpha=0.6)
    
    # Formatting
    ax.set_xlabel('Probability within the Credible Interval', size=16)
    ax.set_ylabel(r'Fraction of events within the Credible Interval', size=16)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.95)
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor=background if background == "black" else "white")
    
    plt.rcdefaults()
    return fig



def plot_epoch_sky_parameters(
    dataset,
    sky_params: list,
    fname: str,
    background: str = "black",
    color: str = "#3498db",
    bins: int = 40,
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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
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
    plt.savefig(fname, facecolor=background, edgecolor='none', dpi=150)
    plt.close()


def plot_corner(samples_cpu, true_param, background="black", fname="plots/corner_plot.png", dataset=None, 
                labels=None, ranges=None, font_family="sans-serif", font_name="Avenir"):
    """Plot corner plot of parameter posterior distribution.
    
    Args:
        samples_cpu (np.ndarray): Posterior samples as numpy array, shape (num_samples, num_params)
        true_param (np.ndarray): True parameter values as numpy array, shape (num_params,)
        background (str): Background color ("black" or "white")
        fname (str): Filename to save plot
        dataset: Optional dataset object (CCSNData or sThetaToy) to extract parameter metadata
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
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(samples_cpu.flatten(), bins=100, color=GENERATED_SIGNAL_COLOUR, 
                alpha=0.7, density=True, edgecolor='none')
        if true_param is not None and len(true_param) > 0:
            ax.axvline(true_param[0], color=SIGNAL_COLOUR, linewidth=2, label='True value')
        ax.set_xlabel(labels[0] if labels else 'Parameter', fontsize=24, color=text_color)
        ax.set_ylabel('Density', fontsize=24, color=text_color)
        if ranges is not None and ranges[0] is not None:
            ax.set_xlim(ranges[0])
        ax.tick_params(labelsize=12, colors=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(spine_color)
        ax.set_facecolor(axes_color)
        fig.patch.set_facecolor(axes_color)
        
        # Add title with quantiles
        q = np.percentile(samples_cpu.flatten(), [16, 50, 84])
        title = f"{q[1]:.4f}$_{{-{q[1]-q[0]:.4f}}}^{{+{q[2]-q[1]:.4f}}}$"
        ax.set_title(title, fontsize=12, color=text_color)
        
        plt.savefig(fname, dpi=300, bbox_inches='tight', transparent=transparent)
        plt.show()
        return

    corner_kwargs = {
        'labels': labels,
        'truths': true_param[:num_params],
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

    # Set figure background
    figure.patch.set_facecolor(axes_color)

    # Reduce spacing between subplots to make plots bigger
    figure.subplots_adjust(hspace=0.05, wspace=0.05)
    
    plt.savefig(fname, dpi=300, bbox_inches='tight', transparent=transparent)
    plt.show()


def plot_eos_ye_distribution(
    eos_values: np.ndarray,
    ye_values: np.ndarray,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    figsize: Tuple[float, float] = (16, 8),
    jitter_strength: float = 0.15,
    alpha: float = 0.7,
    point_size: float = 50
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
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin plot
    sns.violinplot(
        data=df_plot,
        x='EOS',
        y='Ye',
        hue='EOS',
        order=eos_order,
        palette='coolwarm',
        ax=ax,
        inner=None,  # Remove inner details for cleaner look
        alpha=alpha,
        legend=False
    )
    
    # Overlay individual points with jitter
    sns.stripplot(
        data=df_plot,
        x='EOS',
        y='Ye',
        order=eos_order,
        ax=ax,
        size=point_size / 20,  # Scale down seaborn point size
        color='black',
        alpha=0.4,
        jitter=True
    )
    
    # Formatting
    ax.set_xlabel('Equation of State (EOS)', fontsize=16)
    ax.set_ylabel(PARAMETER_LABELS['Ye_c_b'], fontsize=16)
    ax.tick_params(labelsize=12, axis='x')
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
