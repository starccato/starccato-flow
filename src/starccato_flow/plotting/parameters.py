"""Parameter distribution plotting functions."""

from typing import Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import corner
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
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
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
        
        mean_val = np.mean(values)
        ax.axvline(mean_val, color=GENERATED_SIGNAL_COLOUR, linewidth=2.5, linestyle='--')
        
        if labels_dict and param_name in labels_dict:
            param_label = labels_dict[param_name]
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


def plot_corner(samples_cpu, true_params, fname="plots/corner_plot.png", dataset=None, 
                labels=None, ranges=None):
    """Plot corner plot of parameter posterior distribution.
    
    Args:
        samples_cpu (np.ndarray): Posterior samples as numpy array, shape (num_samples, num_params)
        true_params (np.ndarray): True parameter values as numpy array, shape (num_params,)
        fname (str): Filename to save plot
        dataset: Optional dataset object (CCSNData or sThetaToy) to extract parameter metadata
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
            # CCSN data with 4 parameters (use unified PARAMETER_LABELS)
            labels = [
                PARAMETER_LABELS.get('beta1_IC_b', r"$\beta_{IC,b}$"),
                PARAMETER_LABELS.get('omega_0(rad|s)', r"$\omega_0$"),
                PARAMETER_LABELS.get('A(km)', r"$A$"),
                PARAMETER_LABELS.get('Ye_c_b', r"$Y_{e,c,b}$"),
            ]
        elif num_params == 5:
            # CCSN data with 5 parameters (intrinsic + 1 extrinsic like psi)
            labels = [
                PARAMETER_LABELS.get('beta1_IC_b', r"$\beta_{IC,b}$"),
                PARAMETER_LABELS.get('omega_0(rad|s)', r"$\omega_0$"),
                PARAMETER_LABELS.get('A(km)', r"$A$"),
                PARAMETER_LABELS.get('Ye_c_b', r"$Y_{e,c,b}$"),
                PARAMETER_LABELS.get('psi', r"$\psi$"),
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
    
    plt.savefig(fname, dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
