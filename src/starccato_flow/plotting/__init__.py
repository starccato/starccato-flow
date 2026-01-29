"""Plotting utilities for starccato_flow.

This module provides various plotting functions organized by category:
- signals: Signal visualization functions
- losses: Loss plotting functions
- parameters: Parameter distribution plots
- latent: Latent space visualization and morphing
- analysis: Advanced analysis plots (corner plots, p-p plots, animations)

Common utility functions are exposed at the top level.
"""

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from ..utils.plotting_defaults import DEFAULT_FONT_SIZE

# Utility functions
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


# Import plotting functions from submodules
from .signals import (
    plot_signal_grid,
    plot_candidate_signal,
    plot_reconstruction,
    plot_single_signal,
    plot_signal_distribution
)

from .losses import (
    plot_loss,
    plot_individual_loss,
    plot_training_validation_loss,
    plot_gradients
)

from .parameters import (
    plot_parameter_distribution,
    plot_parameter_distribution_grid
)

from .latent import (
    plot_latent_morphs,
    plot_latent_morph_grid,
    animate_latent_morphs,
    plot_latent_morph_up_and_down,
    create_latent_morph_gif,
    plot_latent_space_3d
)

from .analysis import (
    plot_reconstruction_distribution,
    p_p_plot,
    plot_corner,
    create_signal_grid_gif,
    create_snr_variation_gif,
    plot_sky_localisation
)

__all__ = [
    # Utility functions
    'set_plot_style',
    'get_time_axis',
    # Signal plotting
    'plot_signal_grid',
    'plot_candidate_signal',
    'plot_reconstruction',
    'plot_single_signal',
    'plot_signal_distribution',
    # Loss plotting
    'plot_loss',
    'plot_individual_loss',
    'plot_training_validation_loss',
    'plot_gradients',
    # Parameter plotting
    'plot_parameter_distribution',
    'plot_parameter_distribution_grid',
    # Latent space
    'plot_latent_morphs',
    'plot_latent_morph_grid',
    'animate_latent_morphs',
    'plot_latent_morph_up_and_down',
    'create_latent_morph_gif',
    'plot_latent_space_3d',
    # Analysis
    'plot_reconstruction_distribution',
    'p_p_plot',
    'plot_corner',
    'create_signal_grid_gif',
    'create_snr_variation_gif',
    'plot_sky_localisation'
]
