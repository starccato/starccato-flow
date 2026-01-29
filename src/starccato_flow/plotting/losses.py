"""Loss plotting functions for training visualization."""

from typing import List, Optional, Union
import matplotlib.pyplot as plt
from . import set_plot_style
from ..utils.plotting_defaults import (
    SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR, LATENT_SPACE_COLOUR
)


def plot_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    loss_type: str = "Loss",
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
        loss_type (str): Type of loss for y-axis label
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
    
    axes.plot(train_losses, label="Training Loss", color=SIGNAL_COLOUR, 
              linewidth=3, alpha=1.0, linestyle='-')
    
    if val_losses is not None:
        axes.plot(val_losses, label="Validation Loss", color=GENERATED_SIGNAL_COLOUR, 
                  linewidth=3, alpha=1.0, linestyle='-')
    
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel(loss_type, size=20)
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


def plot_gradients(
    encoder_gradients: List[float],
    decoder_gradients: List[float],
    q_gradients: List[float],
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> tuple:
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
        tuple: Figure and list of axes
    """
    set_plot_style(background, font_family, font_name)

    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    colors = ["deepskyblue", GENERATED_SIGNAL_COLOUR, "green"]
    
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
