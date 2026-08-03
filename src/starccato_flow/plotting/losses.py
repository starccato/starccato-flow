"""Loss plotting functions for training visualization."""

from typing import List, Optional
import matplotlib.pyplot as plt
from . import set_plot_style
from ..utils.defaults_plotting import (
    SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR, CM_TO_INCHES
)

def plot_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    loss_type: str = "Loss",
    train_label: Optional[str] = None,
    val_label: Optional[str] = None,
    fname: Optional[str] = None,
    axes: Optional[plt.Axes] = None,
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    fontsize_title: float = 16,
    fontsize_tick: float = 12,
    figsize: tuple[float, float] = (14.5,8)
):
    """Plot training and validation loss curves.
    
    Args:
        train_losses (List[float]): List of training loss values
        val_losses (Optional[List[float]]): List of validation loss values
        loss_type (str): Type of loss for y-axis label
        train_label (Optional[str]): Custom legend label for training loss
        val_label (Optional[str]): Custom legend label for validation loss
        fname (Optional[str]): Filename to save plot
        axes (Optional[plt.Axes]): Existing axes to plot on
        background (str): Background color theme
        font_family (str): Font family to use
        font_name (str): Specific font name
    
    Returns:
        plt.Axes: The plot axes
    """
    set_plot_style(background, font_family, font_name)
    
    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    if axes is None:
        fig = plt.figure(figsize=figsize)
        axes = fig.gca()
    
    if train_label is None:
        train_label = "Training Loss"
    if val_label is None:
        val_label = "Validation Loss"
    
    axes.plot(train_losses, label=train_label, color=SIGNAL_COLOUR, 
              linewidth=2, alpha=1.0, linestyle='-')
    
    if val_losses is not None:
        axes.plot(val_losses, label=val_label, color=GENERATED_SIGNAL_COLOUR, 
                  linewidth=2, alpha=1.0, linestyle='-')
    
    axes.set_xlabel("Epoch", size=fontsize_tick)
    axes.set_ylabel(loss_type, size=fontsize_tick)
    axes.set_xlim(0, len(train_losses) - 1)
    axes.set_ylim(0, max(max(train_losses), max(val_losses) if val_losses is not None else 0) * 1.1)
    axes.legend(fontsize=fontsize_tick, framealpha=0.0)
    axes.tick_params(labelsize=fontsize_tick)
    axes.grid(False)
    
    # Set y-axis ticks to 0.005 increments
    # axes.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
    
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight", transparent=(background=="black"))
    
    plt.rcdefaults()

def plot_gradients(
    encoder_gradients: List[float],
    decoder_gradients: List[float],
    q_gradients: List[float],
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    figsize: tuple[float, float] = (25, 46)
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

    figsize = (figsize[0] / CM_TO_INCHES, figsize[1] / CM_TO_INCHES)
    fig, axes = plt.subplots(3, 1, figsize=figsize)
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
