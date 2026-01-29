"""Latent space visualization and morphing functions."""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
from PIL import Image
import io

from ..utils.defaults import DEVICE, TEN_KPC
from ..utils.plotting_defaults import (
    SIGNAL_COLOUR, 
    GENERATED_SIGNAL_COLOUR, 
    LATENT_SPACE_COLOUR,
    SIGNAL_LIM_UPPER, 
    SIGNAL_LIM_LOWER
)
from . import set_plot_style, get_time_axis


def plot_latent_morphs(
    model,
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
        model: VAE model for encoding/decoding
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
    model,
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
        model: VAE model for encoding/decoding
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
    model,
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
        model: VAE model for encoding/decoding
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
    """Plot a 2-panel figure showing signal interpolation and latent space path."""
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
    ax2.scatter(mean_2[0].cpu(), mean_2[1].cpu(), color='deepskyblue', label="Signal 2", alpha=0.75, edgecolors='none', s=100)
    ax2.scatter(mean_mid[0].cpu(), mean_mid[1].cpu(), color='red', label="Interpolant", edgecolors='none', s=100)
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
    plt.show()
    plt.rcdefaults()


def create_latent_morph_gif(
    model,
    train_dataset,
    signal_1_index: int,
    signal_2_index: int,
    max_value: float,
    num_frames: int = 30,
    background: str = "white",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    fname: str = "plots/latent_morph_animation.gif",
    duration: int = 100
):
    """Create an animated GIF showing interpolation between two signals in latent space.
    
    This function creates a smooth animation that moves back and forth between two signals,
    showing how the interpolation evolves in both signal space and latent space.
    
    Args:
        model: Trained VAE model with encoder and decoder
        train_dataset: Dataset to get signals and compute background latent scatter
        signal_1_index (int): Index of first signal in dataset
        signal_2_index (int): Index of second signal in dataset
        max_value (float): Maximum strain value for scaling signals
        num_frames (int): Number of frames in the animation (half cycle)
        background (str): Background color theme ("white" or "black")
        font_family (str): Font family to use
        font_name (str): Specific font name
        fname (str): Filename to save the GIF
        duration (int): Duration of each frame in milliseconds
    """
    # Get the two signals from the dataset
    signal_1 = train_dataset[signal_1_index][0].unsqueeze(0).to(DEVICE)
    signal_2 = train_dataset[signal_2_index][0].unsqueeze(0).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        # Encode signals
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)
        
        # Reconstruct signals
        signal_1_np = train_dataset.denormalise_signals(signal_1.cpu().detach().numpy().flatten()) / TEN_KPC
        signal_2_np = train_dataset.denormalise_signals(signal_2.cpu().detach().numpy().flatten()) / TEN_KPC
        
        # X-axis
        x_vals = [i / 4096 for i in range(0, 256)]
        x_vals = [x - (53 / 4096) for x in x_vals]
        
        # Posterior means for background latent scatter
        all_means = []
        for _, x, _ in train_dataset:
            x = torch.tensor(x).to(DEVICE)
            mean, _ = model.encoder(x)
            all_means.append(mean.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)
    
    # Create alpha values for interpolation (back and forth)
    alphas_forward = np.linspace(0, 1, num_frames)
    alphas_backward = np.linspace(1, 0, num_frames)
    alphas = np.concatenate([alphas_forward, alphas_backward])
    
    frames = []
    print(f"Creating {len(alphas)} frames for latent morph GIF...")
    
    for frame_idx, alpha in enumerate(alphas):
        with torch.no_grad():
            # Interpolate in latent space
            mean_interp = mean_1 * (1 - alpha) + mean_2 * alpha
            signal_interp = train_dataset.denormalise_signals(model.decoder(mean_interp).cpu().detach().numpy().flatten()) / TEN_KPC
        
        # Set up styling
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
            vline_color = 'black'
        
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.sans-serif'] = font_name
        plt.rcParams['font.size'] = 12
        
        mean_1_sq = mean_1.squeeze()
        mean_2_sq = mean_2.squeeze()
        mean_interp_sq = mean_interp.squeeze()
        
        # Set up figure with same dimensions as animate_latent_morphs
        fig = plt.figure(figsize=(10, 17))
        
        # Signal plot - show only interpolation
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(x_vals, signal_interp, color='red', linewidth=2, zorder=3)
        plt.axvline(x=0, color=vline_color, linestyle='dashed', alpha=0.5)
        ax1.set_ylim(SIGNAL_LIM_LOWER, SIGNAL_LIM_UPPER)
        ax1.set_xlim(left=x_vals[0], right=x_vals[-1])
        ax1.set_xlabel("time (s)", fontsize=16)
        ax1.set_ylabel("h", fontsize=16)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Latent space plot - show all with blue original signals and red interpolation
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.scatter(all_means[:, 0], all_means[:, 1], alpha=0.2, color='gray', edgecolors='none', s=50)
        ax2.plot(
            [mean_1_sq[0].cpu(), mean_2_sq[0].cpu()],
            [mean_1_sq[1].cpu(), mean_2_sq[1].cpu()],
            color='red', linestyle='--', linewidth=3
        )
        ax2.scatter(mean_1_sq[0].cpu(), mean_1_sq[1].cpu(), color='deepskyblue', label="Signal 1", edgecolors='none', s=100)
        ax2.scatter(mean_interp_sq[0].cpu(), mean_interp_sq[1].cpu(), color='red', label="Interpolant", edgecolors='none', s=100)
        ax2.scatter(mean_2_sq[0].cpu(), mean_2_sq[1].cpu(), color='deepskyblue', label="Signal 2", edgecolors='none', s=100)
        ax2.set_xlabel('Latent Dimension 1', fontsize=16)
        ax2.set_ylabel('Latent Dimension 2', fontsize=16)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.legend(loc='upper center', fontsize=12, facecolor='none', bbox_to_anchor=(0.5, 1.125), ncol=3, frameon=False)
        
        # Sample size note
        n = len(train_dataset)
        plt.text(
            0.98, 0.02, f"n = {n}",
            ha='right', va='bottom',
            transform=plt.gca().transAxes,
            fontsize=12, color=text_colour,
            alpha=0.8
        )
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2)
        
        # Save frame to buffer
        plt.ioff()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        
        plt.close(fig)
        plt.ion()
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  Generated {frame_idx + 1}/{len(alphas)} frames")
    
    # Save as GIF
    print(f"Saving GIF to {fname}...")
    frames[0].save(
        fname,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"GIF created successfully with {len(alphas)} frames!")
    plt.rcdefaults()


def plot_latent_space_3d(
    model,
    dataloader: torch.utils.data.DataLoader,
    fname: Optional[str] = None,
    background: str = "white",
    font_family: str = "serif",
    font_name: str = "Times New Roman"
) -> plt.Figure:
    """Plot 3D visualization of the latent space.
    
    Args:
        model: VAE model for encoding
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
