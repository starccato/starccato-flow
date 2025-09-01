from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

import numpy as np
import pandas as pd

import torch

from ..nn.vae import VAE

from ..utils.defaults import DEVICE

from .plotting_defaults import SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR, LATENT_SPACE_COLOUR

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

def plot_waveform_grid(
    signals: np.ndarray,
    max_value: float,
    num_cols: int = 2,
    num_rows: int = 4,
    fname: str = None,
    generated: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    if generated:
        signal_colour = GENERATED_SIGNAL_COLOUR
    else:
        signal_colour = SIGNAL_COLOUR

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(10, 15)
    )

    axes = axes.flatten()

    # plot each signal on a separate subplot
    for i, ax in enumerate(axes):
        d = [i / 4096 for i in range(0, 256)]
        d = [value - (53 / 4096) for value in d]
        y = signals[i].flatten()
        y = y * max_value
        ax.set_ylim(-600, 300)
        ax.plot(d, y, color=signal_colour)

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.grid(True)

        # remove y-axis ticks for the right-hand column
        if i % num_cols == num_cols - 1:
            ax.yaxis.set_ticklabels([])

        # remove x-axis tick labels for all but the bottom two plots
        if i < num_cols * (num_rows - 1):
            ax.xaxis.set_ticklabels([])

    # for i in range(512, 8 * 4):
    #     fig.delaxes(axes[i])

    fig.supxlabel('time (s)', fontsize=32)
    fig.supylabel('hD (cm)', fontsize=32)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, axes

def plot_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float,
    fname: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(15, 4))

    d = [i / 4096 for i in range(0, 256)]
    d = [value - (53 / 4096) for value in d]

    # plot the original signal
    y_original = original.flatten() * max_value
    ax.plot(d, y_original, color="blue", label="Original Signal")
    
    # plot the reconstructed signal
    y_reconstructed = reconstructed.flatten() * max_value
    ax.plot(d, y_reconstructed, color="orange", label="Decoder Reconstructed Signal")

    ax.set_ylim(-600, 300)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.grid(True)
    ax.set_title("Original and Reconstructed Signals")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hD (cm)")
    ax.legend()

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, ax

def plot_loss(
    losses: List[float],
    fname: str = None,
    axes: plt.Axes = None,
):
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()
    axes.plot(losses, label="Total Training Loss)")
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel("Loss", size=20)
    # axes.set_ylim(0, 100)
    axes.legend(fontsize=16)
    
    plt.tight_layout()

    # if fname:
    #     plt.savefig(fname)
    
    # return axes.get_figure()

def plot_individual_loss(
    total_losses: List[float],
    reconstruction_losses: List[float],
    kld_losses: List[float],
    fname: str = None,
): 
    fig = plt.figure(figsize=(10, 6))
    axes = fig.gca()

    axes.plot(total_losses, label="Total Training Loss", color='orange')
    axes.plot(reconstruction_losses, label="Total Validation Loss", color='yellow')
    axes.plot(kld_losses, label="Total Validation Loss", color='red')
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel("Loss", size=20)
    axes.legend(fontsize=16)
    
    plt.tight_layout()

    # if fname:
    #     plt.savefig(fname)
    
    # return axes.get_figure()

def plot_training_validation_loss(
    losses: List[float],
    validation_losses: List[float],
    fname: str = None,
    axes: plt.Axes = None,
):
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()

    axes.plot(losses, label="Total Training Loss", color='orange')
    axes.plot(validation_losses, label="Total Validation Loss", color='grey')
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel("Loss", size=20)
    axes.legend(fontsize=16)
    
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
    
    # return axes.get_figure()

def plot_latent_morphs(
    model: VAE, 
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float, 
    steps=10
):
    model.eval()

    with torch.no_grad():
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)


        interpolated_latents = [mean_1 * (1 - alpha) + mean_2 * alpha for alpha in np.linspace(0, 1, steps)]
        morphed_signals = [model.decoder(latent).cpu().detach().numpy() for latent in interpolated_latents]

    num_plots = steps + 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    axes = axes.flatten()

    # X-axis values (shared across all plots)
    d_vals = [i / 4096 for i in range(0, 256)]
    d_vals = [value - (53 / 4096) for value in d_vals]

    # Plot signal_1 (blue)
    y1 = signal_1.cpu().detach().numpy().flatten() * max_value
    axes[0].plot(d_vals, y1, color="blue")
    axes[0].set_ylim(-600, 300)
    axes[0].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[0].grid(True)
    axes[0].set_title("Original Signal 1 (Start)")

    # Plot the interpolated signals (red)
    for i, signal in enumerate(morphed_signals):
        y_interp = signal.flatten() * max_value
        # y_interp = signal.flatten()
        axes[i + 1].plot(d_vals, y_interp, color=GENERATED_SIGNAL_COLOUR)
        axes[i + 1].set_ylim(-600, 300)
        axes[i + 1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
        axes[i + 1].grid(True)
        axes[i + 1].set_title(f"Interpolated Signal {i + 1}")

    # Plot signal_2 (blue)
    y2 = signal_2.cpu().detach().numpy().flatten() * max_value
    axes[-1].plot(d_vals, y2, color="blue")
    axes[-1].set_ylim(-600, 300)
    axes[-1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[-1].grid(True)
    axes[-1].set_title("Original Signal 2 (End)")

    # Keep all tick labels
    fig.supxlabel('time (s)', fontsize=16)
    fig.supylabel('hD (cm)', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_latent_morph_grid(
    model,
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float,
    train_dataset,
    steps=10
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
        d_vals = [i / 4096 for i in range(0, 256)]
        d_vals = [d - (53 / 4096) for d in d_vals]

        # Posterior means for background latent scatter
        all_means = []
        for x, _ in train_dataset:
            x = torch.tensor(x).to(model.DEVICE)
            mean, _ = model.encoder(x)
            all_means.append(mean.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)

    # Set up figure
    fig = plt.figure(figsize=(15, 10))
    
    # ----- Row 1: Signals -----
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(d_vals, signal_1_np, color='blue')
    ax1.axvline(x=0, linestyle="--", color="black", alpha=0.5)
    ax1.set_ylim(-600, 300)
    ax1.set_title("Start Signal")
    ax1.set_ylabel("hD (cm)", fontsize=32)
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(d_vals, signal_mid, color=GENERATED_SIGNAL_COLOUR)
    ax2.axvline(x=0, linestyle="--", color="black", alpha=0.5)
    ax2.set_ylim(-600, 300)
    ax2.set_title("Interpolated Signal")
    ax2.set_xlabel("time (s)", fontsize=32)
    ax2.grid(True)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(d_vals, signal_2_np, color='blue')
    ax3.axvline(x=0, linestyle="--", color="black", alpha=0.5)
    ax3.set_ylim(-600, 300)
    ax3.set_title("End Signal")
    ax3.set_xlabel("time (s)")
    ax3.grid(True)

    # ----- Row 2: Latent space -----
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    ax4.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], alpha=0.1, color=LATENT_SPACE_COLOUR)
    ax4.scatter(mean_1[0].cpu(), mean_1[1].cpu(), mean_1[2].cpu(), color='blue', label="Start")
    ax4.set_title("Start in Latent Space")
    ax4.set_xlabel('Latent Dim 1')
    ax4.set_ylabel('Latent Dim 2')
    ax4.set_zlabel('Latent Dim 3')

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ax5.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], alpha=0.1, color=LATENT_SPACE_COLOUR)
    ax5.plot(
        [mean_1[0].cpu(), mean_2[0].cpu()],
        [mean_1[1].cpu(), mean_2[1].cpu()],
        [mean_1[2].cpu(), mean_2[2].cpu()],
        color=GENERATED_SIGNAL_COLOUR, linestyle='--'
    )
    ax5.scatter(mean_1[0].cpu(), mean_1[1].cpu(), mean_1[2].cpu(), color='blue', label="Start", alpha=0.5)
    ax5.scatter(mean_mid[0].cpu(), mean_mid[1].cpu(), mean_mid[2].cpu(), color=GENERATED_SIGNAL_COLOUR, label="Midpoint")
    ax5.scatter(mean_2[0].cpu(), mean_2[1].cpu(), mean_2[2].cpu(), color='blue', label="End", alpha=0.5)
    ax5.set_title("Latent Space Path")
    ax5.set_xlabel('Latent Dim 1')
    ax5.set_ylabel('Latent Dim 2')
    ax5.set_zlabel('Latent Dim 3')

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    ax6.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], alpha=0.1, color=LATENT_SPACE_COLOUR)
    ax6.scatter(mean_2[0].cpu(), mean_2[1].cpu(), mean_2[2].cpu(), color='blue', label="End")
    ax6.set_title("End in Latent Space")
    ax6.set_xlabel('Latent Dim 1')
    ax6.set_ylabel('Latent Dim 2')
    ax6.set_zlabel('Latent Dim 3')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Add vertical space between rows
    plt.show()

def animate_latent_morphs(
    model,  # Assuming model is a VAE instance
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float, 
    train_dataset,
    steps=10,
    interval=200,
    save_path=None
):
    model.eval()

    with torch.no_grad():
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)

        # Forward and backward interpolation
        forward_interpolated = [mean_1 * (1 - alpha) + mean_2 * alpha for alpha in np.linspace(0, 1, steps)]
        backward_interpolated = [mean_2 * (1 - alpha) + mean_1 * alpha for alpha in np.linspace(0, 1, steps)]
        interpolated_latents = forward_interpolated + backward_interpolated
        morphed_signals = [model.decoder(latent).cpu().detach().numpy() for latent in interpolated_latents]

        # Compute the posterior distribution
        all_means = []
        for x, y in train_dataset:
            x = torch.tensor(x).to(model.DEVICE)
            mean, _ = model.encoder(x)
            all_means.append(mean.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)

    fig = plt.figure(figsize=(10, 17))  # Adjust the figure size for vertical stacking

    # Create 3D plot for latent space
    ax_latent = fig.add_subplot(211, projection='3d')  # First plot (top) in vertical layout
    ax_latent.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], color=LATENT_SPACE_COLOUR, alpha=0.2, label='Posterior Distribution')
    ax_latent.scatter(mean_1[0].cpu().numpy(), mean_1[1].cpu().numpy(), mean_1[2].cpu().numpy(), color='blue', s=50, label='Signal 1')
    ax_latent.scatter(mean_2[0].cpu().numpy(), mean_2[1].cpu().numpy(), mean_2[2].cpu().numpy(), color='green', s=50, label='Signal 2')
    ax_latent.plot([mean_1[0].cpu().numpy(), mean_2[0].cpu().numpy()],
                   [mean_1[1].cpu().numpy(), mean_2[1].cpu().numpy()],
                   [mean_1[2].cpu().numpy(), mean_2[2].cpu().numpy()], color=GENERATED_SIGNAL_COLOUR, linestyle='--', label='Interpolation Path', linewidth=2)
    moving_point, = ax_latent.plot([], [], [], 'ro', markersize=7, label='Interpolated Point')
    # ax_latent.set_title('Latent Space Interpolation')
    ax_latent.set_xlabel('Latent Dim 1')
    ax_latent.set_ylabel('Latent Dim 2')
    ax_latent.set_zlabel('Latent Dim 3')
    # ax_latent.legend()

    # Create plot for signal morphing
    ax_signal = fig.add_subplot(212)  # Second plot (bottom) in vertical layout

    # X-axis values (shared across all plots)
    d_vals = [i / 4096 for i in range(0, 256)]
    d_vals = [value - (53 / 4096) for value in d_vals]

    # Initialize the plot
    line, = ax_signal.plot([], [], color=GENERATED_SIGNAL_COLOUR)
    ax_signal.set_xlim(min(d_vals), max(d_vals))
    ax_signal.set_ylim(-600, 300)
    ax_signal.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax_signal.grid(True)
    ax_signal.set_xlabel('time (s)', fontsize=16)
    ax_signal.set_ylabel('hD (cm)', fontsize=16)

    def init():
        line.set_data([], [])
        moving_point.set_data([], [])
        moving_point.set_3d_properties([])
        return line, moving_point

    def update(frame):
        y_interp = morphed_signals[frame].flatten() * max_value
        line.set_data(d_vals, y_interp)
        # ax_signal.set_title(f"Interpolated Signal {frame + 1}")

        # Update the moving point in the latent space
        latent_point = interpolated_latents[frame].cpu().numpy()
        moving_point.set_data(latent_point[0], latent_point[1])
        moving_point.set_3d_properties(latent_point[2])
        return line, moving_point

    ani = animation.FuncAnimation(fig, update, frames=len(interpolated_latents), init_func=init, blit=True, interval=interval, repeat=True)

    if save_path:
        ani.save(save_path, writer='imagemagick', fps=30)

    plt.show()

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

def plot_signal_distribution(
    signals: np.ndarray,  # (y_length, num_signals)
    generated: bool = True,
    background: str = "black",
    font_family: str = "serif",
    font_name: str = "Times New Roman",
    fname: str = None
):
    # set figure size
    plt.figure(figsize=(6, 6))

    # Set font globally for this plot
    plt.rcParams.update({
        'font.family': font_family,
        f'font.{font_family}': [font_name]
    })

    # Choose signal colour
    distribution_color = GENERATED_SIGNAL_COLOUR if generated else SIGNAL_COLOUR

    signals_df = pd.DataFrame(signals)
    median_line = signals_df.median(axis=1)

    # Transform x values
    d = [i / 4096 for i in range(0, 256)]
    d = [value - (53 / 4096) for value in d]

    # Theme colors
    if background == "black":
        plt.style.use("dark_background")
        text_color = "white"
        median_color = "white"
        vline_color = "white"
        grid_color = "gray"
        legend_facecolor = "none"
        transparent = True
    else:
        plt.style.use("default")
        text_color = "black"
        median_color = "black"
        vline_color = "black"
        grid_color = "lightgray"
        legend_facecolor = "none"
        transparent = False

    # === Percentiles with white base + overlay ===
    p2_5 = signals_df.quantile(0.025, axis=1)
    p97_5 = signals_df.quantile(0.975, axis=1)
    plt.fill_between(d, p2_5, p97_5, color="white", alpha=0.2)
    plt.fill_between(d, p2_5, p97_5, color=distribution_color, alpha=0.4)

    p25 = signals_df.quantile(0.25, axis=1)
    p75 = signals_df.quantile(0.75, axis=1)
    plt.fill_between(d, p25, p75, color="white", alpha=0.4)
    plt.fill_between(d, p25, p75, color=distribution_color, alpha=0.6)

    # === Median line ===
    plt.plot(d, median_line.values, color=median_color,
             linestyle=(0, (1, 1)), linewidth=1.5, alpha=1.0)

    # Vertical reference line
    plt.axvline(x=0, color=vline_color, linestyle='dashed', alpha=0.5)

    # Labels, limits, and grid
    plt.ylim(-600, 300)
    plt.xlim(min(d), max(d))
    plt.xlabel('time (s)', size=16, color=text_color)
    plt.ylabel('hD (cm)', size=16, color=text_color)

    # Sample size note
    n = signals.shape[1] if signals.ndim > 1 else len(signals)
    plt.text(
        0.98, 0.02, f"n = {n}",
        ha='right', va='bottom',
        transform=plt.gca().transAxes,
        fontsize=12, color=text_color,
        alpha=0.8
    )
    plt.tick_params(axis='both', which='major', labelsize=12, colors=text_color)

    # === Custom Legend (with true colors) ===
    legend_handles = [
        mpatches.Patch(color=distribution_color, alpha=0.6, label="Central 95%"),
        mpatches.Patch(color=distribution_color, alpha=1.00, label="Central 50%"),
        mlines.Line2D([], [], color=median_color, linestyle=(0, (1, 1)), linewidth=1.5, label="Median of signals")
    ]
    plt.legend(handles=legend_handles,
               facecolor="none", edgecolor=text_color, labelcolor=text_color, fontsize=12, loc = 'upper right',
               framealpha=0.0)

    # Save or show
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight",
                    facecolor="none" if transparent else legend_facecolor,
                    transparent=transparent)
    plt.show()

    plt.rcdefaults()


def plot_single_signal(
    signal: np.ndarray,
    max_value: float,
    fname: str = None,  # Added missing comma
):
    # Generate x-axis values
    d = [i / 4096 for i in range(0, 256)]
    d = [value - (53 / 4096) for value in d]

    # Process signal for plotting
    y = signal.flatten()
    y = y * max_value

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(d, y, color='blue')
    plt.axvline(x=0, color='black', linestyle='dotted', alpha=0.5)
    plt.ylim(-600, 300)
    plt.xlabel('time (s)', size=20)
    plt.ylabel('hD (cm)', size=20)
    plt.grid(True)

    # Save or show the plot
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()

def plot_gradients(
    encoder_gradients: List[float],
    decoder_gradients: List[float],
    q_gradients: List[float],
    fname: str = None
):
    """Plot encoder, decoder, and Q network gradient norms on separate subplots in the same figure."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))

    # Plot encoder gradients
    axes[0].plot(encoder_gradients, label='Encoder Gradients', color='blue')
    axes[0].set_title('Encoder Gradient Norms During Training')
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Gradient Norm')
    axes[0].legend()

    # Plot decoder gradients
    axes[1].plot(decoder_gradients, label='Decoder Gradients', color='orange')
    axes[1].set_title('Decoder Gradient Norms During Training')
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Gradient Norm')
    axes[1].legend()

    # Plot Q network gradients
    axes[2].plot(q_gradients, label='Q Gradients', color='green')
    axes[2].set_title('Q Gradient Norms During Training')
    axes[2].set_xlabel('Training Steps')
    axes[2].set_ylabel('Gradient Norm')
    axes[2].legend()

    # Adjust layout
    plt.tight_layout()

    # Save the figure if a filename is provided
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, axes

def plot_latent_space_3d(model, dataloader):
    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for y, d in dataloader:
            y = y.to(DEVICE)
            d = d.to(DEVICE)
            d = d.view(d.size(0), -1)
            y = y.view(y.size(0), -1)

            mean, _ = model.encoder(y)
            latent_vectors.append(mean.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)

    # Use the first 3 latent dimensions directly
    latent_3d = latent_vectors[:, :3]

    # Plot in 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2],
                         c='blue', s=40, alpha=0.7)

    ax.set_title('Latent Space Representation (3D)', fontsize=20)
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_zlabel('Latent Dimension 3')
    ax.grid(True)

    plt.show()


    import matplotlib.ticker as mtick

def plot_signal_grid(
    signals: np.ndarray,
    max_value: float,
    num_cols: int = 2,
    num_rows: int = 4,
    fname: str = None,
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    background: str = "white",
    x_y_label_on: bool = True,
    generated: bool = False,
):
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(5.6, 2,)
    )

    # Set font globally for this plot
    plt.rcParams.update({
        'font.family': font_family,
        f'font.{font_family}': [font_name]
    })

    signal_color = "red" if generated else "deepskyblue"

    # Theme colors
    if background == "black":
        plt.style.use("dark_background")
        text_color = "white"
        vline_color = "white"
        grid_color = "gray"
        transparent = True
    else:
        plt.style.use("default")
        text_color = "black"
        vline_color = "black"
        grid_color = "lightgray"
        transparent = False

    axes = axes.flatten()

    for i, ax in enumerate(axes):
        d = [j / 4096 for j in range(256)]
        d = [value - (53 / 4096) for value in d]

        y = signals[i].flatten() * max_value
        ax.set_ylim(-600, 300)
        ax.set_xlim(min(d), max(d))
        ax.plot(d, y, color=signal_color)

        ax.axvline(x=0, color=vline_color, linestyle="--", alpha=0.5)

        # ✅ Force ticks
        ax.set_yticks([300, 0, -600])
        ax.set_xticks([0.00])

        # ✅ Format ticks to 2 decimals
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

        ax.tick_params(axis="both", colors=text_color, labelsize=12)

        # Remove y-axis ticks for right-hand columns
        if i % num_cols != 0:
            ax.set_yticklabels([])

        # Remove x-axis ticks for all but bottom row
        if i < num_cols * (num_rows - 1):
            ax.set_xticklabels([])

        # Spine colors
        for spine in ax.spines.values():
            spine.set_color(text_color)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, transparent=transparent, dpi=300, bbox_inches="tight")

    plt.show()
    plt.rcdefaults()