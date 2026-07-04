"""Handles CCSN location generation and coordinate transformations for sky localization."""

from typing import Optional, Tuple
import os
import io
import numpy as np
import pandas as pd
from PIL import Image

from ..plotting.analysis import plot_surface_density


class Supernovae:
    """Manages supernova locations in galactic and equatorial coordinates."""
    
    # Earth's position in galactic coordinates (kpc)
    SUN_LOCATION = np.array([0.0, 8.178, 0.0208]) # Sun is about 8.178 kpc from galactic center, and ~20.8 pc above the galactic plane
    EARTH_LOCATION = np.array([0.0, 0.0, 0.0])  # Assume that the sun and earth are co-located for simplicity in heliocentric coordinates
    
    def __init__(
        self,
        locations_file: Optional[str] = None,
        rotation_offset: float = np.deg2rad(60.0),
        limit: Optional[int] = None,
    ):
        """Initialize supernova location handler.
        
        Args:
            locations_file: Path to CSV file with galactic coordinates (x_kpc, y_kpc, z_kpc)
                          If None, locations will be generated on demand
            rotation_offset: Additional rotation angle (in radians) to apply to Earth's orientation.
                           Positive values rotate eastward. Default is +60 degrees.
            limit: Maximum number of locations to load (None for all)
        """
        self.locations_file = locations_file
        self.rotation_offset = rotation_offset
        self._galactic_coords = None
        self._equatorial_coords = None
        self._distances = None
        
        if locations_file is not None:
            self.load_locations(locations_file, limit)
        else:
            self.generate_locations(num_supernovae=2000000, seed=42)  # Generate a default set of locations if no file is provided
    
    def load_locations(self, filepath: str, limit: Optional[int] = None) -> None:
        """Load supernova locations from CSV file.
        
        Args:
            filepath: Path to CSV with columns: x_kpc, y_kpc, z_kpc
            limit: Maximum number of locations to load (None for all)
        """
        data = pd.read_csv(filepath)
        self._galactic_coords = np.column_stack([
            data['x_kpc'].values,
            data['y_kpc'].values,
            data['z_kpc'].values
        ])

        if limit is not None:
            # sample a subset of the locations if limit is specified
            np.random.shuffle(self._galactic_coords)
            self._galactic_coords = self._galactic_coords[:limit]
        
        # Compute derived quantities
        self._compute_equatorial_coordinates()
        self._compute_distances()
        
        print(f"✓ Loaded {len(self._galactic_coords)} supernova locations from {filepath}")

    def _plot_surface_density(self, fname: str, font_family: str = "serif", font_name: str = "Times New Roman", transparent: bool = True) -> None:
        """Plot surface density of supernovae in the galactic plane."""
        
        plot_surface_density(
            fname=fname,
            font_family=font_family,
            font_name=font_name,
            transparent=transparent
        )

    def radial_pdf(self, r):
            """Radial distribution of galactic supernovae."""
            return self.A * np.sin((np.pi * r) / self.r_0 + self.theta_0) * np.exp(-self.beta * r)
        
    def pdf_2d(self, r):
            """2D PDF accounting for area element in polar coordinates."""
            return self.radial_pdf(r) * r
    
    def generate_locations(self, num_supernovae: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate galactic supernova locations using rejection sampling.
        
        Args:
            num_supernovae: Number of supernovae to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (num_supernovae, 3) with galactic Cartesian coordinates (x, y, z) in kpc
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.A = 1.96
        self.r_0 = 17.2
        self.theta_0 = 0.08
        self.beta = 0.13
        
        # Find maximum for rejection sampling
        r_test = np.linspace(0.01, 16.8, 1000)
        pdf_max = np.max(np.abs(self.pdf_2d(r_test)))
        
        # Rejection sampling for radial distances
        r_samples = []
        while len(r_samples) < num_supernovae:
            # Propose samples uniformly
            r_proposal = np.random.uniform(0.01, 16.8, num_supernovae * 2)
            u = np.random.uniform(0, pdf_max, num_supernovae * 2)
            # Accept where u < pdf_2d(r)
            accepted = r_proposal[u < np.abs(self.pdf_2d(r_proposal))]
            r_samples.extend(accepted)
        
        r = np.array(r_samples[:num_supernovae])
        
        # Sample angles uniformly (azimuthally symmetric disk)
        theta = np.random.uniform(0, 2 * np.pi, num_supernovae)
        
        # Convert polar to Cartesian (x, y)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Sample z heights from Gaussian (scale height ~100 pc = 0.1 kpc)
        z = np.random.normal(loc=0, scale=0.3, size=num_supernovae)
        
        # Store galactic coordinates
        self._galactic_coords = np.column_stack([x, y, z])
        
        # Compute derived quantities
        self._compute_equatorial_coordinates()
        self._compute_distances()
            
    def sample_locations(self, num_supernovae: int, min_kiloparsec: float = 0.0, max_kiloparsec: float = 16800.0) -> np.ndarray:
        """Sample supernova locations from region between min_kiloparsec and max_kiloparsec from Earth."""
        selected_region = self._galactic_coords[(self._distances >= min_kiloparsec) & (self._distances <= max_kiloparsec)]
        selected_locations = selected_region[np.random.choice(selected_region.shape[0], size=num_supernovae, replace=False)]

        return selected_locations

    
    def _compute_distances(self) -> None:
        """Compute distances from Earth to each supernova."""
        if self._galactic_coords is None:
            raise ValueError("No galactic coordinates available. Load or generate locations first.")
        
        # Heliocentric coordinates (relative to Sun/Earth location in galactocentric frame)
        helio_coords = self._galactic_coords - self.SUN_LOCATION
        
        # Distance in kpc
        self._distances = np.linalg.norm(helio_coords, axis=1)
    
    def _compute_equatorial_coordinates(self) -> None:
        """Convert galactic Cartesian coordinates to equatorial (RA, Dec)."""
        if self._galactic_coords is None:
            raise ValueError("No galactic coordinates available. Load or generate locations first.")
        
        # Heliocentric coordinates (relative to Sun)
        helio_coords = self._galactic_coords - self.SUN_LOCATION
        earth_coords = helio_coords - self.EARTH_LOCATION
        x_gal, y_gal, z_gal = earth_coords.T
        
        # In this coordinate system:
        # - Galactic disk is in the x-y plane (z perpendicular to disk)
        # - Galactic center is at (0, -8.178, 0.0208) from Sun
        # Apply proper rotation to equatorial coordinates (J2000)
        # North Galactic Pole: RA=192.86°, Dec=27.13°
        # Galactic center direction from Sun: toward negative y-axis
        
        # Standard galactic to equatorial rotation matrix
        # This assumes: galactic x→center, y→l=90°, z→NGP
        # Our system has disk in x-y, so we need to account for this
        
        # Use Astropy's Galactocentric frame to transform to equatorial
        from astropy.coordinates import Galactocentric, ICRS
        import astropy.units as u
        
        # Create Galactocentric coordinates (x, y, z in kpc)
        gal_centric = Galactocentric(
            x=x_gal*u.kpc,
            y=y_gal*u.kpc,
            z=z_gal*u.kpc
        )
        
        # Transform to ICRS (equatorial)
        icrs = gal_centric.transform_to(ICRS())
        ra = icrs.ra.rad
        dec = icrs.dec.rad
        
        # Apply rotation offset if set (additional rotation around z-axis)
        if self.rotation_offset != 0.0:
            cos_rot = np.cos(self.rotation_offset)
            sin_rot = np.sin(self.rotation_offset)
            x_eq = np.cos(dec) * np.cos(ra)
            y_eq = np.cos(dec) * np.sin(ra)
            z_eq = np.sin(dec)
            
            x_rot = cos_rot * x_eq - sin_rot * y_eq
            y_rot = sin_rot * x_eq + cos_rot * y_eq
            z_rot = z_eq
            
            ra = np.arctan2(y_rot, x_rot)
            dec = np.arcsin(z_rot)
        
        self._equatorial_coords = np.column_stack([ra, dec])
    
    
    def equatorial_to_galactic(self, ra: np.ndarray, dec: np.ndarray, distance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert equatorial (RA, Dec, distance) to galactic Cartesian.
        
        Args:
            ra: Right Ascension in radians
            dec: Declination in radians
            distance: Distance in kpc
            
        Returns:
            x, y, z: Galactic Cartesian coordinates in kpc
        """
        # Convert from spherical to Cartesian in equatorial frame
        x_eq = distance * np.cos(dec) * np.cos(ra)
        y_eq = distance * np.cos(dec) * np.sin(ra)
        z_eq = distance * np.sin(dec)
        
        # Apply inverse rotation to undo rotation_offset
        if self.rotation_offset != 0.0:
            cos_rot = np.cos(-self.rotation_offset)
            sin_rot = np.sin(-self.rotation_offset)
            x_eq_rot = cos_rot * x_eq - sin_rot * y_eq
            y_eq_rot = sin_rot * x_eq + cos_rot * y_eq
            x_eq, y_eq = x_eq_rot, y_eq_rot
        
        # Apply inverse of the galactic→equatorial rotation matrix
        # Standard rotation matrix (from _compute_equatorial_coordinates)
        T11, T12, T13 = -0.0548755604, -0.8734370902, -0.4838350155
        T21, T22, T23 = +0.4941094279, -0.4448296300, +0.7469822445
        T31, T32, T33 = -0.8676661490, -0.1980763734, +0.4559837762
        
        # Transpose to get inverse (rotation matrices are orthogonal)
        T_inv = np.array([
            [T11, T21, T31],
            [T12, T22, T32],
            [T13, T23, T33]
        ])
        
        # Apply inverse rotation
        coords_eq = np.column_stack([x_eq, y_eq, z_eq])
        coords_std = coords_eq @ T_inv.T
        X_std, Y_std, Z_std = coords_std.T
        
        # Convert from standard galactic to our coordinate system
        # Standard: X→GC, Y→l=90°, Z→NGP
        # Our system: x,y in disk plane, z⊥disk, GC at -y direction
        x_gal = Y_std
        y_gal = -X_std
        z_gal = Z_std
        
        # Add Earth's position to get galactocentric coordinates
        x = x_gal + self.EARTH_LOCATION[0]
        y = y_gal + self.EARTH_LOCATION[1]
        z = z_gal + self.EARTH_LOCATION[2]
        
        return x, y, z
    
    @property
    def galactic_coords(self) -> Optional[np.ndarray]:
        """Get galactic Cartesian coordinates (x, y, z) in kpc."""
        return self._galactic_coords
    
    @property
    def equatorial_coords(self) -> Optional[np.ndarray]:
        """Get equatorial coordinates (RA, Dec) in radians."""
        return self._equatorial_coords
    
    @property
    def ra(self) -> Optional[np.ndarray]:
        """Get Right Ascension in radians."""
        return self._equatorial_coords[:, 0] if self._equatorial_coords is not None else None
    
    @property
    def dec(self) -> Optional[np.ndarray]:
        """Get Declination in radians."""
        return self._equatorial_coords[:, 1] if self._equatorial_coords is not None else None
    
    @property
    def distances(self) -> Optional[np.ndarray]:
        """Get distances from Sun in kpc."""
        return self._distances
    
    def get_sky_params(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get sky localization parameters (RA, Dec, distance).
        
        Args:
            indices: Optional indices to select specific supernovae
            
        Returns:
            Array of shape (N, 3) with [RA, Dec, distance]
        """
        if self._equatorial_coords is None or self._distances is None:
            raise ValueError("No coordinates available. Load or generate locations first.")
        
        ra = self._equatorial_coords[:, 0]
        dec = self._equatorial_coords[:, 1]
        
        sky_params = np.column_stack([ra, dec, self._distances])
        
        if indices is not None:
            return sky_params[indices]
        return sky_params
    
    def get_galactic_center_direction(self) -> Tuple[float, float]:
        """Get the direction to the Galactic Center in equatorial coordinates.
        
        Returns:
            Tuple of (RA, Dec) in radians for the Galactic Center direction from Earth
        """
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        
        # Galactic center is at galactic longitude 0, latitude 0
        gc = SkyCoord(l=0*u.deg, b=0*u.deg, frame='galactic')
        
        # Convert to ICRS (equatorial) coordinates
        gc_icrs = gc.icrs
        ra = gc_icrs.ra.rad
        dec = gc_icrs.dec.rad
        
        return ra, dec

    def plot_galactic_distribution(
        self,
        fname_3d: Optional[str] = "plots/galactic_supernovae_3d.png",
        fname_xy: Optional[str] = "plots/galactic_supernovae_xy.png",
        fname_xz: Optional[str] = "plots/galactic_supernovae_xz.png",
        fname_xy_closeup: Optional[str] = "plots/galactic_supernovae_xy_closeup.png",
        fname_yx_zx: Optional[str] = "plots/galactic_supernovae_yx_zx.png",
        background: str = "black",
        transparent: Optional[bool] = None,
        light_year: bool = False,
        highlight_indices: Optional[np.ndarray] = None,
        font_family: str = "sans-serif",
        font_name: str = "Avenir",
        scatter_size: float = 0.001,
        sun_marker_size: float = 100,
        show: bool = False,
        dpi: int = 300,
        figsize: tuple = (16, 16),
    ):
        """Plot galactic supernova locations in 3D and projected views.

        Args:
            fname_3d: Output path for the 3D plot
            fname_xy: Output path for the X-Y projection plot
            fname_xz: Output path for the X-Z projection plot
            fname_xy_closeup: Output path for the X-Y closeup projection plot
            fname_yx_zx: Output path for the stacked Y-X and Z-X plot
            background: Plot theme, either "white" or "black"
            font_family: Font family to use
            font_name: Specific font name to use
            scatter_size: Marker size for supernova points
            sun_marker_size: Marker size for the sun marker
            show: Whether to display plots instead of closing after creation
            dpi: DPI used when saving output files
            figsize: Figure size in inches as (width, height). Default (16, 16) produces ~2400x2400 pixels at 150 dpi.

        Returns:
            List of matplotlib figures in [3D, X-Y, X-Z] order
        """
        if self._galactic_coords is None:
            raise ValueError("No galactic coordinates available. Load or generate locations first.")

        from ..plotting import plot_galactic_distribution

        return plot_galactic_distribution(
            galactic_coords=self._galactic_coords,
            sun_location=self.SUN_LOCATION,
            highlight_indices=highlight_indices,
            fname_3d=fname_3d,
            fname_xy=fname_xy,
            fname_xz=fname_xz,
            fname_xy_closeup=fname_xy_closeup,
            fname_yx_zx=fname_yx_zx,
            background=background,
            transparent=transparent,
            light_year=light_year,
            font_family=font_family,
            font_name=font_name,
            scatter_size=scatter_size,
            sun_marker_size=sun_marker_size,
            show=show,
            dpi=dpi,
            figsize=figsize,
        )
    
    def sample_supernovae_for_epoch(
        self,
        epoch: int,
        n_samples: int,
        num_epochs: int,
        exponential: bool = True,
        epoch_dir: Optional[str] = None,
        fname: Optional[str] = None,
        font_family: str = "sans-serif",
        font_name: str = "Avenir",
        transparent: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample RA/Dec/d sky parameters for an epoch distance shell.

        When ``exponential`` is True, samples are weighted to favor larger
        distances in the shell (near ``max_kiloparsec``), with the weighting
        strength increasing over epochs.
        
        Args:
            epoch: Current epoch number
            n_samples: Number of samples to draw
            num_epochs: Total number of epochs (for exponential weighting)
            exponential: Whether to use exponential weighting (favor larger distances)
            epoch_dir: Optional directory to save galactic distribution plots
            fname: Optional filename for thesis plots
            font_family: Font family for plots
            font_name: Font name for plots
            transparent: Whether to make plots transparent
            
        Returns:
            Tuple of (ra, dec, d) arrays with sampled sky parameters
        """
        from ..utils.defaults import MAX_DISTANCE_KPC
        
        threshold_d = MAX_DISTANCE_KPC
        min_d_mask = 0.0
        max_d_mask = min(threshold_d, (epoch / num_epochs) * threshold_d + 0.5)
        distance_mask = (
            (self.distances >= min_d_mask)
            & (self.distances <= max_d_mask)
        )
        candidate_indices = np.where(distance_mask)[0]
        
        if candidate_indices.size == 0:
            raise ValueError(
                f"No supernovae found in [{min_d_mask:.3f}, {max_d_mask:.3f}] kpc range."
            )

        sample_probs = None
        if exponential:
            candidate_distances = self.distances[candidate_indices]
            shell_width = max(max_d_mask - min_d_mask, 1e-8)
            normalized_distance = np.clip((candidate_distances - min_d_mask) / shell_width, 0.0, 1.0)

            # Increase bias through training so later epochs concentrate more strongly
            # near the far edge of each shell.
            epoch_fraction = (epoch + 1) / max(num_epochs, 1)
            growth = 1.0 + 7.0 * epoch_fraction
            weights = np.exp(growth * normalized_distance)
            weight_sum = np.sum(weights)
            if np.isfinite(weight_sum) and weight_sum > 0.0:
                sample_probs = weights / weight_sum

        sampled_indices = np.random.choice(
            candidate_indices,
            size=n_samples,
            replace=candidate_indices.size < n_samples, # conditional sampling with replacement if not enough candidates
            p=sample_probs,
        )
        if fname is not None:
            # Use provided filename (thesis plots)
            self.plot_galactic_distribution(
                fname_xy=fname,
                background="black",
                light_year=False,
                highlight_indices=sampled_indices,
                show=False,
                dpi=300,
                font_family=font_family,
                font_name=font_name,
                transparent=transparent
            )
        elif epoch_dir is not None:
            # Construct filename from epoch_dir (training plots)
            os.makedirs(epoch_dir, exist_ok=True)
            epoch_fname = os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}_galactic_xy.png")
            
            self.plot_galactic_distribution(
                fname_xy=epoch_fname,
                background="black",
                transparent=False,
                light_year=False,
                highlight_indices=sampled_indices,
                show=False,
                dpi=300,
                font_family=font_family,
                font_name=font_name,
            )
        sampled_sky_params = self.get_sky_params(indices=sampled_indices)

        return sampled_sky_params[:, 0], sampled_sky_params[:, 1], sampled_sky_params[:, 2]
    
    def __len__(self) -> int:
        """Return number of locations."""
        return len(self._galactic_coords) if self._galactic_coords is not None else 0
    
    def create_epoch_gif(
        self,
        epoch_dir: str,
        output_fname: str,
        duration: int = 200,
        loop: int = 0,
        num_epochs: int = 256
    ) -> None:
        """Create an animated GIF from epoch training plots.
        
        Args:
            epoch_dir: Directory containing epoch_XXXX_galactic_xy.png files
            output_fname: Output path for the animated GIF
            duration: Duration of each frame in milliseconds (default 200ms)
            loop: Number of loops (0 = infinite loop)
            num_epochs: Total number of epochs to process
            
        Returns:
            None
            
        Example:
            >>> supernovae.create_epoch_gif(
            ...     epoch_dir="outdir/flow_matching/epoch_data",
            ...     output_fname="supernovae_training_animation.gif",
            ...     duration=150
            ... )
        """
        frames = []
        
        print(f"Creating GIF from epoch frames in {epoch_dir}...")
        
        # Load frames in order
        for epoch in range(num_epochs):
            epoch_fname = os.path.join(epoch_dir, f"epoch_{epoch + 1:04d}_galactic_xy.png")
            
            if not os.path.exists(epoch_fname):
                print(f"  Warning: {epoch_fname} not found, skipping epoch {epoch + 1}")
                continue
            
            try:
                frame = Image.open(epoch_fname)
                frames.append(frame)
                if (epoch + 1) % 50 == 0:
                    print(f"  Loaded {epoch + 1}/{num_epochs} frames")
            except Exception as e:
                print(f"  Error loading {epoch_fname}: {e}")
                continue
        
        if not frames:
            raise ValueError(f"No epoch frames found in {epoch_dir}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_fname) if os.path.dirname(output_fname) else ".", exist_ok=True)
        
        # Save as animated GIF
        frames[0].save(
            output_fname,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=False
        )
        
        print(f"✓ Created GIF with {len(frames)} frames: {output_fname}")
        print(f"  Duration per frame: {duration}ms")
        print(f"  Total duration: ~{len(frames) * duration / 1000:.1f} seconds")
    
    def create_training_animation_gif(
        self,
        fname: str,
        n_samples: int = 18000,
        num_epochs: int = 256,
        duration: int = 150,
        loop: int = 0,
        font_family: str = "sans-serif",
        font_name: str = "Avenir"
    ) -> None:
        """Create an animated GIF of training progress on-the-fly without saving intermediate files.
        
        Generates epoch plots in memory and directly creates the GIF. No temporary files are saved.
        
        Args:
            fname: Output path for the animated GIF
            n_samples: Number of samples to draw per epoch
            num_epochs: Total number of epochs to generate (default 256)
            duration: Duration of each frame in milliseconds (default 150ms)
            loop: Number of loops (0 = infinite loop)
            font_family: Font family for plots
            font_name: Specific font name for plots
            
        Returns:
            None
            
        Example:
            >>> supernovae.create_training_animation_gif(
            ...     output_fname="training_animation.gif",
            ...     n_samples=18000,
            ...     num_epochs=256,
            ...     duration=150
            ... )
        """
        import matplotlib.pyplot as plt
        from ..utils.defaults import MAX_DISTANCE_KPC
        
        frames = []
        print(f"Generating {num_epochs} epoch frames for GIF...")
        
        for epoch in range(num_epochs):
            try:
                # Calculate distance shell for this epoch
                threshold_d = MAX_DISTANCE_KPC
                min_d_mask = 0.0
                max_d_mask = min(threshold_d, (epoch / num_epochs) * threshold_d + 0.5)
                distance_mask = (
                    (self.distances >= min_d_mask)
                    & (self.distances <= max_d_mask)
                )
                candidate_indices = np.where(distance_mask)[0]
                
                if candidate_indices.size == 0:
                    continue
                
                # Sample with exponential weighting
                candidate_distances = self.distances[candidate_indices]
                shell_width = max(max_d_mask - min_d_mask, 1e-8)
                normalized_distance = np.clip((candidate_distances - min_d_mask) / shell_width, 0.0, 1.0)
                epoch_fraction = (epoch + 1) / max(num_epochs, 1)
                growth = 1.0 + 7.0 * epoch_fraction
                weights = np.exp(growth * normalized_distance)
                weight_sum = np.sum(weights)
                sample_probs = weights / weight_sum if (np.isfinite(weight_sum) and weight_sum > 0.0) else None
                
                sampled_indices = np.random.choice(
                    candidate_indices,
                    size=n_samples,
                    replace=candidate_indices.size < n_samples,
                    p=sample_probs,
                )
                
                # Generate plot in memory
                from ..plotting.analysis import plot_galactic_distribution
                fig_list = plot_galactic_distribution(
                    galactic_coords=self._galactic_coords,
                    sun_location=self.SUN_LOCATION,
                    highlight_indices=sampled_indices,
                    fname_3d=None,
                    fname_xy=None,
                    fname_xz=None,
                    background="black",
                    transparent=True,
                    light_year=False,
                    font_family=font_family,
                    font_name=font_name,
                    scatter_size=0.001,
                    sun_marker_size=100,
                    show=False,
                    dpi=100,
                    figsize=(12, 12),
                )
                
                # Get the X-Y plot (second figure)
                fig = fig_list[1] if len(fig_list) > 1 else fig_list[0]
                
                # Save figure to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', transparent=True)
                buf.seek(0)
                frame = Image.open(buf).copy()
                frames.append(frame)
                buf.close()
                
                # Close all figures
                for f in fig_list:
                    plt.close(f)
                
                if (epoch + 1) % 50 == 0:
                    print(f"  Generated {epoch + 1}/{num_epochs} frames")
                    
            except Exception as e:
                print(f"  Error generating epoch {epoch + 1}: {e}")
                plt.close('all')
                continue
        
        if not frames:
            raise ValueError("Failed to generate any frames")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_fname) if os.path.dirname(output_fname) else ".", exist_ok=True)
        
        # Save as animated GIF
        print(f"Creating GIF from {len(frames)} frames...")
        frames[0].save(
            output_fname,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=loop,
            optimize=False
        )
        
        # Clean up frames from memory
        for frame in frames:
            frame.close()
        
        print(f"✓ Created training animation GIF: {fname}")
        print(f"  Total frames: {len(frames)}")
        print(f"  Duration per frame: {duration}ms")
        print(f"  Total duration: ~{len(frames) * duration / 1000:.1f} seconds")
