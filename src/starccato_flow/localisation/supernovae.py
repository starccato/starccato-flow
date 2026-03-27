"""Handles CCSN location generation and coordinate transformations for sky localization."""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class Supernovae:
    """Manages supernova locations in galactic and equatorial coordinates."""
    
    # Earth's position in galactic coordinates (kpc)
    SUN_LOCATION = np.array([0.0, 8.178, 0.0208]) # Sun is about 8.178 kpc from galactic center, and ~20.8 pc above the galactic plane
    EARTH_LOCATION = np.array([0.0, 0.0, 0.0])  # Assume that the sun and earth are co-located for simplicity in heliocentric coordinates
    
    def __init__(self, locations_file: Optional[str] = None, rotation_offset: float = 0.0, limit: Optional[int] = None):
        """Initialize supernova location handler.
        
        Args:
            locations_file: Path to CSV file with galactic coordinates (x_kpc, y_kpc, z_kpc)
                          If None, locations will be generated on demand
            rotation_offset: Additional rotation angle (in radians) to apply to Earth's orientation.
                           Positive values rotate eastward. Default is 0 (standard J2000 orientation).
            limit: Maximum number of locations to load (None for all)
            rotation_offset: Additional rotation angle (in radians) to apply to Earth's orientation.
                           Positive values rotate eastward. Default is 0 (standard J2000 orientation).
        """
        self.locations_file = locations_file
        self.rotation_offset = rotation_offset
        self._galactic_coords = None
        self._equatorial_coords = None
        self._distances = None
        
        if locations_file is not None:
            self.load_locations(locations_file, limit)
    
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
        
        # Radial distribution parameters (Faucher-Giguère & Kaspi 2006)
        A = 1.96
        r_0 = 17.2
        theta_0 = 0.08
        beta = 0.13
        
        def radial_pdf(r):
            """Radial distribution of galactic supernovae."""
            return A * np.sin((np.pi * r) / r_0 + theta_0) * np.exp(-beta * r)
        
        def pdf_2d(r):
            """2D PDF accounting for area element in polar coordinates."""
            return radial_pdf(r) * r
        
        # Find maximum for rejection sampling
        r_test = np.linspace(0.01, 16.8, 1000)
        pdf_max = np.max(np.abs(pdf_2d(r_test)))
        
        # Rejection sampling for radial distances
        r_samples = []
        while len(r_samples) < num_supernovae:
            # Propose samples uniformly
            r_proposal = np.random.uniform(0.01, 16.8, num_supernovae * 2)
            u = np.random.uniform(0, pdf_max, num_supernovae * 2)
            # Accept where u < pdf_2d(r)
            accepted = r_proposal[u < np.abs(pdf_2d(r_proposal))]
            r_samples.extend(accepted)
        
        r = np.array(r_samples[:num_supernovae])
        
        # Sample angles uniformly (azimuthally symmetric disk)
        theta = np.random.uniform(0, 2 * np.pi, num_supernovae)
        
        # Convert polar to Cartesian (x, y)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Sample z heights from Gaussian (scale height ~100 pc = 0.1 kpc)
        z = np.random.normal(loc=0, scale=0.1, size=num_supernovae)
        
        # Store galactic coordinates
        self._galactic_coords = np.column_stack([x, y, z])
        
        # Compute derived quantities
        self._compute_equatorial_coordinates()
        self._compute_distances()
        
        return self._galactic_coords
    
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
        
        # First, convert to standard galactic coordinates
        # In standard system: X→GC, Y→l=90°, Z→NGP
        # Our system: x,y in disk plane, z⊥disk, GC at -y direction
        # So: X_std = -y_gal, Y_std = x_gal, Z_std = z_gal
        X_std = -y_gal
        Y_std = x_gal  
        Z_std = z_gal
        
        # Now apply standard galactic→equatorial rotation matrix
        T11, T12, T13 = -0.0548755604, -0.8734370902, -0.4838350155
        T21, T22, T23 = +0.4941094279, -0.4448296300, +0.7469822445
        T31, T32, T33 = -0.8676661490, -0.1980763734, +0.4559837762
        
        x_eq = T11 * X_std + T12 * Y_std + T13 * Z_std
        y_eq = T21 * X_std + T22 * Y_std + T23 * Z_std
        z_eq = T31 * X_std + T32 * Y_std + T33 * Z_std
        
        # Apply additional rotation offset around z-axis (simulates Earth rotation or different time)
        if self.rotation_offset != 0.0:
            cos_rot = np.cos(self.rotation_offset)
            sin_rot = np.sin(self.rotation_offset)
            x_eq_rot = cos_rot * x_eq - sin_rot * y_eq
            y_eq_rot = sin_rot * x_eq + cos_rot * y_eq
            x_eq, y_eq = x_eq_rot, y_eq_rot
        
        # Convert to spherical (RA, Dec)
        distance = np.sqrt(x_eq**2 + y_eq**2 + z_eq**2)
        ra = np.arctan2(y_eq, x_eq)  # radians
        dec = np.arcsin(z_eq / (distance + 1e-10))  # radians
        
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
        # Convert to heliocentric Cartesian
        x_rel = distance * np.sin(ra) * np.cos(dec)
        y_rel = -distance * np.cos(ra) * np.cos(dec)
        z_rel = distance * np.sin(dec)
        
        # Convert to galactic coordinates (add Earth's position)
        x = x_rel + self.EARTH_LOCATION[0]
        y = y_rel + self.EARTH_LOCATION[1]
        z = z_rel + self.EARTH_LOCATION[2]
        
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
        # The galactic center is located at (0, -8.178 kpc, 0.0208 kpc) in our coordinate system
        # But we want the direction to it, not its absolute position
        # Direction vector from Earth (at origin) to galactic center
        gc_x = 0.0 - self.EARTH_LOCATION[0]
        gc_y = -8.178 - self.EARTH_LOCATION[1]  # Galactic center is at -y direction
        gc_z = 0.0208 - self.EARTH_LOCATION[2]
        
        # Convert to standard galactic coordinates for transformation
        X_std = -gc_y
        Y_std = gc_x
        Z_std = gc_z
        
        # Apply standard galactic→equatorial rotation matrix
        T11, T12, T13 = -0.0548755604, -0.8734370902, -0.4838350155
        T21, T22, T23 = +0.4941094279, -0.4448296300, +0.7469822445
        T31, T32, T33 = -0.8676661490, -0.1980763734, +0.4559837762
        
        x_eq = T11 * X_std + T12 * Y_std + T13 * Z_std
        y_eq = T21 * X_std + T22 * Y_std + T23 * Z_std
        z_eq = T31 * X_std + T32 * Y_std + T33 * Z_std
        
        # Apply rotation offset if set
        if self.rotation_offset != 0.0:
            cos_rot = np.cos(self.rotation_offset)
            sin_rot = np.sin(self.rotation_offset)
            x_eq_rot = cos_rot * x_eq - sin_rot * y_eq
            y_eq_rot = sin_rot * x_eq + cos_rot * y_eq
            x_eq, y_eq = x_eq_rot, y_eq_rot
        
        # Convert to spherical coordinates
        distance = np.sqrt(x_eq**2 + y_eq**2 + z_eq**2)
        ra = np.arctan2(y_eq, x_eq)
        dec = np.arcsin(z_eq / distance)
        
        return ra, dec

    def plot_galactic_distribution(
        self,
        fname_3d: Optional[str] = "plots/galactic_supernovae_3d.png",
        fname_xy: Optional[str] = "plots/galactic_supernovae_xy.png",
        fname_xz: Optional[str] = "plots/galactic_supernovae_xz.png",
        background: str = "black",
        font_family: str = "sans-serif",
        font_name: str = "Avenir",
        scatter_size: float = 0.001,
        sun_marker_size: float = 100,
        show: bool = False,
        dpi: int = 150,
    ):
        """Plot galactic supernova locations in 3D and projected views.

        Args:
            fname_3d: Output path for the 3D plot
            fname_xy: Output path for the X-Y projection plot
            fname_xz: Output path for the X-Z projection plot
            background: Plot theme, either "white" or "black"
            font_family: Font family to use
            font_name: Specific font name to use
            scatter_size: Marker size for supernova points
            sun_marker_size: Marker size for the sun marker
            show: Whether to display plots instead of closing after creation
            dpi: DPI used when saving output files

        Returns:
            List of matplotlib figures in [3D, X-Y, X-Z] order
        """
        if self._galactic_coords is None:
            raise ValueError("No galactic coordinates available. Load or generate locations first.")

        from ..plotting import plot_galactic_distribution

        return plot_galactic_distribution(
            galactic_coords=self._galactic_coords,
            sun_location=self.SUN_LOCATION,
            fname_3d=fname_3d,
            fname_xy=fname_xy,
            fname_xz=fname_xz,
            background=background,
            font_family=font_family,
            font_name=font_name,
            scatter_size=scatter_size,
            sun_marker_size=sun_marker_size,
            show=show,
            dpi=dpi,
        )
    
    def __len__(self) -> int:
        """Return number of locations."""
        return len(self._galactic_coords) if self._galactic_coords is not None else 0
