"""Flat all-sky constellation projection utilities."""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.collections import LineCollection
from matplotlib.path import Path

from .sky import _hip_lookup_table_with_mag
from . import set_plot_style
from ..utils.defaults_general import SKY_MAP_ROOT

try:
    from astropy.coordinates import SkyCoord, FK4
    import astropy.units as u
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


def _check_ra_wrapping(ra1, ra2, threshold=180):
    """Check if a line between two RA values crosses the 0/360 boundary."""
    ra_diff = abs(ra2 - ra1)
    return ra_diff > threshold


def _constellation_stick_segments_flat(constellation_style="western", custom_constellations=None):
    """Get constellation stick figure segments for flat equirectangular projection.
    
    Properly handles RA wrapping by:
    1. Using shortest angular path to determine intended direction
    2. If wrapping occurs, interpolates split point at boundary
    3. Draws segments on both sides of boundary for visual continuity
    
    Parameters
    ----------
    constellation_style : str
        Which constellation set to use:
        - "western" (default): uses constellations_rey.txt
        - "chinese": uses constellations_chinese.txt
        - "custom": uses custom_constellations dict (must be provided)
    custom_constellations : dict, optional
        Custom constellation definitions for constellation_style="custom".
        Format: {"ConstellationName": [hip1, hip2, hip3, ...]} or
                {"ConstellationName": [[hip1, hip2], [hip3, hip4, hip5]]} for multiple lines.
        Each sequence creates line segments connecting consecutive HIP IDs.
    
    Returns
    -------
    segments : list of np.ndarray
        Line segments in RA/Dec coordinates, each segment is shape (2, 2)
    """
    # Determine which constellation file to load
    if constellation_style.lower() == "custom":
        if custom_constellations is None:
            raise ValueError("constellation_style='custom' requires custom_constellations dict")
        filename = None
    elif constellation_style.lower() == "chinese":
        filename = os.path.join(SKY_MAP_ROOT, "constellations_chinese.txt")
    elif constellation_style.lower() == "western":
        filename = os.path.join(SKY_MAP_ROOT, "constellations_rey.txt")
    else:
        raise ValueError(f"constellation_style must be 'western', 'chinese', or 'custom', got '{constellation_style}'")
    
    hip_lookup = _hip_lookup_table_with_mag()
    
    segments = []
    
    # Helper function to create segments from HIP sequences
    def create_segments_from_hips(hip_sequences):
        """Create line segments from HIP star sequences."""
        segs = []
        for line_sequence in hip_sequences:
            # Handle both single values and nested lists
            if isinstance(line_sequence, (list, tuple)):
                hips_in_line = line_sequence
            else:
                hips_in_line = [line_sequence]
            
            # Create segments between consecutive HIP IDs
            for j in range(len(hips_in_line) - 1):
                hip1 = hips_in_line[j]
                hip2 = hips_in_line[j + 1]
                
                # Skip if either star not in catalog
                if hip1 not in hip_lookup or hip2 not in hip_lookup:
                    continue
                
                ra1, dec1, _ = hip_lookup[hip1]
                ra2, dec2, _ = hip_lookup[hip2]
                
                # Use shortest angular path (from sky.py line 316-317)
                delta_ra = ((ra2 - ra1 + 180.0) % 360.0) - 180.0
                
                # Simple case: no boundary crossing
                if abs(delta_ra) <= 180 and 0 <= ra1 + delta_ra <= 360:
                    segs.append(np.array([[ra1, dec1], [ra1 + delta_ra, dec2]]))
                else:
                    # Wrapping case: segment crosses the 0/360 boundary
                    if delta_ra > 0:
                        crossing = 360
                        t = (crossing - ra1) / delta_ra
                    else:
                        crossing = 0
                        t = (crossing - ra1) / delta_ra
                    
                    dec_cross = dec1 + t * (dec2 - dec1)
                    segs.append(np.array([[ra1, dec1], [crossing, dec_cross]]))
                    
                    ra2_mapped = ra2 % 360
                    opposite = 0 if crossing == 360 else 360
                    segs.append(np.array([[opposite, dec_cross], [ra2_mapped, dec2]]))
        
        return segs
    
    # Process based on constellation style
    if constellation_style.lower() == "custom":
        # Custom constellations provided as dict
        for const_name, hip_data in custom_constellations.items():
            if isinstance(hip_data, (list, tuple)):
                # Check if it's a list of lists (multiple line sequences) or single sequence
                if len(hip_data) > 0 and isinstance(hip_data[0], (list, tuple)):
                    # Multiple line sequences: [line1, line2, ...]
                    segments.extend(create_segments_from_hips(hip_data))
                else:
                    # Single sequence: [hip1, hip2, hip3, ...]
                    segments.extend(create_segments_from_hips([hip_data]))
            else:
                raise ValueError(f"Custom constellation '{const_name}' must have list/tuple value, got {type(hip_data)}")
        
        print(f"Loaded {len(custom_constellations)} custom constellations")
    
    else:
        # File-based constellations
        with open(filename, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                
                parts = line.split()
                n_segments = int(parts[1])
                hips = list(map(int, parts[2:]))
                
                # Create segments from the HIP pairs stored in the file
                for i in range(n_segments):
                    hip1 = hips[2*i]
                    hip2 = hips[2*i + 1]
                    
                    # Skip if either star not in catalog
                    if hip1 not in hip_lookup or hip2 not in hip_lookup:
                        continue
                    
                    ra1, dec1, _ = hip_lookup[hip1]
                    ra2, dec2, _ = hip_lookup[hip2]
                    
                    # Use shortest angular path (from sky.py line 316-317)
                    delta_ra = ((ra2 - ra1 + 180.0) % 360.0) - 180.0
                    
                    # Simple case: no boundary crossing
                    if abs(delta_ra) <= 180 and 0 <= ra1 + delta_ra <= 360:
                        segments.append(np.array([[ra1, dec1], [ra1 + delta_ra, dec2]]))
                    else:
                        # Wrapping case: segment crosses the 0/360 boundary
                        if delta_ra > 0:
                            crossing = 360
                            t = (crossing - ra1) / delta_ra
                        else:
                            crossing = 0
                            t = (crossing - ra1) / delta_ra
                        
                        dec_cross = dec1 + t * (dec2 - dec1)
                        segments.append(np.array([[ra1, dec1], [crossing, dec_cross]]))
                        
                        ra2_mapped = ra2 % 360
                        opposite = 0 if crossing == 360 else 360
                        segments.append(np.array([[opposite, dec_cross], [ra2_mapped, dec2]]))
    
    return segments


def plot_flat_constellation_projection(
    output_path,
    page_width_cm=14.5,
    page_height_cm=19.0,
    page_dpi=300,
    background="white",
    font_family="sans-serif",
    font_name="Futura",
    mag_limit=6.0,
    plot_stars=True,
    supernovae=None,
    n_supernova_contours=4,
    rotation_degrees=0,
    plot_text=True,
    orientation="portrait",
    transparent=False,
    constellation_style="western",
    custom_constellations=None,
):
    """Generate a flat equirectangular all-sky constellation projection.
    
    Parameters
    ----------
    output_path : str
        Path where to save the output PDF
    page_width_cm : float
        Page width in centimeters (default: 14.5 cm for A4 portrait)
    page_height_cm : float
        Page height in centimeters (default: 19.0 cm for A4 portrait)
    page_dpi : int
        Resolution in DPI (default: 300 for print quality)
    background : str
        Background color ('white' or dark color code, default: 'white')
    font_family : str
        Font family (default: 'sans-serif')
    font_name : str
        Font name (default: 'Futura')
    mag_limit : float
        Magnitude limit for stars to display (default: 3.0, lower = brighter)
    supernovae : Supernovae, optional
        Optional Supernovae instance to overlay galactic contours (default: None)
    n_supernova_contours : int
        Number of contour levels for supernovae (default: 4)
    rotation_degrees : float
        Rotate the entire figure by this many degrees clockwise (default: 0)
    plot_text : bool
        Whether to display text labels (Crab Supernova label) (default: True)
    orientation : str
        Page orientation: 'portrait' (width < height) or 'landscape' (width > height)
        If 'landscape', the width and height dimensions are swapped (default: 'portrait')
    transparent : bool
        If True, the background is transparent (only affects PNG/PNG-like outputs)
        If False, uses the background color parameter (default: False)
    constellation_style : str
        Which constellation set to use:
        - "western" (default): uses constellations_rey.txt from SKY_MAP_ROOT
        - "chinese": uses constellations_chinese.txt from SKY_MAP_ROOT
        - "custom": uses custom_constellations dict (must be provided)
    custom_constellations : dict, optional
        Custom constellation definitions for constellation_style="custom".
        Format: {"ConstellationName": [hip1, hip2, hip3, ...]} or
                {"ConstellationName": [[hip1, hip2], [hip3, hip4, hip5]]} for multiple lines.
        Each sequence creates line segments connecting consecutive HIP IDs (default: None)
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Handle orientation: swap dimensions if landscape
    width_cm = page_width_cm
    height_cm = page_height_cm
    
    if orientation.lower() == "landscape":
        # Swap width and height for landscape orientation
        width_cm, height_cm = height_cm, width_cm
        print(f"Using landscape orientation: {width_cm} cm × {height_cm} cm")
    elif orientation.lower() == "portrait":
        print(f"Using portrait orientation: {width_cm} cm × {height_cm} cm")
    else:
        raise ValueError(f"orientation must be 'portrait' or 'landscape', got '{orientation}'")
    
    # Convert page dimensions to inches
    figwidth_inch = width_cm / 2.54
    figheight_inch = height_cm / 2.54
    
    set_plot_style(background, font_family, font_name)
    
    # Create figure with equirectangular projection
    fig, ax = plt.subplots(figsize=(figwidth_inch, figheight_inch), dpi=page_dpi)
    
    # Set background (with transparency option)
    # First determine text color based on background color
    if background == "white":
        text_color = 'black'
    else:
        text_color = 'white'
    
    # Then apply background rendering, considering transparency
    if transparent:
        # Transparent background - don't set facecolor to allow transparency through
        fig.patch.set_alpha(0)
        ax.set_facecolor((0, 0, 0, 0))  # RGBA with alpha=0 for transparency
        print(f"Using transparent background with text color: {text_color}")
    elif background == "white":
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    else:
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
    
    # Setup the projection: RA (0-360°) on x-axis, Dec (-90 to +90°) on y-axis
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    # Remove axes frame and labels
    ax.axis('off')
    
    # Plot galactic supernovae contours if provided (at bottom layer)
    if supernovae is not None:
        try:
            from matplotlib.colors import to_rgba
            from astropy.coordinates import SkyCoord, Galactic, CartesianRepresentation
            import astropy.units as u
            
            gal_coords = supernovae.galactic_coords
            if gal_coords is not None:
                # Convert galactic Cartesian (x, y, z in kpc) to RA/Dec
                x_gal = gal_coords[:, 0]
                y_gal = gal_coords[:, 1]
                z_gal = gal_coords[:, 2]
                
                # Create SkyCoord in galactic frame using Cartesian coordinates
                coords_gal = SkyCoord(
                    CartesianRepresentation(
                        x=x_gal * u.kpc,
                        y=y_gal * u.kpc,
                        z=z_gal * u.kpc
                    ),
                    frame=Galactic,
                )
                
                # Transform to ICRS (J2000)
                coords_icrs = coords_gal.transform_to("icrs")
                ra_deg = coords_icrs.ra.deg
                dec_deg = coords_icrs.dec.deg
                
                # Sample only the N closest supernovae (like in sky.py)
                n_background_supernovae = 20000
                if hasattr(supernovae, 'distance') and supernovae.distance is not None:
                    distances = np.asarray(supernovae.distance)
                    sorted_indices = np.argsort(distances)
                    n_sample = min(n_background_supernovae, len(sorted_indices))
                    sample_indices = sorted_indices[:n_sample]
                    ra_deg = ra_deg[sample_indices]
                    dec_deg = dec_deg[sample_indices]
                    print(f"Sampled {n_sample} closest supernovae for contour")
                else:
                    print(f"Using all {len(ra_deg)} supernovae for contour (no distance data)")
                
                # Create 2D histogram in RA/Dec space (match sky.py: bins=320)
                bins = 320
                hist_range = [[0, 360], [-90, 90]]
                h, ra_edges, dec_edges = np.histogram2d(ra_deg, dec_deg, bins=bins, range=hist_range)
                
                # Smooth with Gaussian kernel
                k_radius = 3
                k_sigma = 1.2
                k_axis = np.arange(-k_radius, k_radius + 1)
                kernel = np.exp(-(k_axis**2) / (2.0 * k_sigma**2))
                kernel /= kernel.sum()
                
                h_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=h)
                h_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=h_smooth)
                
                ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
                dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
                ra_grid, dec_grid = np.meshgrid(ra_centers, dec_centers)
                
                # Compute contour levels from density quantiles (match sky.py: [0.995, 0.80, 0.50, 0.25])
                h_flat = h_smooth.ravel()
                h_flat = h_flat[h_flat > 0]
                
                if h_flat.size > 0:
                    blue_probs = [0.995, 0.80, 0.50, 0.25]
                    vals = np.sort(h_flat)[::-1]
                    cdf = np.cumsum(vals) / np.sum(vals)
                    
                    thr_shared = []
                    for p in blue_probs:
                        idx = np.searchsorted(cdf, p, side="left")
                        idx = min(idx, vals.size - 1)
                        thr_shared.append(float(vals[idx]))
                    
                    levels_shared = np.sort(np.array(thr_shared, dtype=float))
                    top_shared = max(levels_shared[-1] * 1.001, np.max(h_flat) * 1.001)
                    fill_levels_shared = np.concatenate([levels_shared, [top_shared]])
                    
                    # Define contour colors with smooth interpolation (match sky.py)
                    blue_bases = ["#486ac8", "#488af4", "#60a5fa", "#bfdbfe"]
                    fill_colors = [
                        to_rgba(blue_bases[0], alpha=0.20),
                        to_rgba(blue_bases[1], alpha=0.40),
                        to_rgba(blue_bases[2], alpha=0.62),
                        to_rgba(blue_bases[3], alpha=0.88),
                    ]
                    
                    # Create smooth transitions by interpolating colors in RGBA space
                    n_per_segment = 4  # Create 4 intermediate colors between each pair
                    smooth_colors = []
                    
                    for i in range(len(fill_colors) - 1):
                        color_a = np.array(fill_colors[i])
                        color_b = np.array(fill_colors[i + 1])
                        
                        # Interpolate between current and next color
                        for j in range(n_per_segment):
                            alpha = j / n_per_segment
                            interp_color = color_a * (1 - alpha) + color_b * alpha
                            smooth_colors.append(tuple(interp_color))
                    
                    # Add the last color
                    smooth_colors.append(fill_colors[-1])
                    
                    # Create levels to match the number of colors
                    smooth_levels = np.linspace(fill_levels_shared[0], fill_levels_shared[-1], len(smooth_colors) + 1)
                    
                    # Plot contours at bottom layer
                    ax.contourf(ra_grid, dec_grid, h_smooth.T, levels=smooth_levels, colors=smooth_colors, 
                                antialiased=True, zorder=0)
                    print(f"✓ Plotted galactic supernovae contours with {len(blue_probs)} levels")
        except Exception as e:
            print(f"Warning: Could not plot supernovae contours: {e}")
    
    # Load Hipparcos data with magnitudes
    hip_lookup = _hip_lookup_table_with_mag()
    print(f"Loaded {len(hip_lookup)} stars from Hipparcos catalog")
    
    # Plot stars by magnitude limit (not fixed count)
    if plot_stars:
        stars_list = []
        
        for hip, (ra, dec, mag) in hip_lookup.items():
            if mag <= mag_limit:  # Use magnitude limit instead of fixed count
                stars_list.append((mag, ra, dec))
        
        if stars_list:
            mag_all, ra_all, dec_all = zip(*stars_list)
            ra_all = np.asarray(ra_all)
            dec_all = np.asarray(dec_all)
            mag_all = np.asarray(mag_all)
            
            print(f"DEBUG: Selected {len(ra_all)} stars with mag <= {mag_limit}")
            if len(ra_all) > 0:
                print(f"DEBUG: Magnitude range: {mag_all.min():.2f} to {mag_all.max():.2f}")
                print(f"DEBUG: RA range: {ra_all.min():.2f} to {ra_all.max():.2f}")
                print(f"DEBUG: Dec range: {dec_all.min():.2f} to {dec_all.max():.2f}")
                print(f"DEBUG: NaN count in RA: {np.isnan(ra_all).sum()}")
                print(f"DEBUG: NaN count in Dec: {np.isnan(dec_all).sum()}")
                print(f"DEBUG: Sample star: RA={ra_all[0]:.2f}, Dec={dec_all[0]:.2f}, mag={mag_all[0]:.2f}")
            
            # Size stars using inverse magnitude scale: brighter = larger
            # Formula: size = 60 * 10^(-0.4*mag), clipped to reasonable range
            # At mag=-1: 60 * 10^0.4 ≈ 151 → clipped to 10
            # At mag=3:  60 * 10^-1.2 ≈ 3.8 → stays as 3.8
            # At mag=8:  60 * 10^-3.2 ≈ 0.05 → faintest stars are very tiny
            star_sizes = np.clip(60 * 10 ** (-0.4 * mag_all), 0.05, 10)
            
            # Calculate alpha values: brightest stars full (1.0), faintest at 0.7
            # Normalize magnitude to 0-1 range, then invert so brighter = higher
            mag_min, mag_max = mag_all.min(), mag_all.max()
            norm_mag = (mag_all - mag_min) / (mag_max - mag_min) if mag_max > mag_min else np.zeros_like(mag_all)
            star_alphas = 1.0 - 0.3 * norm_mag  # brightest (norm=0) → 1.0, faintest (norm=1) → 0.7
            
            print(f"DEBUG: Star sizes range: {star_sizes.min():.2f} to {star_sizes.max():.2f}")
            print(f"DEBUG: Star alphas range: {star_alphas.min():.2f} to {star_alphas.max():.2f}")
            
            # Plot stars in 10 alpha bins for efficiency
            n_bins = 10
            alpha_bins = np.linspace(0.7, 1.0, n_bins + 1)
            for i in range(len(alpha_bins) - 1):
                mask = (star_alphas >= alpha_bins[i]) & (star_alphas < alpha_bins[i + 1])
                if mask.any():
                    alpha_val = (alpha_bins[i] + alpha_bins[i + 1]) / 2
                    # Use conditional edge colors like sky.py: none for dark, "#b1cbed" for light
                    edge_color = "none" if background == "black" else "#b1cbed"
                    ax.scatter(ra_all[mask], dec_all[mask], s=star_sizes[mask], c="white", 
                              alpha=alpha_val, edgecolors=edge_color, linewidths=0.2 if background == "white" else 0.0, 
                              zorder=10)
            
            print(f"✓ Plotted {len(ra_all)} stars with mag <= {mag_limit} sized by magnitude, alpha {star_alphas.min():.2f}-{star_alphas.max():.2f}")
        else:
            print(f"⊘ No stars found with mag <= {mag_limit}")
    else:
        print(f"⊘ Stars disabled (plot_stars=False)")
    
    # Plot constellation stick figures
    try:
        stick_segments = _constellation_stick_segments_flat(constellation_style=constellation_style, 
                                                            custom_constellations=custom_constellations)
        if stick_segments:
            ax.add_collection(LineCollection(stick_segments, colors="#6ca3eb", 
                                            alpha=1.0, linewidth=0.75, zorder=2,
                                            joinstyle="round", capstyle="round"))
            print(f"✓ Plotted {len(stick_segments)} constellation stick figure segments ({constellation_style})")
        else:
            print(f"⊘ No stick figure segments found")
    except Exception as e:
        print(f"Error plotting stick figures: {e}")
    
    # Plot constellation borders from file
    # Determine which border file to load based on constellation style
    if constellation_style.lower() == "custom":
        # Custom constellations don't have border files
        print("⊘ Custom constellation boundaries not available (no border file)")
        border_file = None
    elif constellation_style.lower() == "chinese":
        # Chinese boundary data is sparse and incomplete (only 28 points vs 7121 for Western)
        # Skip plotting borders for Chinese constellations until complete data is available
        print("⊘ Chinese constellation boundaries not available (data file incomplete)")
        border_file = None
    elif constellation_style.lower() == "western":
        border_file = os.path.join(SKY_MAP_ROOT, "lines_in_18.txt")
    else:
        border_file = None
    
    if border_file and os.path.exists(border_file):
        try:
            if not _ASTROPY_AVAILABLE:
                print("Warning: Astropy not available, skipping constellation borders")
            else:
                borders = []
                ra_hrs = []
                dec_degs = []
                
                with open(border_file, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        ra_hr, dec_deg, border = line.split()
                        borders.append(border)
                        ra_hrs.append(float(ra_hr))
                        dec_degs.append(float(dec_deg))
                
                ra_hrs = np.asarray(ra_hrs)
                dec_degs = np.asarray(dec_degs)
                
                # Precess B1875 -> J2000
                coords_b1875 = SkyCoord(
                    ra=ra_hrs * 15.0 * u.deg,
                    dec=dec_degs * u.deg,
                    frame=FK4(equinox="B1875"),
                )
                coords_j2000 = coords_b1875.transform_to("icrs")
                ra_deg_j2000 = coords_j2000.ra.deg
                dec_deg_j2000 = coords_j2000.dec.deg
                
                # Handle Chinese vs Western borders differently
                all_border_segments = []
                
                if constellation_style.lower() == "chinese":
                    # Chinese borders: each line is a single point representing a boundary marker
                    # Connect consecutive points to form boundary segments
                    for i in range(len(ra_deg_j2000) - 1):
                        ra1, dec1 = ra_deg_j2000[i], dec_deg_j2000[i]
                        ra2, dec2 = ra_deg_j2000[i+1], dec_deg_j2000[i+1]
                        
                        # Create segment between consecutive points
                        if not _check_ra_wrapping(ra1, ra2):
                            all_border_segments.append(np.array([[ra1, dec1], [ra2, dec2]]))
                        else:
                            # Handle RA wrapping
                            if ra1 > 180:
                                all_border_segments.append(np.array([[ra1, dec1], [360, dec1]]))
                                all_border_segments.append(np.array([[0, dec2], [ra2, dec2]]))
                            else:
                                all_border_segments.append(np.array([[ra1, dec1], [0, dec1]]))
                                all_border_segments.append(np.array([[360, dec2], [ra2, dec2]]))
                    
                    print(f"DEBUG: Created {len(all_border_segments)} Chinese border segments from {len(ra_deg_j2000)} points")
                else:
                    # Western borders: group points by constellation pair to form continuous paths
                    borders_by_constellation = {}
                    current_border = None
                    current_ra = []
                    current_dec = []
                    
                    for border, ra, dec in zip(borders, ra_deg_j2000, dec_deg_j2000):
                        if border != current_border and current_border is not None:
                            # Store this constellation's border path
                            if len(current_ra) > 1:
                                borders_by_constellation[current_border] = (current_ra, current_dec)
                            current_ra = []
                            current_dec = []
                        
                        current_border = border
                        current_ra.append(ra)
                        current_dec.append(dec)
                    
                    # Store final border
                    if len(current_ra) > 1:
                        borders_by_constellation[current_border] = (current_ra, current_dec)
                    
                    # Build continuous paths with RA wrapping detection
                    for constellation, (ras, decs) in borders_by_constellation.items():
                        constellation_path = []
                        current_path = []
                        
                        for j in range(len(ras) - 1):
                            ra1, dec1 = ras[j], decs[j]
                            ra2, dec2 = ras[j+1], decs[j+1]
                            
                            if not _check_ra_wrapping(ra1, ra2):
                                # Normal segment - add to current path
                                current_path.append([ra1, dec1])
                            else:
                                # RA wrap detected - extend to edges and create two segments
                                if current_path:
                                    current_path.append([ra1, dec1])
                                    constellation_path.append(np.array(current_path))
                                    current_path = []
                                
                                # Add segment from ra1 to edge (RA=0 or 360)
                                if ra1 > 180:
                                    constellation_path.append(np.array([[ra1, dec1], [360, dec1]]))
                                    current_path = [[0, dec2]]
                                else:
                                    constellation_path.append(np.array([[ra1, dec1], [0, dec1]]))
                                    current_path = [[360, dec2]]
                        
                        # Add final point and finalize path
                        if current_path:
                            current_path.append([ras[-1], decs[-1]])
                            constellation_path.append(np.array(current_path))
                        
                        all_border_segments.extend(constellation_path)
                    
                    print(f"DEBUG: Created {len(all_border_segments)} Western border segments")
                
                # Add borders with dashes
                if all_border_segments:
                    print(f"DEBUG: Total border segments: {len(all_border_segments)}")
                    # Use conditional colors like sky.py: light blue for dark, dark blue for light
                    border_color = "#b1cbed" if background == "black" else "#1e293b"
                    ax.add_collection(LineCollection(all_border_segments, colors=border_color, 
                                                     alpha=0.5, linewidth=0.5, zorder=1,
                                                     linestyle=(0, (5, 5)),
                                                     joinstyle="round", capstyle="round"))
                
                print(f"✓ Constellation borders plotted from {border_file} ({constellation_style})")
        except Exception as e:
            print(f"Error plotting borders: {e}")
    elif border_file:
        print(f"Warning: Border file not found at {border_file}")
    else:
        print(f"Warning: Unable to determine border file for constellation style '{constellation_style}'")
    
    # Mark Crab Nebula (M1) with red 4-pointed star
    try:
        crab_ra = 83.633  # degrees, J2000
        crab_dec = 22.015  # degrees, J2000
        
        # Create a simple 4-pointed star marker with skinny points
        # Outer points at radius 1.0, inner valleys at radius 0.25
        star_verts = [
            (0.0, 1.0),   # top point
            (0.2, 0.2),   # upper-right valley
            (1.0, 0.0),   # right point
            (0.2, -0.2),  # lower-right valley
            (0.0, -1.0),  # bottom point
            (-0.2, -0.2), # lower-left valley
            (-1.0, 0.0),  # left point
            (-0.2, 0.2),  # upper-left valley
            (0.0, 1.0),   # close path
        ]
        codes = [Path.MOVETO] + [Path.LINETO] * 7 + [Path.CLOSEPOLY]
        star_marker = Path(star_verts, codes)
        
        # Draw 4-pointed star at Crab Nebula location using text color (white for dark, black for light)
        ax.scatter(crab_ra, crab_dec, s=550, marker=star_marker, c=text_color, alpha=1.0, 
                  edgecolors='none', linewidth=0.0, zorder=100, label='Crab Nebula (M1)')
        print(f"✓ Crab Nebula (M1) marked at RA={crab_ra:.3f}°, Dec={crab_dec:.3f}° with 4-pointed star (color: {text_color})")
    except Exception as e:
        print(f"Warning: Could not mark Crab Nebula: {e}")
    
    # Ensure axes limits are set after adding collections (LineCollection needs explicit limits)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    # Add Crab Supernova label near bottom center (if text is enabled)
    if plot_text:
        ax.text(180, -82, "Crab Supernova\n4th July, 1054", 
               ha='center', va='top', fontsize=7, fontfamily='Futura', 
               color=text_color, zorder=50)
    
    # Apply rotation if requested (rotate around center of projection at RA=180, Dec=0)
    if rotation_degrees != 0:
        from matplotlib.transforms import Affine2D
        
        # Convert rotation to radians (negative for clockwise rotation)
        rotation_rad = -np.radians(rotation_degrees)
        
        # Center point in data coordinates
        center_ra = 180.0
        center_dec = 0.0
        
        # Create rotation transform centered at (center_ra, center_dec)
        # Translate to origin -> Rotate -> Translate back
        trans = Affine2D()
        trans.translate(-center_ra, -center_dec)
        trans.rotate(rotation_rad)
        trans.translate(center_ra, center_dec)
        
        # Apply rotation to all collections (stick figures, borders, contours, supernovae, stars)
        for collection in ax.collections:
            if hasattr(collection, 'set_transform'):
                collection.set_transform(trans + ax.transData)
    
    # Make axes fill entire figure (no margins)
    ax.set_position([0, 0, 1, 1])
    
    # Explicitly position axes to fill entire figure (full bleed)
    ax.set_position([0, 0, 1, 1])
    
    # Ensure canvas is properly configured
    canvas = fig.canvas
    canvas.draw()
    
    # Save to PNG first for debugging (no tight_layout, full bleed to edges)
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=page_dpi, bbox_inches=None, 
                facecolor=background if not transparent else None, 
                transparent=transparent,
                edgecolor='none', pad_inches=0)
    print(f"✓ Saved debug PNG to: {png_path}")
    
    # Save to PDF (no tight_layout, full bleed to edges)
    fig.savefig(output_path, dpi=page_dpi, bbox_inches=None, 
                facecolor=background if not transparent else None,
                transparent=transparent,
                edgecolor='none', pad_inches=0)
    print(f"✓ Flat sky projection saved to: {output_path}")
    
    return fig, ax


def plot_galactic_supernovae_flat_contour(
    supernovae,
    output_path=None,
    figsize=(14.5, 19.0),
    page_dpi=300,
    background="white",
    font_family="sans-serif",
    font_name="Futura",
    n_contours=4,
    cmap_colors=None,
):
    """Plot galactic supernovae distribution as a contour in flat RA/Dec projection.
    
    Projects the galactic coordinates from a Supernovae instance into RA/Dec (J2000),
    creates a 2D density histogram, and displays it as contours in an equirectangular
    flat-sky projection.
    
    Parameters
    ----------
    supernovae : Supernovae
        Instance with galactic_coords (x, y, z in kpc) and optionally distance data
    output_path : str, optional
        Path to save output. If None, only returns fig/ax
    figsize : tuple
        Figure size in cm (width, height). Default (14.5, 19.0) for A4 portrait
    page_dpi : int
        DPI for output (default: 300)
    background : str
        Background color ('white' or hex code, default: 'white')
    font_family : str
        Font family (default: 'sans-serif')
    font_name : str
        Font name (default: 'Futura')
    n_contours : int
        Number of contour levels (default: 4)
    cmap_colors : list, optional
        List of colors for contours. If None, uses blue gradient
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    from matplotlib.colors import to_rgba
    
    if not _ASTROPY_AVAILABLE:
        raise ImportError("Astropy is required for coordinate transformations")
    
    # Convert figsize from cm to inches
    figwidth_inch = figsize[0] / 2.54
    figheight_inch = figsize[1] / 2.54
    
    set_plot_style(background, font_family, font_name)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(figwidth_inch, figheight_inch), dpi=page_dpi)
    
    # Set background
    if background == "white":
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        text_color = 'black'
    else:
        fig.patch.set_facecolor(background)
        ax.set_facecolor(background)
        text_color = 'white'
    
    # Setup equirectangular projection: RA (0-360°) on x-axis, Dec (-90 to +90°) on y-axis
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.axis('off')
    
    # Get galactic coordinates and convert to RA/Dec
    gal_coords = supernovae.galactic_coords
    if gal_coords is None:
        raise ValueError("Supernovae instance has no galactic coordinates")
    
    # Convert galactic Cartesian (x, y, z in kpc) to RA/Dec
    # Using Astropy: galactic frame -> ICRS (J2000)
    from astropy.coordinates import SkyCoord, Galactic, CartesianRepresentation
    import astropy.units as u
    
    x_gal = gal_coords[:, 0]
    y_gal = gal_coords[:, 1]
    z_gal = gal_coords[:, 2]
    
    # Create SkyCoord in galactic frame using Cartesian coordinates
    coords_gal = SkyCoord(
        CartesianRepresentation(
            x=x_gal * u.kpc,
            y=y_gal * u.kpc,
            z=z_gal * u.kpc
        ),
        frame=Galactic,
    )
    
    # Transform to ICRS (J2000)
    coords_icrs = coords_gal.transform_to("icrs")
    ra_deg = coords_icrs.ra.deg
    dec_deg = coords_icrs.dec.deg
    
    # Sample only the N closest supernovae (like in sky.py)
    n_background_supernovae = 20000
    if hasattr(supernovae, 'distance') and supernovae.distance is not None:
        distances = np.asarray(supernovae.distance)
        sorted_indices = np.argsort(distances)
        n_sample = min(n_background_supernovae, len(sorted_indices))
        sample_indices = sorted_indices[:n_sample]
        ra_deg = ra_deg[sample_indices]
        dec_deg = dec_deg[sample_indices]
        print(f"Sampled {n_sample} closest supernovae for contour")
    else:
        print(f"Using all {len(ra_deg)} supernovae for contour (no distance data)")
    
    # Create 2D histogram in RA/Dec space (match sky.py: bins=320)
    bins = 320
    hist_range = [[0, 360], [-90, 90]]
    h, ra_edges, dec_edges = np.histogram2d(ra_deg, dec_deg, bins=bins, range=hist_range)
    
    # Smooth with Gaussian kernel
    k_radius = 3
    k_sigma = 1.2
    k_axis = np.arange(-k_radius, k_radius + 1)
    kernel = np.exp(-(k_axis**2) / (2.0 * k_sigma**2))
    kernel /= kernel.sum()
    
    h_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=h)
    h_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=h_smooth)
    
    ra_centers = 0.5 * (ra_edges[:-1] + ra_edges[1:])
    dec_centers = 0.5 * (dec_edges[:-1] + dec_edges[1:])
    ra_grid, dec_grid = np.meshgrid(ra_centers, dec_centers)
    
    # Compute contour levels from density quantiles (match sky.py: [0.995, 0.80, 0.50, 0.25])
    h_flat = h_smooth.ravel()
    h_flat = h_flat[h_flat > 0]
    
    if h_flat.size == 0:
        print("Warning: No density data to plot")
        return fig, ax
    
    blue_probs = [0.995, 0.80, 0.50, 0.25]
    vals = np.sort(h_flat)[::-1]
    cdf = np.cumsum(vals) / np.sum(vals)
    
    thr_shared = []
    for p in blue_probs:
        idx = np.searchsorted(cdf, p, side="left")
        idx = min(idx, vals.size - 1)
        thr_shared.append(float(vals[idx]))
    
    levels_shared = np.sort(np.array(thr_shared, dtype=float))
    top_shared = max(levels_shared[-1] * 1.001, np.max(h_flat) * 1.001)
    fill_levels_shared = np.concatenate([levels_shared, [top_shared]])
    
    # Define contour colors with smooth interpolation (match sky.py)
    blue_bases = ["#486ac8", "#488af4", "#60a5fa", "#bfdbfe"]
    fill_colors = [
        to_rgba(blue_bases[0], alpha=0.20),
        to_rgba(blue_bases[1], alpha=0.40),
        to_rgba(blue_bases[2], alpha=0.62),
        to_rgba(blue_bases[3], alpha=0.88),
    ]
    
    # Create smooth transitions by interpolating colors in RGBA space
    n_per_segment = 4  # Create 4 intermediate colors between each pair
    smooth_colors = []
    
    for i in range(len(fill_colors) - 1):
        color_a = np.array(fill_colors[i])
        color_b = np.array(fill_colors[i + 1])
        
        # Interpolate between current and next color
        for j in range(n_per_segment):
            alpha = j / n_per_segment
            interp_color = color_a * (1 - alpha) + color_b * alpha
            smooth_colors.append(tuple(interp_color))
    
    # Add the last color
    smooth_colors.append(fill_colors[-1])
    
    # Create levels to match the number of colors
    smooth_levels = np.linspace(fill_levels_shared[0], fill_levels_shared[-1], len(smooth_colors) + 1)
    
    # Plot contours
    ax.contourf(ra_grid, dec_grid, h_smooth.T, levels=smooth_levels, colors=smooth_colors, 
                antialiased=True, zorder=0)
    
    ax.set_position([0, 0, 1, 1])
    fig.canvas.draw()
    
    if output_path:
        fig.savefig(output_path, dpi=page_dpi, bbox_inches=None,
                   facecolor=background, edgecolor='none', pad_inches=0)
        print(f"✓ Galactic supernovae contour saved to: {output_path}")
    
    return fig, ax
