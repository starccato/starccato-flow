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


def _constellation_stick_segments_flat():
    """Get constellation stick figure segments for flat equirectangular projection.
    
    Properly handles RA wrapping by:
    1. Using shortest angular path to determine intended direction
    2. If wrapping occurs, interpolates split point at boundary
    3. Draws segments on both sides of boundary for visual continuity
    
    Returns
    -------
    segments : list of np.ndarray
        Line segments in RA/Dec coordinates, each segment is shape (2, 2)
    """
    filename = os.path.join(SKY_MAP_ROOT, "constellations_rey.txt")
    hip_lookup = _hip_lookup_table_with_mag()
    
    segments = []
    
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            
            parts = line.split()
            n_segments = int(parts[1])
            hips = list(map(int, parts[2:]))
            
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
                    # Find where it crosses (0 or 360)
                    if delta_ra > 0:
                        # Going forward through 360 boundary
                        crossing = 360
                        t = (crossing - ra1) / delta_ra
                    else:
                        # Going backward through 0 boundary  
                        crossing = 0
                        t = (crossing - ra1) / delta_ra
                    
                    # Interpolated Dec at crossing point
                    dec_cross = dec1 + t * (dec2 - dec1)
                    
                    # Draw segment from ra1 to boundary
                    segments.append(np.array([[ra1, dec1], [crossing, dec_cross]]))
                    
                    # Draw segment from opposite boundary to ra2
                    # Map ra2 back to [0, 360] if needed
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
    
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Convert page dimensions to inches
    figwidth_inch = page_width_cm / 2.54
    figheight_inch = page_height_cm / 2.54
    
    set_plot_style(background, font_family, font_name)
    
    # Create figure with equirectangular projection
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
            # At mag=-1: 60 * 10^0.4 ≈ 151 → clipped to 12
            # At mag=3:  60 * 10^-1.2 ≈ 3.8 → stays as 3.8
            star_sizes = np.clip(60 * 10 ** (-0.4 * mag_all), 0.2, 10)
            
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
                    ax.scatter(ra_all[mask], dec_all[mask], s=star_sizes[mask], c=text_color, 
                              alpha=alpha_val, edgecolors='none', zorder=10)
            
            print(f"✓ Plotted {len(ra_all)} stars with mag <= {mag_limit} sized by magnitude, alpha {star_alphas.min():.2f}-{star_alphas.max():.2f}")
        else:
            print(f"⊘ No stars found with mag <= {mag_limit}")
    else:
        print(f"⊘ Stars disabled (plot_stars=False)")
    
    # Plot constellation stick figures
    try:
        stick_segments = _constellation_stick_segments_flat()
        if stick_segments:
            ax.add_collection(LineCollection(stick_segments, colors="#6ca3eb", 
                                            alpha=1.0, linewidth=0.75, zorder=2,
                                            joinstyle="round", capstyle="round"))
            print(f"✓ Plotted {len(stick_segments)} constellation stick figure segments")
        else:
            print(f"⊘ No stick figure segments found")
    except Exception as e:
        print(f"Error plotting stick figures: {e}")
    
    # Plot constellation borders from file
    border_file = os.path.join(SKY_MAP_ROOT, "lines_in_18.txt")
    if os.path.exists(border_file):
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
                
                # Collect borders grouped by constellation to enable proper dashing
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
                
                # Build constellation borders: unwrap RA, render segments individually
                # No polyline grouping - each segment is independent to avoid thickening artifacts
                all_border_segments = []
                
                for constellation, (ras, decs) in borders_by_constellation.items():
                    ras_array = np.array(ras)
                    decs_array = np.array(decs)
                    
                    # Unwrap RA values using shortest angular path
                    ras_unwrapped = np.zeros_like(ras_array)
                    ras_unwrapped[0] = ras_array[0]
                    
                    for j in range(1, len(ras_array)):
                        delta_ra = ((ras_array[j] - ras_unwrapped[j-1] + 180.0) % 360.0) - 180.0
                        ras_unwrapped[j] = ras_unwrapped[j-1] + delta_ra
                    
                    # Add segments individually - no grouping
                    for j in range(len(ras_unwrapped) - 1):
                        ra1 = ras_unwrapped[j]
                        ra2 = ras_unwrapped[j + 1]
                        dec1 = decs_array[j]
                        dec2 = decs_array[j + 1]
                        
                        # Only include if both points are within [0, 360]
                        if 0 <= ra1 <= 360 and 0 <= ra2 <= 360:
                            # Skip tiny segments
                            if abs(ra2 - ra1) > 0.01 or abs(dec2 - dec1) > 0.01:
                                all_border_segments.append(np.array([[ra1, dec1], [ra2, dec2]]))
                
                # Add borders with dashes
                if all_border_segments:
                    print(f"DEBUG: Total border segments: {len(all_border_segments)}")
                    ax.add_collection(LineCollection(all_border_segments, colors="#1e293b", 
                                                     alpha=0.35, linewidth=0.5, zorder=1,
                                                     linestyle=(0, (5, 5)),
                                                     joinstyle="round", capstyle="round"))
                
                print(f"✓ Constellation borders plotted from {border_file}")
        except Exception as e:
            print(f"Error plotting borders: {e}")
    else:
        print(f"Warning: Border file not found at {border_file}")
    
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
        
        # Draw black 4-pointed star at Crab Nebula location (no outline)
        ax.scatter(crab_ra, crab_dec, s=550, marker=star_marker, c='black', alpha=1.0, 
                  edgecolors='none', linewidth=0.0, zorder=100, label='Crab Nebula (M1)')
        print(f"✓ Crab Nebula (M1) marked at RA={crab_ra:.3f}°, Dec={crab_dec:.3f}° with 4-pointed star")
    except Exception as e:
        print(f"Warning: Could not mark Crab Nebula: {e}")
    
    # Ensure axes limits are set after adding collections (LineCollection needs explicit limits)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    # Add Crab Supernova label near bottom center
    ax.text(180, -82, "Crab Supernova\n4th July, 1054", 
           ha='center', va='top', fontsize=7, fontfamily='Futura', 
           color=text_color, zorder=50)
    
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
                facecolor=background, edgecolor='none', pad_inches=0)
    print(f"✓ Saved debug PNG to: {png_path}")
    
    # Save to PDF (no tight_layout, full bleed to edges)
    fig.savefig(output_path, dpi=page_dpi, bbox_inches=None, 
                facecolor=background, edgecolor='none', pad_inches=0)
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
