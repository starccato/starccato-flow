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
    
    Similar to sky.py's _constellation_stick_segments but for flat RA/Dec coordinates.
    Returns line segments that avoid RA wrapping artifacts at 0/360° boundary.
    
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
                
                # Skip segment if it crosses RA boundary (0/360 wraparound)
                if _check_ra_wrapping(ra1, ra2):
                    continue
                
                segments.append(np.array([[ra1, dec1], [ra2, dec2]]))
    
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
            star_sizes = np.clip(60 * 10 ** (-0.4 * mag_all), 0.5, 12)
            
            print(f"DEBUG: Star sizes range: {star_sizes.min():.2f} to {star_sizes.max():.2f}")
            
            # Plot stars with solid opacity on top layer (above constellation lines)
            scatter = ax.scatter(ra_all, dec_all, s=star_sizes, c=text_color, alpha=0.9, 
                                edgecolors=text_color, linewidth=0.1, zorder=10)
            
            print(f"✓ Plotted {len(ra_all)} stars with mag <= {mag_limit} sized by magnitude")
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
                
                # Collect borders as line segments per constellation
                border_segments = []
                current_border = None
                current_ra = []
                current_dec = []
                
                for border, ra, dec in zip(borders, ra_deg_j2000, dec_deg_j2000):
                    if border != current_border and current_border is not None:
                        # Process previous border constellation
                        if len(current_ra) > 1:
                            # Check for RA wrapping in this border line
                            has_wrapping = False
                            for j in range(len(current_ra) - 1):
                                if _check_ra_wrapping(current_ra[j], current_ra[j+1]):
                                    has_wrapping = True
                                    break
                            
                            # Only collect if no wrapping detected
                            if not has_wrapping:
                                # Add each segment of this border
                                for j in range(len(current_ra) - 1):
                                    border_segments.append(np.array([[current_ra[j], current_dec[j]], 
                                                                    [current_ra[j+1], current_dec[j+1]]]))
                        current_ra = []
                        current_dec = []
                    
                    current_border = border
                    current_ra.append(ra)
                    current_dec.append(dec)
                
                # Process final border
                if len(current_ra) > 1:
                    has_wrapping = False
                    for j in range(len(current_ra) - 1):
                        if _check_ra_wrapping(current_ra[j], current_ra[j+1]):
                            has_wrapping = True
                            break
                    
                    if not has_wrapping:
                        for j in range(len(current_ra) - 1):
                            border_segments.append(np.array([[current_ra[j], current_dec[j]], 
                                                            [current_ra[j+1], current_dec[j+1]]]))
                
                # Add all border segments as a single LineCollection (dashed, matching sky.py)
                if border_segments:
                    ax.add_collection(LineCollection(border_segments, colors="#1e293b", 
                                                     alpha=0.5, linewidth=0.5, zorder=1,
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
        
        # Draw red 4-pointed star at Crab Nebula location (no outline)
        ax.scatter(crab_ra, crab_dec, s=450, marker=star_marker, c='red', alpha=1.0, 
                  edgecolors='none', linewidth=0.0, zorder=100, label='Crab Nebula (M1)')
        print(f"✓ Crab Nebula (M1) marked at RA={crab_ra:.3f}°, Dec={crab_dec:.3f}° with 4-pointed star")
    except Exception as e:
        print(f"Warning: Could not mark Crab Nebula: {e}")
    
    # Ensure axes limits are set after adding collections (LineCollection needs explicit limits)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    
    # Save with tight layout
    plt.tight_layout(pad=0)
    
    # Ensure canvas is properly configured
    canvas = fig.canvas
    canvas.draw()
    
    # Save to PNG first for debugging
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=page_dpi, bbox_inches='tight', 
                facecolor=background, edgecolor='none', pad_inches=0.02)
    print(f"✓ Saved debug PNG to: {png_path}")
    
    # Save to PDF
    fig.savefig(output_path, dpi=page_dpi, bbox_inches='tight', 
                facecolor=background, edgecolor='none', pad_inches=0.02)
    print(f"✓ Flat sky projection saved to: {output_path}")
    
    return fig, ax
