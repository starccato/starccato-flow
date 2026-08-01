"""Sky plotting utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from matplotlib.pylab import norm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Circle, Patch
from matplotlib.path import Path
from . import set_plot_style

from ..utils.defaults_plotting import (
    SIGNAL_COLOUR,
    GENERATED_SIGNAL_COLOUR,
    SIGNAL_LIM_UPPER,
    SIGNAL_LIM_LOWER,
    CM_TO_INCHES
)

from ..utils.defaults_general import SKY_MAP_ROOT

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Galactic, ICRS, FK4

    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


IMPORTANT_CONSTELLATIONS = {
    "Ori": "Orion",
    "Tau": "Taurus",
    "CMa": "Canis Major",
    "Cru": "Southern Cross",
}

import os


def _apply_astropy_ra_rotation_deg(
    ra_deg: np.ndarray | float,
    rotation_offset_deg: float = 0.0,
) -> np.ndarray | float:
    """Apply Supernovae RA rotation offset to Astropy-resolved RA values."""
    return np.mod(ra_deg + rotation_offset_deg, 360.0)


@lru_cache(maxsize=16)
def _get_betelgeuse_icrs_deg(rotation_offset_deg: float = 0.0) -> tuple[float, float, str]:
    """Return Betelgeuse ICRS (RA, Dec) in degrees and a source label."""
    if not _ASTROPY_AVAILABLE:
        return np.nan, np.nan, "unavailable"

    try:
        coord = SkyCoord.from_name("Betelgeuse")
        ra_deg = float(_apply_astropy_ra_rotation_deg(float(coord.ra.deg), rotation_offset_deg))
        return ra_deg, float(coord.dec.deg), "astropy"
    except Exception:
        return np.nan, np.nan, "unavailable"


@lru_cache(maxsize=512)
def _resolve_named_star_icrs_deg(name: str, rotation_offset_deg: float = 0.0) -> tuple[float, float] | None:
    """Resolve a star name to ICRS RA/Dec degrees using Astropy only."""
    if not _ASTROPY_AVAILABLE:
        return None

    try:
        coord = SkyCoord.from_name(name)
        ra_deg = float(_apply_astropy_ra_rotation_deg(float(coord.ra.deg), rotation_offset_deg))
        return ra_deg, float(coord.dec.deg)
    except Exception:
        return None


def _hpd_thresholds(
    density_grid: np.ndarray,
    valid_mask: np.ndarray,
    probs: Iterable[float],
) -> list[float]:
    """Return density thresholds whose highest-density regions enclose target probabilities."""
    vals = density_grid[valid_mask]
    vals = vals[vals > 0]
    probs = list(probs)
    if vals.size == 0:
        return [1.0 for _ in probs]

    vals = np.sort(vals)[::-1]
    cdf = np.cumsum(vals) / np.sum(vals)
    thresholds = []
    for p in probs:
        idx = np.searchsorted(cdf, p, side="left")
        idx = min(idx, vals.size - 1)
        thresholds.append(float(vals[idx]))
    return thresholds


def _project_to_hemisphere(
    ra_val: float,
    dec_val: float,
) -> tuple[str, float, float]:
    """Project a single (RA, Dec) point into north/south hemisphere panel coordinates."""
    ra_use = ra_val
    if dec_val >= 0.0:
        rr = (np.pi / 2 - dec_val) / (np.pi / 2)
        xx = rr * np.sin(ra_use)
        yy = rr * np.cos(ra_use)
        return "north", float(xx), float(yy)

    rr = (np.pi / 2 + dec_val) / (np.pi / 2)
    xx = -rr * np.sin(ra_use)
    yy = rr * np.cos(ra_use)
    return "south", float(xx), float(yy)

@lru_cache(maxsize=1)
def _constellation_stick_segments():
    filename = os.path.join(SKY_MAP_ROOT, "constellations_rey.txt")
    hip_ids = _read_constellation_hip_ids(filename)
    print(len(hip_ids))

    hip_lookup = _hip_lookup_table()

    north_lines = []
    south_lines = []

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

                if hip1 not in hip_lookup or hip2 not in hip_lookup:
                    continue  # star not resolved in catalog - skip this segment

                ra1, dec1 = hip_lookup[hip1]
                ra2, dec2 = hip_lookup[hip2]

                p1 = _project_to_hemisphere(np.deg2rad(ra1), np.deg2rad(dec1))
                p2 = _project_to_hemisphere(np.deg2rad(ra2), np.deg2rad(dec2))

                if p1[0] == p2[0]:
                    segment = np.array([[p1[1], p1[2]], [p2[1], p2[2]]])
                    if p1[0] == "north":
                        north_lines.append(segment)
                    else:
                        south_lines.append(segment)
                    continue

                # Straddles the celestial equator: find where the line between
                # the two stars crosses Dec=0, then clip each half at that
                # point projected onto its own hemisphere's edge (radius=1
                # in both panels, since Dec=0 is the boundary circle).
                if dec1 >= 0:
                    ra_n, dec_n, pt_n = ra1, dec1, p1
                    ra_s, dec_s, pt_s = ra2, dec2, p2
                else:
                    ra_n, dec_n, pt_n = ra2, dec2, p2
                    ra_s, dec_s, pt_s = ra1, dec1, p1

                t = dec_n / (dec_n - dec_s)  # fraction of the way from north star to south star
                delta_ra = ((ra_s - ra_n + 180.0) % 360.0) - 180.0  # shortest angular delta
                ra_cross_deg = (ra_n + t * delta_ra) % 360.0
                ang = np.deg2rad(ra_cross_deg)

                edge_n = (np.sin(ang), np.cos(ang))
                edge_s = (-np.sin(ang), np.cos(ang))

                north_lines.append(np.array([[pt_n[1], pt_n[2]], list(edge_n)]))
                south_lines.append(np.array([[edge_s[0], edge_s[1]], [pt_s[1], pt_s[2]]]))

    return north_lines, south_lines

@lru_cache(maxsize=1)
def _constellation_border_segments():
    border_file = os.path.join(SKY_MAP_ROOT, "lines_in_18.txt")

    # --- read raw points first ---
    borders, ra_hrs, dec_degs = [], [], []
    with open(border_file) as f:
        for line in f:
            if not line.strip():
                continue
            ra_hr, dec_deg, border = line.split()
            borders.append(border)
            ra_hrs.append(float(ra_hr))
            dec_degs.append(float(dec_deg))

    ra_hrs = np.asarray(ra_hrs)
    dec_degs = np.asarray(dec_degs)

    # --- precess B1875 (original Delporte epoch) -> J2000 to match Hipparcos ---
    coords_b1875 = SkyCoord(
        ra=ra_hrs * 15.0 * u.deg,
        dec=dec_degs * u.deg,
        frame=FK4(equinox="B1875"),
    )
    coords_j2000 = coords_b1875.transform_to("icrs")
    ra_deg_j2000 = coords_j2000.ra.deg
    dec_deg_j2000 = coords_j2000.dec.deg

    north_lines, south_lines = [], []
    current_border = None
    current_points = []

    def flush():
        if len(current_points) < 2:
            return
        north, south = [], []
        for ra_deg, dec_deg in current_points:
            hemi, x, y = _project_to_hemisphere(np.deg2rad(ra_deg), np.deg2rad(dec_deg))
            (north if hemi == "north" else south).append((x, y))
        if len(north) >= 2:
            north_lines.append(np.asarray(north))
        if len(south) >= 2:
            south_lines.append(np.asarray(south))

    # Only split into a new polyline on a genuine constellation-name change
    # (see previous message - the old dra/ddec jump heuristic over-fragmented
    # long straight edges and has been dropped here too).
    for border, ra_deg, dec_deg in zip(borders, ra_deg_j2000, dec_deg_j2000):
        if current_border is None or border != current_border:
            flush()
            current_points = []
            current_border = border
        current_points.append((ra_deg, dec_deg))
    flush()

    return north_lines, south_lines

@lru_cache(maxsize=8)
def _constellation_centers_icrs_deg(n_ra: int = 360, n_dec: int = 180) -> dict[str, tuple[float, float]]:
    """Estimate constellation label centers (RA, Dec deg) from an ICRS sampling grid."""
    if not _ASTROPY_AVAILABLE:
        return {}

    ra_deg = np.linspace(0.0, 360.0, n_ra, endpoint=False)
    dec_deg = np.linspace(-89.5, 89.5, n_dec)
    ra_mesh, dec_mesh = np.meshgrid(ra_deg, dec_deg)

    sky = SkyCoord(ra=ra_mesh.ravel() * u.deg, dec=dec_mesh.ravel() * u.deg, frame="icrs")
    const_names = np.asarray(sky.get_constellation(short_name=True))
    dec_flat = dec_mesh.ravel()
    ra_flat = ra_mesh.ravel()

    centers: dict[str, tuple[float, float]] = {}
    for short_name in np.unique(const_names):
        mask = const_names == short_name
        if not np.any(mask):
            continue

        # Circular mean for RA avoids a seam artifact around 0/360 deg.
        ra_rad = np.deg2rad(ra_flat[mask])
        mean_ra = np.mod(np.rad2deg(np.arctan2(np.mean(np.sin(ra_rad)), np.mean(np.cos(ra_rad)))), 360.0)
        mean_dec = float(np.mean(dec_flat[mask]))
        centers[str(short_name)] = (float(mean_ra), mean_dec)

    return centers


import cartopy.io.shapereader as shpreader
from shapely.geometry import LineString, MultiLineString


def _read_constellation_hip_ids(filename):

    hip_ids = set()

    with open(filename, "r") as f:
        for line in f:

            if line.startswith("#") or not line.strip():
                continue

            parts = line.split()

            n_segments = int(parts[1])

            ids = list(map(int, parts[2:]))

            for i in range(2 * n_segments):
                hip_ids.add(ids[i])

    return sorted(hip_ids)

@lru_cache(maxsize=1)
def _hip_lookup_table() -> dict[int, tuple[float, float]]:
    """Build a {HIP id: (ra_deg, dec_deg)} lookup from the Hipparcos catalog.

    Downloads the full Hipparcos main catalogue (Vizier I/239/hip_main) via
    astroquery the first time it's needed, caches it to disk under
    SKY_MAP_ROOT, and reuses that cache on subsequent calls/runs.
    """
    cache_path = os.path.join(SKY_MAP_ROOT, "hip_lookup_cache.pkl")
    if os.path.exists(cache_path):
        import pickle
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    try:
        from astroquery.vizier import Vizier
    except ImportError as exc:
        raise ImportError(
            "astroquery is required to resolve constellation stick-figure star "
            "positions (constellations=True). Install it with "
            "`pip install astroquery`, or call with constellations=False."
        ) from exc

    vizier = Vizier(columns=["HIP", "_RAJ2000", "_DEJ2000"])
    vizier.ROW_LIMIT = -1
    table = vizier.get_catalogs("I/239/hip_main")[0]

    ra_col = "_RAJ2000" if "_RAJ2000" in table.colnames else "RAJ2000"
    dec_col = "_DEJ2000" if "_DEJ2000" in table.colnames else "DEJ2000"

    hip_lookup: dict[int, tuple[float, float]] = {}
    for row in table:
        try:
            hip_lookup[int(row["HIP"])] = (float(row[ra_col]), float(row[dec_col]))
        except (ValueError, TypeError):
            continue  # masked/missing entries

    try:
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump(hip_lookup, f)
    except OSError:
        pass  # caching is best-effort

    return hip_lookup


def plot_galactic_supernovae_polar_hemispheres(
    ccsn,
    fname: str = "plots/galactic_supernovae_polar_hemispheres.png",
    posterior_ra_samples: np.ndarray | None = None,
    posterior_dec_samples: np.ndarray | None = None,
    true_ra_override: float | None = None,
    true_dec_override: float | None = None,
    show_constellation_borders: bool = False,
    constellations: bool = True,
    galaxy: bool = True,
    background: str = "black",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    show_detectors: bool = True,
    transparent: bool = False,
    format: str = "poster",
    n_background_supernovae: int = 20000,
    coastline: bool = False,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot CCSN sky distribution as tangent north/south pole-centered hemispheres.

    Args:
        ccsn: Supernovae-like object exposing ``ra``, ``dec`` and
            ``get_galactic_center_direction()``.
        fname: Output image path.
        posterior_ra_samples: Optional posterior RA samples (radians). If provided
            with ``posterior_dec_samples``, red sky-location contours are built from these.
        posterior_dec_samples: Optional posterior Dec samples (radians).
        true_ra_override: Optional true RA (radians) to override
            ``ccsn.get_galactic_center_direction()`` for center marker logic.
        true_dec_override: Optional true Dec (radians).
        show_constellation_borders: If True, overlay IAU constellation boundaries.
        show_all_constellation_labels: If True and constellation borders are enabled,
            annotate all 88 IAU constellations with their 3-letter codes.
        show: If True, call ``plt.show()``.
        dpi: Image save DPI.
        background: Background color theme ("white" or "black").
        font_family: Font family to use.
        font_name: Specific font name.
        red_blob_mode: Red contour center mode. One of
            ``"middle_star"``, ``"density_peak"``, ``"true_center"``.
        show_detectors: If True, add detector markers (LIGO Hanford, LIGO Livingston, Virgo)
            and highlight the first supernova as the true location.
        format: Layout format - "poster" for the A1 landscape poster (two hemispheres
            side by side), or "thesis" for a 14.5 x 19 cm portrait figure with the
            two hemispheres stacked vertically (North on top, South on bottom).
        n_background_supernovae: Number of closest supernovae to use for background distribution.
            If not enough supernovae are available, uses all. Default 50000.
    """
    # Configure sizes based on mode
    # Figsize in mm (converted to inches for matplotlib)
    mm_to_inch = 1 / 25.4
    mode_config = {
        "poster": {
            "figsize_mm": (841, 594),  # A1 landscape
            "fontsize_title": 28,
            "fontsize_main": 22,
            "fontsize_label": 28,
            "fontsize_tick": 12,
            "fontsize_small": 18,
            "fontsize_tiny": 12,
            "fontsize_constellation": 18,
            "fontsize_object": 16,
        },
        "thesis": {
            "figsize_mm": (145, 190),  # 14.5 x 19 cm portrait
            "fontsize_title": 16,
            "fontsize_main": 11,
            "fontsize_label": 11,
            "fontsize_tick": 11,
            "fontsize_small": 7,
            "fontsize_tiny": 7,
            "fontsize_constellation": 7,
            "fontsize_object": 7,
        }
    }
    
    if format not in mode_config:
        raise ValueError(f"format must be 'poster' or 'thesis', got {format}")
    
    config = mode_config[format]
    figsize_mm = config["figsize_mm"]
    figsize_inches = (figsize_mm[0] * mm_to_inch, figsize_mm[1] * mm_to_inch)
    if figsize is not None:
        figsize_inches = (figsize[0] * mm_to_inch, figsize[1] * mm_to_inch)
    figsize = figsize_inches
    fontsize_title = config["fontsize_title"]
    fontsize_main = config["fontsize_main"]
    fontsize_label = config["fontsize_label"]
    fontsize_tick = config["fontsize_tick"]
    fontsize_small = config["fontsize_small"]
    fontsize_tiny = config["fontsize_tiny"]
    fontsize_constellation = config["fontsize_constellation"]
    fontsize_object = config["fontsize_object"]
    
    set_plot_style(background, font_family, font_name)
    
    # If transparent, override rcParams to allow transparent background
    if transparent:
        plt.rcParams['figure.facecolor'] = 'none'
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['savefig.facecolor'] = 'none'
    
    # Ensure SVG text is vectorized (not rasterized)
    plt.rcParams['svg.fonttype'] = 'path'
    
    astropy_rotation_offset_deg = 0.0
    
    # Extract RA and Dec, optionally sampling only the closest supernovae
    all_ra = np.mod(np.asarray(ccsn.ra), 2 * np.pi)
    all_dec = np.asarray(ccsn.dec)
    
    # If distance data available, sample the N closest supernovae
    if hasattr(ccsn, 'distance') and ccsn.distance is not None:
        distances = np.asarray(ccsn.distance)
        sorted_indices = np.argsort(distances)
        n_sample = min(n_background_supernovae, len(sorted_indices))
        sample_indices = sorted_indices[:n_sample]
        ra_supernovae = all_ra[sample_indices]
        dec_supernovae = all_dec[sample_indices]
    else:
        # Fall back to all supernovae if no distance data
        ra_supernovae = all_ra
        dec_supernovae = all_dec

    use_posterior_samples = (
        posterior_ra_samples is not None
        and posterior_dec_samples is not None
        and np.asarray(posterior_ra_samples).size > 0
        and np.asarray(posterior_dec_samples).size > 0
    )
    ra_posterior = np.mod(np.asarray(posterior_ra_samples), 2 * np.pi) if use_posterior_samples else None
    dec_posterior = np.asarray(posterior_dec_samples) if use_posterior_samples else None

    # Build the galactic streak directly from Supernovae RA/Dec.
    ra_rot_supernovae = ra_supernovae

    fig_facecolor = None if transparent else background
    if background == "black":
        fig_facecolor = "#0f0d33"
    text_color = "black" if background == "white" else "white"
    fig = plt.figure(figsize=figsize, facecolor=fig_facecolor)
    # Keep a small canvas margin so boundary lines and circles are not clipped at image edges.
    # ax_l always holds the Northern-Sky panel, ax_r always holds the Southern-Sky panel.
    if format == "thesis":
        # Portrait layout: North stacked on top of South, with a small gap between
        # them for the "Credible Intervals" legend and room at the bottom for the
        # main legend.
        ax_l = fig.add_axes([0.0, 0.525, 1.0, 0.425], facecolor=fig_facecolor)
        ax_r = fig.add_axes([0.0, 0.05, 1.0, 0.425], facecolor=fig_facecolor)
    else:
        ax_l = fig.add_axes([0.015, 0.07, 0.48, 0.94], facecolor=fig_facecolor)
        ax_r = fig.add_axes([0.505, 0.07, 0.48, 0.94], facecolor=fig_facecolor)

    north_mask = dec_supernovae >= 0
    ra_n = ra_rot_supernovae[north_mask]
    dec_n = dec_supernovae[north_mask]
    r_n = (np.pi / 2 - dec_n) / (np.pi / 2)
    x_n = r_n * np.sin(ra_n)  # RA increases counter-clockwise for north pole
    y_n = r_n * np.cos(ra_n)

    south_mask = dec_supernovae <= 0
    ra_s = ra_rot_supernovae[south_mask]
    dec_s = dec_supernovae[south_mask]
    r_s = (np.pi / 2 + dec_s) / (np.pi / 2)
    x_s = -r_s * np.sin(ra_s)  # RA increases clockwise for south pole
    y_s = r_s * np.cos(ra_s)

    theta = np.linspace(0, 2 * np.pi, 600)
    lat_step_deg = 10
    lat_radii = [(90.0 - lat_deg) / 90.0 for lat_deg in range(lat_step_deg, 90, lat_step_deg)]

    bins = 320
    hist_range = [[-1.0, 1.0], [-1.0, 1.0]]
    h_n, xedges, yedges = np.histogram2d(x_n, y_n, bins=bins, range=hist_range)
    h_s, _, _ = np.histogram2d(x_s, y_s, bins=bins, range=hist_range)

    k_radius = 3
    k_sigma = 1.2
    k_axis = np.arange(-k_radius, k_radius + 1)
    kernel = np.exp(-(k_axis**2) / (2.0 * k_sigma**2))
    kernel /= kernel.sum()

    h_n_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=h_n)
    h_n_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=h_n_smooth)
    h_s_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=h_s)
    h_s_smooth = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=h_s_smooth)

    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    xxc, yyc = np.meshgrid(xcenters, ycenters)
    inside_circle = (xxc**2 + yyc**2) <= 1.0

    h_n_plot = np.ma.array(h_n_smooth.T, mask=~inside_circle)
    h_s_plot = np.ma.array(h_s_smooth.T, mask=~inside_circle)

    if galaxy:
        blue_probs = [0.995, 0.80, 0.50, 0.25]
        combined_vals = np.concatenate([
            h_n_smooth.T[inside_circle],
            h_s_smooth.T[inside_circle],
        ])
        combined_vals = combined_vals[combined_vals > 0]
        if combined_vals.size == 0:
            thr_shared = [1.0 for _ in blue_probs]
        else:
            vals = np.sort(combined_vals)[::-1]
            cdf = np.cumsum(vals) / np.sum(vals)
            thr_shared = []
            for p in blue_probs:
                idx = np.searchsorted(cdf, p, side="left")
                idx = min(idx, vals.size - 1)
                thr_shared.append(float(vals[idx]))

        levels_shared = np.sort(np.array(thr_shared, dtype=float))
        top_shared = max(levels_shared[-1] * 1.001, np.max(combined_vals) * 1.001)
        fill_levels_shared = np.concatenate([levels_shared, [top_shared]])

        blue_bases = ["#486ac8", "#488af4", "#60a5fa", "#bfdbfe"]
        # Contourf colors are mapped outer->inner because levels are ascending.
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

        ax_l.contourf(xcenters, ycenters, h_n_plot, levels=smooth_levels, colors=smooth_colors, antialiased=True)
        ax_r.contourf(xcenters, ycenters, h_s_plot, levels=smooth_levels, colors=smooth_colors, antialiased=True)

    for r_lat in lat_radii:
        ax_l.plot(r_lat * np.cos(theta), r_lat * np.sin(theta), color=text_color, alpha=0.2, lw=0.75)
        ax_r.plot(r_lat * np.cos(theta), r_lat * np.sin(theta), color=text_color, alpha=0.2, lw=0.75)

    ax_l.plot(np.cos(theta), np.sin(theta), color=text_color, lw=1.5)
    ax_r.plot(np.cos(theta), np.sin(theta), color=text_color, lw=1.5)

    meridian_angles_deg = [0, 30, 60, 90, 120, 150]  # replace with e.g. np.arange(0, 360, 30) for more spokes

    for ang_deg in meridian_angles_deg:
        ang = np.deg2rad(ang_deg)

        # North panel: full diameter from angle to angle+180, through the center.
        x1_n, y1_n = np.sin(ang), np.cos(ang)
        x2_n, y2_n = -np.sin(ang), -np.cos(ang)
        ax_l.plot([x1_n, x2_n], [y1_n, y2_n], color=text_color, alpha=0.2, lw=0.75, zorder=10)

        # South panel (mirrored x, per the rest of the file's convention).
        x1_s, y1_s = -np.sin(ang), np.cos(ang)
        x2_s, y2_s = np.sin(ang), -np.cos(ang)
        ax_r.plot([x1_s, x2_s], [y1_s, y2_s], color=text_color, alpha=0.2, lw=0.75, zorder=10)

    # Add "Northern Sky" label directly above 0h RA (top of hemisphere)
    if background == "black" and format == "poster":
        ax_l.text(
            0.0,
            1.12,
            "Northern Sky",
            color=text_color,
            fontsize=fontsize_title,
            ha="center",
            va="bottom",
            fontweight="bold",
            alpha=0.9,
        )
    ax_l.text(
        0.0,
        0.0,
        "North\nPole",
        color=text_color,
        fontsize=fontsize_tick,
        ha="center",
        va="center",
        multialignment="center",
        alpha=0.95,
        zorder=10
    )

    # Add "Southern Sky" label directly above 0h RA (top of hemisphere)
    if background == "black" and format == "poster":
        ax_r.text(
            0.0,
            1.12,
            "Southern Sky",
            color=text_color,
            fontsize=fontsize_title,
            ha="center",
            va="bottom",
            fontweight="bold",
            alpha=0.9,
        )
    ax_r.text(
        0.0,
        0.0,
        "South\nPole",
        color=text_color,
        fontsize=fontsize_tick,
        ha="center",
        va="center",
        multialignment="center",
        alpha=0.95,
        zorder=10
    )

    if galaxy:
        # Add curved "MILKY WAY" text below south pole, following the arc
        milky_way_text = "MILKY WAY"
        n_chars = len(milky_way_text)
        # Spread text across lower arc (from -120° to -60° in compass bearing, or 210° to 300° in standard angle)
        angle_start = 240  # degrees in standard angle (bottom-left)
        angle_end = 300    # degrees in standard angle (bottom-right)
        angles = np.linspace(np.deg2rad(angle_start), np.deg2rad(angle_end), n_chars)
        radius = 0.72  # position along the arc (raised from south pole)
    
        for i, (char, angle) in enumerate(zip(milky_way_text, angles)):
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            # Rotate text to follow the curve (tangent to circle)
            rotation = np.rad2deg(angle) + 90  # +90 to align text tangentially
            ax_r.text(
                x,
                y,
                char,
                color=text_color,
                fontsize=fontsize_title if format == "poster" else fontsize_main,
                ha="center",
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                alpha=1.0,
                fontweight="bold",
                zorder=5,
                clip_on=False,
            )

    # if galaxy:
    #     from astropy.coordinates import SkyCoord
    #     import astropy.units as u

    #     # Sample the Galactic equator
    #     l = np.linspace(0, 360, 720) * u.deg
    #     b = np.zeros_like(l.value) * u.deg

    #     gal_plane = SkyCoord(l=l, b=b, frame="galactic").icrs

    #     ra = gal_plane.ra.rad
    #     dec = gal_plane.dec.rad

    #     south_curve = []

    #     for ra_i, dec_i in zip(ra, dec):
    #         panel, x, y = _project_to_hemisphere(ra_i, dec_i)

    #         if panel == "south":
    #             south_curve.append((x, y))

    #     south_curve = np.asarray(south_curve)

    #     ax_r.plot(
    #         south_curve[:,0],
    #         south_curve[:,1],
    #         color="red",
    #         lw=1,
    #     )

    #     d = np.sqrt(np.sum(np.diff(south_curve, axis=0)**2, axis=1))
    #     s = np.concatenate([[0], np.cumsum(d)])

    #     text = "MILKY WAY"

    #     centre = 0.55 * s[-1]        # move this left/right
    #     spacing = 0.035              # adjust letter spacing

    #     letter_pos = centre + (
    #         np.arange(len(text))
    #         - (len(text)-1)/2
    #     ) * spacing
    
    #     for char, target_s in zip(text, letter_pos):
    #         idx = np.searchsorted(s, target_s)

    #         idx = np.clip(idx, 1, len(s)-1)

    #         t = (target_s - s[idx-1]) / (s[idx] - s[idx-1])

    #         x = (1-t)*south_curve[idx-1,0] + t*south_curve[idx,0]
    #         y = (1-t)*south_curve[idx-1,1] + t*south_curve[idx,1]

    #         dx = south_curve[idx,0] - south_curve[idx-1,0]
    #         dy = south_curve[idx,1] - south_curve[idx-1,1]

    #         rotation = np.degrees(np.arctan2(dy, dx))

    #         ax_r.text(
    #             x,
    #             y,
    #             char,
    #             fontsize=fontsize_title,
    #             color=text_color,
    #             rotation=rotation,
    #             rotation_mode="anchor",
    #             ha="center",
    #             va="center",
    #             fontweight="bold",
    #             clip_on=False,
    #         )



    # RA degree labels, curved tangent to each hemisphere panel, centered on the pole.
    ra_label_deg = np.arange(0, 360, 30) # every 45 degrees but not 180 (which is the seam)
    if format == "poster":
        ra_label_deg = np.delete(ra_label_deg, np.where(ra_label_deg == 90))  # remove 90 degrees
    ra_label_radius = 1.02
    for ra_deg in ra_label_deg:
        label = f"{int(ra_deg)}\u00b0"
        _draw_curved_ra_label(ax_l, "north", ra_deg, ra_label_radius + 0.02, label, text_color, fontsize_small, 0.75)
        _draw_curved_ra_label(ax_r, "south", ra_deg, ra_label_radius + 0.02, label, text_color, fontsize_small, 0.75)

    dec_abs_ticks = [80, 60, 40, 20]
    # Place Dec ticks on the 0h/24h RA meridian.
    ux = 0.0
    uy = 1.0
    for dec_abs in dec_abs_ticks:
        r_tick = (90.0 - float(dec_abs)) / 90.0
        x0 = r_tick * ux
        y0 = r_tick * uy

        # North: positive Dec labels.
        # ax_l.plot(
        #     [x0 - 0.012, x0 + 0.012],
        #     [y0, y0],
        #     color=text_color,
        #     alpha=0.2,
        #     lw=0.75,
        #     zorder=9,
        # )
        ax_l.text(
            x0 + 0.020,
            y0,
            f"+{dec_abs}°",
            color=text_color,
            fontsize=fontsize_small,
            ha="left",
            va="center",
            alpha=0.75,
            zorder=9,
        )

        # South: negative Dec labels.
        # ax_r.plot(
        #     [x0 - 0.012, x0 + 0.012],
        #     [y0, y0],
        #     color=text_color,
        #     alpha=0.2,
        #     lw=0.75,
        #     zorder=6,
        # )
        ax_r.text(
            x0 + 0.020,
            y0,
            f"-{dec_abs}°",
            color=text_color,
            fontsize=fontsize_small,
            ha="left",
            va="center",
            alpha=0.75,
            zorder=6,
        )

    if show_constellation_borders:
        if _ASTROPY_AVAILABLE:
            north_lines, south_lines = _constellation_border_segments()
            ax_l.add_collection(
                LineCollection(
                    north_lines,
                    colors="#b1cbed" if background == "black" else "#1e293b",
                    linewidths=0.5,
                    alpha=0.5,
                    zorder=4,
                    linestyle=(0, (5, 5)),
                    joinstyle="round",
                    capstyle="round"
                )
            )
            ax_r.add_collection(
                LineCollection(
                    south_lines,
                    colors="#b1cbed" if background == "black" else "#1e293b",
                    linewidths=0.5,
                    alpha=0.5,
                    zorder=4,
                    linestyle=(0, (5, 5)),
                    joinstyle="round",
                    capstyle="round"
                )
            )
        else:
            print("Constellation borders requested, but astropy is not installed in this environment.")

    if galaxy and constellations:
        north_lines, south_lines = _constellation_stick_segments()
        ax_l.add_collection(
            LineCollection(
                north_lines,
                colors="#6ca3eb",
                linewidths=0.75,
                alpha=1.0,
                zorder=4,
                joinstyle="round",
                capstyle="round"
            )
        )
        ax_r.add_collection(
            LineCollection(
                south_lines,
                colors="#6ca3eb",
                linewidths=0.75,
                alpha=1.0,
                zorder=4,
                joinstyle="round",
                capstyle="round"
            )
        )

    if format == "thesis":
        # Panels are stacked, not side by side, so there is no shared seam to keep flush.
        ax_l.set_xlim(-1.05, 1.05)
        ax_r.set_xlim(-1.05, 1.05)
    else:
        # Make circles touch at center seam while keeping extra margin on outer edges.
        ax_l.set_xlim(-1.03, 1.02)
        ax_r.set_xlim(-1.02, 1.03)

    for ax in (ax_l, ax_r):
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(-1.03, 1.03)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if galaxy:
        # Keep Galactic Center fixed to the physical galactic center direction.
        gc_ra, gc_dec = ccsn.get_galactic_center_direction()
        
        # Print galactic center coordinates
        print(f"\n{'='*60}")
        print(f"Galactic Center Direction:")
        print(f"  RA:  {gc_ra:.6f} rad = {np.degrees(gc_ra):.2f}°")
        print(f"  Dec: {gc_dec:.6f} rad = {np.degrees(gc_dec):.2f}°")
        print(f"{'='*60}\n")

    # Handle example mode: use first supernova as true location and prepare detector markers.
    detector_markers = []
    if show_detectors:
        detector_markers = [
            ("LIGO Hanford", np.deg2rad(240.6), np.deg2rad(46.5), text_color),
            ("LIGO Livingston", np.deg2rad(269.2), np.deg2rad(30.5), text_color),
            ("Virgo", np.deg2rad(10.5), np.deg2rad(43.6), text_color),
        ]

    if galaxy:
        # Use the true galactic center for black hole visualization.
        true_gc_panel, true_gc_x, true_gc_y = _project_to_hemisphere(gc_ra, gc_dec)

    # Optional true event location marker (independent of Galactic Center).
    true_loc_panel = None
    true_loc_x = np.nan
    true_loc_y = np.nan
    if true_ra_override is not None and true_dec_override is not None:
        true_loc_panel, true_loc_x, true_loc_y = _project_to_hemisphere(
            float(true_ra_override),
            float(true_dec_override),
        )

    if use_posterior_samples:
        # Choose red sky-location density: posterior contour map if provided, otherwise legacy blob.
        red_bases = [GENERATED_SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR, GENERATED_SIGNAL_COLOUR]
        red_fill_colors = [
            to_rgba(red_bases[0], alpha=0.40),
            to_rgba(red_bases[1], alpha=0.62),
            to_rgba(red_bases[2], alpha=0.88),
        ]
        ra_rot_posterior = ra_posterior
        post_north = dec_posterior >= 0
        post_south = dec_posterior <= 0

        ra_pn = ra_rot_posterior[post_north]
        dec_pn = dec_posterior[post_north]
        
        # Detect and handle RA wrapping at 0/2π boundary
        # If samples span > π radians, shift to avoid discontinuity
        if ra_pn.size > 0:
            ra_range = np.max(ra_pn) - np.min(ra_pn)
            if ra_range > np.pi:
                # Likely wrapping issue - shift samples > π to negative range
                ra_pn = np.where(ra_pn > np.pi, ra_pn - 2*np.pi, ra_pn)
        
        r_pn = (np.pi / 2 - dec_pn) / (np.pi / 2)
        x_pn = r_pn * np.sin(ra_pn)
        y_pn = r_pn * np.cos(ra_pn)

        ra_ps = ra_rot_posterior[post_south]
        dec_ps = dec_posterior[post_south]
        
        # Same wrapping fix for south hemisphere
        if ra_ps.size > 0:
            ra_range = np.max(ra_ps) - np.min(ra_ps)
            if ra_range > np.pi:
                ra_ps = np.where(ra_ps > np.pi, ra_ps - 2*np.pi, ra_ps)
        
        r_ps = (np.pi / 2 + dec_ps) / (np.pi / 2)
        x_ps = -r_ps * np.sin(ra_ps)
        y_ps = r_ps * np.cos(ra_ps)

        # Use coarser bins and stronger smoothing for readable posterior contours.
        post_bins = 180
        h_pn, pxedges, pyedges = np.histogram2d(x_pn, y_pn, bins=post_bins, range=hist_range)
        h_ps, _, _ = np.histogram2d(x_ps, y_ps, bins=post_bins, range=hist_range)

        post_k_radius = 5
        post_k_sigma = 2.4
        post_k_axis = np.arange(-post_k_radius, post_k_radius + 1)
        post_kernel = np.exp(-(post_k_axis**2) / (2.0 * post_k_sigma**2))
        post_kernel /= post_kernel.sum()

        h_pn_smooth = np.apply_along_axis(lambda m: np.convolve(m, post_kernel, mode="same"), axis=0, arr=h_pn)
        h_pn_smooth = np.apply_along_axis(lambda m: np.convolve(m, post_kernel, mode="same"), axis=1, arr=h_pn_smooth)
        h_ps_smooth = np.apply_along_axis(lambda m: np.convolve(m, post_kernel, mode="same"), axis=0, arr=h_ps)
        h_ps_smooth = np.apply_along_axis(lambda m: np.convolve(m, post_kernel, mode="same"), axis=1, arr=h_ps_smooth)

        pxcenters = 0.5 * (pxedges[:-1] + pxedges[1:])
        pycenters = 0.5 * (pyedges[:-1] + pyedges[1:])
        pxxc, pyyc = np.meshgrid(pxcenters, pycenters)
        post_inside_circle = (pxxc**2 + pyyc**2) <= 1.0

        h_pn_plot = np.ma.array(h_pn_smooth.T, mask=~post_inside_circle)
        h_ps_plot = np.ma.array(h_ps_smooth.T, mask=~post_inside_circle)
        post_vals = np.concatenate([
            h_pn_smooth.T[post_inside_circle],
            h_ps_smooth.T[post_inside_circle],
        ])
        post_vals = post_vals[post_vals > 0]

        if post_vals.size > 0:
            vals = np.sort(post_vals)[::-1]
            cdf = np.cumsum(vals) / np.sum(vals)
            post_thr = []
            # Use the same probability bands and palette as the legacy red blob,
            # but sourced from posterior samples.
            posterior_probs = [0.68, 0.90, 0.95]
            for p in posterior_probs:
                idx = np.searchsorted(cdf, p, side="left")
                idx = min(idx, vals.size - 1)
                post_thr.append(float(vals[idx]))
            post_levels = np.sort(np.array(post_thr, dtype=float))
            post_top = max(post_levels[-1] * 1.001, np.max(post_vals) * 1.001)
            post_fill_levels = np.concatenate([post_levels, [post_top]])

            ax_l.contourf(
                pxcenters,
                pycenters,
                h_pn_plot,
                levels=post_fill_levels,
                colors=red_fill_colors,
                antialiased=True,
                zorder=20
            )
            ax_r.contourf(
                pxcenters,
                pycenters,
                h_ps_plot,
                levels=post_fill_levels,
                colors=red_fill_colors,
                antialiased=True,
                zorder=20
            )

            posterior_legend_handles = [
                Patch(
                    facecolor=red_fill_colors[0],
                    edgecolor="none",
                    label="95%",
                ),
                Patch(
                    facecolor=red_fill_colors[1],
                    edgecolor="none",
                    label="90%",
                ),
                Patch(
                    facecolor=red_fill_colors[2],
                    edgecolor="none",
                    label="68%",
                ),
            ]
            if format == "thesis":
                # Use a blank-handle entry for "Credible Intervals:" instead of the legend's
                # `title=` row - a separate title row adds height above the axes and was
                # getting clipped against the figure edge. This keeps everything on one line.
                inline_title_handle = Patch(facecolor="none", edgecolor="none", label="Credible Intervals:")
                ci_legend = ax_l.legend(
                    handles=[inline_title_handle] + posterior_legend_handles,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),   # 2% above the axes
                    ncol=4,
                    frameon=False,
                    fontsize=fontsize_main,
                    labelcolor=text_color,
                    handlelength=1.2,
                    handletextpad=0.5,
                    columnspacing=0.8,
                    borderaxespad=0.2,            # small padding
                )
                # Make the leading label read like a title even though it's inline.
                ci_legend.get_texts()[0].set_fontsize(fontsize_main)
            else:
                fig.legend(
                    handles=posterior_legend_handles,
                    loc="center",
                    bbox_to_anchor=(0.5, 0.18),
                    ncol=1,
                    frameon=False,
                    fontsize=fontsize_main,
                    labelcolor=text_color,
                    handlelength=1.2,
                    handletextpad=0.5,
                    columnspacing=0.8,
                    borderaxespad=0.0,
                    title="Credible Intervals",
                    title_fontsize=fontsize_main,
                )

        # Marker at posterior peak.
        n_plot = np.ma.array(h_pn_smooth.T, mask=~post_inside_circle)
        s_plot = np.ma.array(h_ps_smooth.T, mask=~post_inside_circle)
        n_max = float(np.max(n_plot.filled(-np.inf)))
        s_max = float(np.max(s_plot.filled(-np.inf)))
        if n_max >= s_max and np.isfinite(n_max):
            iy, ix = np.unravel_index(np.argmax(n_plot.filled(-np.inf)), n_plot.shape)
            gc_panel = "north"
            gc_x = float(xcenters[ix])
            gc_y = float(ycenters[iy])
        elif np.isfinite(s_max):
            iy, ix = np.unravel_index(np.argmax(s_plot.filled(-np.inf)), s_plot.shape)
            gc_panel = "south"
            gc_x = float(xcenters[ix])
            gc_y = float(ycenters[iy])
        else:
            gc_panel, gc_x, gc_y = true_gc_panel, true_gc_x, true_gc_y

    if constellations:
        # Resolve Betelgeuse via Astropy name resolution (no hardcoded coordinate fallback).
        betelgeuse_ra_deg, betelgeuse_dec_deg, betel_source = _get_betelgeuse_icrs_deg(
            rotation_offset_deg=astropy_rotation_offset_deg
        )
        if np.isfinite(betelgeuse_ra_deg) and np.isfinite(betelgeuse_dec_deg):
            betel_panel, betel_x, betel_y = _project_to_hemisphere(
                np.deg2rad(betelgeuse_ra_deg),
                np.deg2rad(betelgeuse_dec_deg),
            )
        else:
            betel_panel, betel_x, betel_y = None, np.nan, np.nan

    if galaxy:
        # Black hole visualization at the true galactic center.
        bh_ax = ax_l if true_gc_panel == "north" else ax_r

        # accretion disk (outer ring).
        bh_disk_outer = Circle(
            (true_gc_x, true_gc_y), 0.015, color="orange", alpha=0.8, zorder=8
        )
        bh_ax.add_patch(bh_disk_outer)

        # Black hole interior (event horizon) - blends into the background so it reads as a void.
        bh_interior = Circle(
            (true_gc_x, true_gc_y), 0.010, color="black", alpha=0.95, zorder=9
        )
        bh_ax.add_patch(bh_interior)

    # Plot true event sky location as an X marker when provided.
    if true_loc_panel is not None:
        true_ax = ax_l if true_loc_panel == "north" else ax_r
        true_ax.scatter(
            [true_loc_x],
            [true_loc_y],
            s=100,
            marker="x",
            c=SIGNAL_COLOUR,
            linewidths=1.8,
            zorder=20,
        )

    if constellations and betel_panel is not None:
            betel_ax = ax_l if betel_panel == "north" else ax_r
            betel_ax.scatter(
                [betel_x],
                [betel_y],
                s=52,
                c="#fbbf24",
                edgecolors="#78350f",
                linewidths=0.9,
                zorder=8,
            )
            betel_ax.text(
                betel_x - 0.035,
                betel_y + 0.02,
                "Betelgeuse",
                color="#fde68a" if background == "black" else "#78350f",
                fontsize=fontsize_constellation,
                ha="right",
                va="center",
                zorder=8,
            )

    if constellations:
        # Orion stick figure using Astropy-resolved named stars only.
        orion_star_names = [
            "Betelgeuse",
            "Bellatrix",
            "Meissa",
            "Mintaka",
            "Alnilam",
            "Alnitak",
            "Saiph",
            "Rigel",
        ]
        orion_edges = [
            ("Betelgeuse", "Bellatrix"),
            ("Betelgeuse", "Meissa"),
            ("Bellatrix", "Meissa"),
            ("Bellatrix", "Mintaka"),
            ("Betelgeuse", "Alnitak"),
            ("Mintaka", "Alnilam"),
            ("Alnilam", "Alnitak"),
            ("Alnitak", "Saiph"),
            ("Mintaka", "Rigel"),
            ("Saiph", "Rigel"),
        ]

        orion_proj: dict[str, tuple[str, float, float]] = {}
        for star_name in orion_star_names:
            resolved = _resolve_named_star_icrs_deg(star_name, rotation_offset_deg=astropy_rotation_offset_deg)
            if resolved is None:
                continue
            star_ra_deg, star_dec_deg = resolved
            orion_proj[star_name] = _project_to_hemisphere(
                np.deg2rad(star_ra_deg),
                np.deg2rad(star_dec_deg),
            )

        for star_name, (panel, sx, sy) in orion_proj.items():
            orion_ax = ax_l if panel == "north" else ax_r
            orion_ax.scatter(
                [sx],
                [sy],
                s=9,
                c="#f8fafc",
                edgecolors="none",
                alpha=0.95,
                zorder=9,
            )

        # Taurus stick figure (head + horns) using Astropy-resolved named stars.
        taurus_star_names = [
            "Aldebaran",
            "Elnath",
            "Zeta Tauri",
            "Gamma Tauri",
            "Delta Tauri",
            "Epsilon Tauri",
        ]

        taurus_proj: dict[str, tuple[str, float, float]] = {}
        for star_name in taurus_star_names:
            resolved = _resolve_named_star_icrs_deg(star_name, rotation_offset_deg=astropy_rotation_offset_deg)
            if resolved is None:
                continue
            star_ra_deg, star_dec_deg = resolved
            taurus_proj[star_name] = _project_to_hemisphere(
                np.deg2rad(star_ra_deg),
                np.deg2rad(star_dec_deg),
            )

        for star_name, (panel, sx, sy) in taurus_proj.items():
            taur_ax = ax_l if panel == "north" else ax_r
            taur_ax.scatter(
                [sx],
                [sy],
                s=16,
                c="#fecaca",
                edgecolors="none",
                alpha=0.95,
                zorder=9,
            )

        if "Aldebaran" in taurus_proj:
            panel, tx, ty = taurus_proj["Aldebaran"]
            taur_lbl_ax = ax_l if panel == "north" else ax_r
            # taur_lbl_ax.text(
            #     tx + 0.03,
            #     ty + 0.016,
            #     "Taurus",
            #     color="#fecaca",
            #     fontsize=fontsize_tiny,
            #     ha="left",
            #     va="center",
            #     zorder=10,
            # )

        # Southern Cross (Crux), pointer stars, Achernar, and Pleiades/Matariki.
        scx_star_names = ["Acrux", "Mimosa", "Gacrux", "Imai", "Epsilon Crucis"]
        scx_edges = [
            ("Gacrux", "Acrux"),
            ("Mimosa", "Imai"),
            ("Acrux", "Mimosa"),
            ("Acrux", "Imai"),
            ("Gacrux", "Mimosa"),
        ]
        pointer_names = ["Alpha Centauri", "Beta Centauri"]
        extra_names = ["Achernar", "Pleiades", "Antares"]

        south_proj: dict[str, tuple[str, float, float]] = {}
        for star_name in scx_star_names + pointer_names + extra_names:
            resolved = _resolve_named_star_icrs_deg(star_name, rotation_offset_deg=astropy_rotation_offset_deg)
            if resolved is None:
                continue
            star_ra_deg, star_dec_deg = resolved
            south_proj[star_name] = _project_to_hemisphere(
                np.deg2rad(star_ra_deg),
                np.deg2rad(star_dec_deg),
            )

        marker_styles = {
            "Acrux": ("#c4b5fd", 20),
            "Mimosa": ("#c4b5fd", 18),
            "Gacrux": ("#c4b5fd", 18),
            "Imai": ("#c4b5fd", 16),
            "Epsilon Crucis": ("#c4b5fd", 14),
            "Alpha Centauri": ("#fde68a", 20),
            "Beta Centauri": ("#fde68a", 20),
            "Achernar": ("#a5f3fc", 24),
            "Pleiades": ("#c4b5fd", 24),
            "Antares": ("#fca5a5", 24),
        }
        for star_name, (panel, sx, sy) in south_proj.items():
            color, size = marker_styles.get(star_name, ("#f8fafc", 14))
            mark_ax = ax_l if panel == "north" else ax_r
            mark_ax.scatter(
                [sx],
                [sy],
                s=size,
                c=color,
                edgecolors="none",
                alpha=0.96,
                zorder=9,
            )

        display_name_overrides = {
            "Pleiades": "Matariki",
            "Acrux": "The Pointers",
            }
        for label_name, label_color in (("Achernar", "#a5f3fc"), ("Pleiades", "#c4b5fd"), ("Acrux", "#c4b5fd"), ("Antares", "#fca5a5")):
            if label_name not in south_proj:
                continue
            panel, lx, ly = south_proj[label_name]
            lbl_ax = ax_l if panel == "north" else ax_r
            y_offset = 0.018
            if label_name == "Achernar":
                y_offset = 0.030
            display_name = display_name_overrides.get(label_name, label_name)
            if label_name == "Acrux":
                y_offset = 0.070
            lbl_ax.text(
                lx,
                ly + y_offset,
                display_name,
                color=label_color if background == "black" else "#1e293b",
                fontsize=fontsize_constellation,
                ha="left",
                va="center",
                zorder=10,
            )

        # Add Southern Cross label
        if "Gacrux" in south_proj:
            panel, gx, gy = south_proj["Gacrux"]
            scx_label_ax = ax_l if panel == "north" else ax_r
            scx_label_ax.text(
                gx - 0.03,
                gy - 0.03,
                "Southern Cross",
                color="#c4b5fd" if background == "black" else "#1e293b",
                fontsize=fontsize_constellation,
                ha="right",
                va="top",
                zorder=10,
            )

        # Additional visible-night-sky stars from Astropy name resolution, plotted without labels.
        # Brightest star from each of the 88 IAU constellations.
        extra_visible_star_names = [
            # Andromeda (3 main stars)
            "Alpheratz",        "Mirach",           "Almach",
            # Antlia (2 main stars)
            "Alpha Antliae",    "Epsilon Antliae",
            # Apus (2 main stars)
            "Alpha Apodis",     "Beta Apodis",
            # Aquarius (3 main stars)
            "Sadalmelik",       "Sadachbia",        "Sadalsuud",
            # Aquila (4 main stars)
            "Altair",           "Alshain",          "Tarazed",           "Beta Aquilae",
            # Ara (3 main stars)
            "Alpha Arae",       "Beta Arae",        "Gamma Arae",
            # Aries (3 main stars)
            "Hamal",            "Sheratan",         "Mesarthim",
            # Auriga (4 main stars)
            "Capella",          "Theta Aurigae",    "Iota Aurigae",      "Beta Aurigae",
            # Boötes (5 main stars)
            "Arcturus",         "Muphrid",          "Izar",              "Pulcherrima",      "Seginus",
            # Caelum (2 main stars)
            "Alpha Caeli",      "Gamma Caeli",
            # Camelopardalis (2 main stars)
            "Beta Camelopardalis",  "Alpha Camelopardalis",
            # Cancer (3 main stars)
            "Altarf",           "Acubens",          "Tzelakot",
            # Canes Venatici (3 main stars)
            "Cor Caroli",       "Chara",            "Beta Canum Venaticorum",
            # Canis Major (4 main stars)
            "Sirius",           "Canopus",          "Adara",             "Wezen",
            # Canis Minor (2 main stars)
            "Procyon",          "Gomeisa",
            # Capella (2 main stars)
            "Deneb Algedi",     "Maaz",
            # Carina (3 main stars)
            "Canopus",          "Avior",            "Miaplacidus",
            # Cassiopeia (5 main stars - the W pattern)
            "Schedar",          "Caph",             "Gamma Cassiopeiae", "Ruchbah",          "Segin",
            # Centaurus (3 main stars)
            "Rigil Kentaurus",  "Hadar",            "Muhlifain",
            # Cepheus (3 main stars)
            "Alderamin",        "Alfirk",           "Errai",
            # Cetus (3 main stars)
            "Menkar",           "Diphda",           "Beta Ceti",
            # Chamaeleon (2 main stars)
            "Alpha Chamaeleontis",  "Gamma Chamaeleontis",
            # Circinus (2 main stars)
            "Alpha Circini",    "Beta Circini",
            # Columba (2 main stars)
            "Phact",            "Wazn",
            # Coma Berenices (2 main stars)
            "Diadem",           "Beta Comae Berenices",
            # Corona Australis (3 main stars)
            "Alpha Coronae Australis",  "Beta Coronae Australis",  "Gamma Coronae Australis",
            # Corona Borealis (4 main stars - the arc)
            "Alphecca",         "Nusakan",          "Theta Coronae Borealis",  "Gamma Coronae Borealis",
            # Corvus (4 main stars - the diamond)
            "Gienah",           "Brachium",         "Kraz",              "Algorab",
            # Crater (3 main stars)
            "Delta Crateris",   "Alkes",            "Labrum",
            # Crux (4 main stars - Southern Cross)
            "Acrux",            "Gacrux",           "Iota Crucis",       "Delta Crucis",
            # Cygnus (5 main stars - Northern Cross)
            "Deneb",            "Sadr",             "Delta Cygni",       "Albireo",          "Epsilon Cygni",
            # Delphinus (4 main stars)
            "Sualocin",         "Rotanev",          "Gamma Delphini",    "Delta Delphini",
            # Dorado (2 main stars)
            "Alpha Doradus",    "Beta Doradus",
            # Draco (3 main stars)
            "Eltanin",          "Rastaban",         "Altais",
            # Equuleus (2 main stars)
            "Kitalpha",         "Gamma Equulei",
            # Eridanus (4 main stars)
            "Achernar",         "Cursa",            "Zanim",             "Azha",
            # Fornax (2 main stars)
            "Fornacis",         "Alpha Fornacis",
            # Gemini (5 main stars - Castor & Pollux twins)
            "Castor",           "Pollux",           "Alhena",            "Wasat",            "Kappa Geminorum",
            # Grus (3 main stars)
            "Alnair",           "Beta Gruis",       "Gamma Gruis",
            # Hercules (4 main stars)
            "Rasalgethi",       "Kornephoros",      "Sarin",             "Tau Herculis",
            # Horologium (2 main stars)
            "Alpha Horologi",   "Beta Horologi",
            # Hydra (4 main stars - largest constellation)
            "Alphard",          "Gamma Hydrae",     "Pi Hydrae",         "Epsilon Hydrae",
            # Hydrus (3 main stars)
            "Alpha Hydri",      "Beta Hydri",       "Gamma Hydri",
            # Indus (2 main stars)
            "Alpha Indi",       "Beta Indi",
            # Lacerta (2 main stars)
            "Alpha Lacertae",   "Beta Lacertae",
            # Leo (5 main stars - the sickle)
            "Regulus",          "Denebola",         "Algieba",           "Zosma",            "Chertan",
            # Leo Minor (2 main stars)
            "Praecipua",        "Beta Leonis Minoris",
            # Lepus (3 main stars)
            "Arneb",            "Nihal",            "Gamma Leporis",
            # Libra (4 main stars)
            "Zubeneschamali",   "Zubenelgenubi",    "Brachium",          "Zuben Elakrab",
            # Lupus (3 main stars)
            "Alpha Lupi",       "Beta Lupi",        "Gamma Lupi",
            # Lynx (2 main stars)
            "Alsciaukat",       "Alpha Lynxis",
            # Lyra (5 main stars)
            "Vega",             "Epsilon Lyrae",    "Zeta Lyrae",        "Delta Lyrae",      "Gamma Lyrae",
            # Mensa (2 main stars)
            "Alpha Mensae",     "Beta Mensae",
            # Microscopium (2 main stars)
            "Epsilon Microscopii",  "Alpha Microscopii",
            # Monoceros (3 main stars)
            "Alpha Monocerotis",    "Beta Monocerotis",  "Gamma Monocerotis",
            # Musca (3 main stars)
            "Alpha Muscae",     "Beta Muscae",      "Gamma Muscae",
            # Norma (3 main stars)
            "Gamma Normae",     "Epsilon Normae",   "Alpha Normae",
            # Octans (2 main stars)
            "Polaris Australis",    "Beta Octantis",
            # Ophiuchus (5 main stars)
            "Rasalhague",       "Sabik",            "Cebalrai",          "Yedprior",         "Yed Posterior",
            # Orion (6 main stars - one of the most recognizable)
            "Rigel",            "Betelgeuse",       "Bellatrix",         "Saiph",            "Alnitak",       "Alnilam",
            # Pavo (3 main stars)
            "Peacock",          "Delta Pavonis",    "Gamma Pavonis",
            # Pegasus (4 main stars - part of the Great Square)
            "Enif",             "Scheat",           "Algenib",           "Markab",
            # Perseus (4 main stars)
            "Mirfak",           "Algenib",          "Atik",              "Epsilon Persei",
            # Phoenix (3 main stars)
            "Ankaa",            "Chert",            "Psi Phoenicis",
            # Pictor (2 main stars)
            "Alpha Pictoris",   "Beta Pictoris",
            # Pisces (3 main stars)
            "Alrescha",         "Fum al Samakah",   "Omega Piscium",
            # Piscis Austrinus (2 main stars)
            "Fomalhaut",        "Delta Piscis Austrini",
            # Puppis (3 main stars)
            "Naos",             "Pi Puppis",        "Zeta Puppis",
            # Pyxis (2 main stars)
            "Alpha Pyxidis",    "Beta Pyxidis",
            # Reticulum (2 main stars)
            "Alpha Reticuli",   "Beta Reticuli",
            # Sagitta (3 main stars - small but distinct)
            "Sham",             "Almach",           "Delta Sagittae",
            # Sagittarius (4 main stars - the teapot asterism)
            "Kaus Australis",   "Kaus Media",       "Kaus Borealis",     "Ascella",
            # Scorpius (4 main stars - scorpion tail curve)
            "Antares",          "Shaula",           "Lesath",            "Acrab",
            # Sculptor (2 main stars)
            "Alpha Sculptoris", "Beta Sculptoris",
            # Scutum (2 main stars)
            "Alpha Scuti",      "Delta Scuti",
            # Serpens (3 main stars)
            "Unukalhai",        "Eta Serpentis",    "Gamma Serpentis",
            # Sextans (2 main stars)
            "Alpha Sextantis",  "Beta Sextantis",
            # Taurus (5 main stars - Pleiades visible)
            "Aldebaran",        "Nath",             "Alcyone",           "Electra",          "Maia",
            # Telescopium (2 main stars)
            "Alpha Telescopii", "Zeta Telescopii",
            # Triangulum (3 main stars)
            "Mothallah",        "Dulcamara",        "Caph",
            # Triangulum Australe (3 main stars)
            "Atria",            "Beta Trianguli Australis",  "Gamma Trianguli Australis",
            # Tucana (3 main stars)
            "Alpha Tucanae",    "Beta Tucanae",     "Gamma Tucanae",
            # Ursa Major (7 main stars - Big Dipper)
            "Dubhe",            "Merak",            "Phecda",            "Megrez",           "Alioth",       "Mizar",        "Alkaid",
            # Ursa Minor (5 main stars - Little Bear, Polaris)
            "Polaris",          "Kochab",           "Pherkad",           "Yildun",           "Epsilon Ursae Minoris",
            # Vela (3 main stars)
            "Gamma Velorum",    "Lambda Velorum",   "Delta Velorum",
            # Virgo (4 main stars)
            "Spica",            "Zavijava",         "Porrima",           "Vindemiatrix",
            # Volans (2 main stars)
            "Alpha Volantis",   "Beta Volantis",
            # Vulpecula (2 main stars)
            "Anser",            "Alpha Vulpeculae",
        ]
        excluded_named_stars = set(orion_star_names + taurus_star_names + scx_star_names + pointer_names + extra_names + ["Betelgeuse"])
        seen_extra_stars: set[str] = set()
        for star_name in extra_visible_star_names:
            if star_name in seen_extra_stars:
                continue
            seen_extra_stars.add(star_name)
            if star_name in excluded_named_stars:
                continue
            resolved = _resolve_named_star_icrs_deg(star_name, rotation_offset_deg=astropy_rotation_offset_deg)
            if resolved is None:
                continue
            star_ra_deg, star_dec_deg = resolved
            panel, sx, sy = _project_to_hemisphere(
                np.deg2rad(star_ra_deg),
                np.deg2rad(star_dec_deg),
            )
            star_ax = ax_l if panel == "north" else ax_r
            star_ax.scatter(
                [sx],
                [sy],
                s=5,
                c="#f8fafc",
                edgecolors="none",
                alpha=0.78,
                zorder=8,
            )

    # Plot a random sample of n supernovae from the galactic distribution (rasterized)
    if hasattr(ccsn, 'galactic_coords') and ccsn.galactic_coords is not None:
        n_sample = min(n_background_supernovae, len(ra_rot_supernovae))
        sample_indices = np.random.choice(len(ra_rot_supernovae), size=n_sample, replace=False)
        
        sampled_ra = ra_rot_supernovae[sample_indices]
        sampled_dec = dec_supernovae[sample_indices]
        
        # Project to hemispheres
        north_sample_mask = sampled_dec >= 0
        south_sample_mask = sampled_dec < 0
        
        ra_n_sample = sampled_ra[north_sample_mask]
        dec_n_sample = sampled_dec[north_sample_mask]
        r_n_sample = (np.pi / 2 - dec_n_sample) / (np.pi / 2)
        x_n_sample = r_n_sample * np.sin(ra_n_sample)
        y_n_sample = r_n_sample * np.cos(ra_n_sample)
        
        ra_s_sample = sampled_ra[south_sample_mask]
        dec_s_sample = sampled_dec[south_sample_mask]
        r_s_sample = (np.pi / 2 + dec_s_sample) / (np.pi / 2)
        x_s_sample = -r_s_sample * np.sin(ra_s_sample)
        y_s_sample = r_s_sample * np.cos(ra_s_sample)
        
        # Plot as rasterized scatter (single rasterized layer, not individual points)
        ax_l.scatter(
            x_n_sample,
            y_n_sample,
            s=2,
            c="#d1d5db",
            edgecolors="none",
            alpha=0.4,
            zorder=7,
            rasterized=True,
        )
        ax_r.scatter(
            x_s_sample,
            y_s_sample,
            s=2,
            c="#d1d5db",
            edgecolors="none",
            alpha=0.4,
            zorder=7,
            rasterized=True,
        )

    # Plot background supernovae (10000 random samples, regardless of hemisphere)
    np.random.seed(42)  # For reproducibility
    
    # Sample n_background_supernovae from all supernovae
    total_sn = len(ra_rot_supernovae)
    if n_background_supernovae is not None and total_sn > n_background_supernovae:
        sample_indices = np.random.choice(total_sn, size=n_background_supernovae, replace=False)
        ra_stars = ra_rot_supernovae[sample_indices]
        dec_stars = dec_supernovae[sample_indices]
    else:
        ra_stars = ra_rot_supernovae
        dec_stars = dec_supernovae
    
    # Project to appropriate hemispheres (vectorized)
    north_mask = dec_stars >= 0
    south_mask = dec_stars < 0
    
    # North hemisphere
    if np.any(north_mask):
        ra_n = ra_stars[north_mask]
        dec_n = dec_stars[north_mask]
        r_n = (np.pi / 2 - dec_n) / (np.pi / 2)
        x_n = r_n * np.sin(ra_n)
        y_n = r_n * np.cos(ra_n)
        
        ax_l.scatter(
            x_n,
            y_n,
            s=0.5,
            c="lightgray",
            edgecolors="none",
            alpha=0.25,
            zorder=6,
            rasterized=True,
        )
    
    # South hemisphere
    if np.any(south_mask):
        ra_s = ra_stars[south_mask]
        dec_s = dec_stars[south_mask]
        r_s = (np.pi / 2 + dec_s) / (np.pi / 2)
        x_s = -r_s * np.sin(ra_s)
        y_s = r_s * np.cos(ra_s)
        
        ax_r.scatter(
            x_s,
            y_s,
            s=0.5,
            c="lightgray",
            edgecolors="none",
            alpha=0.25,
            zorder=6,
            rasterized=True,
        )

    # Plot detector markers when example mode is enabled.
    if show_detectors and detector_markers:
        # Define L-shaped marker (vertical arm on left, horizontal base at bottom)
        t = 0.18      # arm thickness
        L = 0.40      # half-length

        l_marker_verts = np.array([
            [-L, -L],
            [ L, -L],
            [ L, -L+t],
            [-L+t, -L+t],
            [-L+t,  L],
            [-L,    L],
            [-L,   -L],
        ])
        
        def rotate_marker_verts(verts, angle_rad):
            """Rotate marker vertices by given angle in radians."""
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            return verts @ rot_matrix.T
        
        for det_name, det_ra, det_dec, det_color in detector_markers:
            det_panel, det_x, det_y = _project_to_hemisphere(det_ra, det_dec)
            det_ax = ax_l if det_panel == "north" else ax_r
            
            # Rotate marker to point radially outward from pole
            # Compute the angle in plot coordinates using atan2
            # Add π/2 to align the vertical arm of the L to point outward radially
            plot_angle = np.arctan2(det_x, det_y) + np.pi / 2
            rotated_verts = rotate_marker_verts(l_marker_verts, plot_angle)

            print(Path)
            print(type(Path))
            l_marker = MarkerStyle(Path(rotated_verts))
            
            det_ax.scatter(
                [det_x],
                [det_y],
                s=200 if format == "poster" else 50,
                marker=l_marker,
                c=det_color,
                edgecolors="none",   # or edgecolors=det_color
                linewidths=0,
                alpha=0.9,
                zorder=10,
            )
            # Add detector label
            label_offset_x = 0.045 if "Hanford" in det_name or "Livingston" in det_name else 0.045
            label_offset_y = 0.025
            det_ax.text(
                det_x + label_offset_x,
                det_y + label_offset_y,
                det_name,
                color=text_color,
                fontsize=fontsize_constellation,
                ha="left",
                va="center",
                alpha=0.95,
                zorder=10,
            )

    if galaxy:
        ax_r.plot(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=13,
            markerfacecolor="black",
            markeredgecolor="orange",
            markeredgewidth=1.7,
            label="Galactic Center: Sgr A*",
        )

    if true_loc_panel is not None:
        ax_r.plot(
            [],
            [],
            marker="x",
            linestyle="None",
            markersize=11,
            markeredgecolor=SIGNAL_COLOUR,
            markerfacecolor="none",
            markeredgewidth=1.6,
            label="True Supernova Location",
        )
    
    if show_detectors and detector_markers:
        # Create base L marker for legend
        t = 0.18      # arm thickness
        L = 0.40      # half-length

        l_marker_verts_legend = np.array([
            [-L, -L],
            [ L, -L],
            [ L, -L+t],
            [-L+t, -L+t],
            [-L+t,  L],
            [-L,    L],
            [-L,   -L],
        ])
        l_marker_legend = MarkerStyle(Path(l_marker_verts_legend))
        
        ax_r.plot(
            [],
            [],
            marker=l_marker_legend,
            linestyle="None",
            markersize=10,
            markerfacecolor=text_color,
            markeredgecolor=text_color,
            markeredgewidth=0.0,
            label="Gravitational Wave Detectors",
        )
    
    if format == "thesis":
        ax_r.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),   # 2% below the axes
            ncol=3,
            frameon=False,
            labelcolor=text_color,
            fontsize=fontsize_main,
            handletextpad=0.5,
            columnspacing=1.0,
            borderaxespad=0.2,
        )
    else:
        ax_r.legend(
            loc="lower right",
            bbox_to_anchor=(0.98, -0.08),
            frameon=False,
            labelcolor=text_color,
            fontsize=fontsize_main,
            borderaxespad=0.0,
        )

    # -------------------------------------------------
    # Earth coastlines
    # -------------------------------------------------
    if coastline:
        coast_n, coast_s = _coastline_segments()

        print(len(coast_n))
        print(coast_n[0].shape)

        ax_l.add_collection(
            LineCollection(
                coast_n,
                colors="#C0BDBD",
                linewidths=0.3,
                alpha=1.0,
                zorder=1
            )
        )

        ax_r.add_collection(
            LineCollection(
                coast_s,
                colors="#C0BDBD",
                linewidths=0.3,
                alpha=1.0,
                zorder=1
            )
        )


    # Determine format from filename extension
    file_format = None
    if fname.lower().endswith('.svg'):
        file_format = 'svg'

    if background == "black":
        background = "#0f0d33"
    
    save_kwargs = {
        "dpi": 100 if file_format == 'svg' else 300,
        "facecolor": background if not transparent else None,
        "edgecolor": "none",
        "pad_inches": 0,
        "transparent": transparent,
        "bbox_inches": None,
    }
    if file_format:
        save_kwargs["format"] = file_format
    
    
    # Optimize SVG file size by reducing decimal precision
    # if fname.endswith('.svg'):
    #     import re
    #     try:
    #         with open(fname, 'r', encoding='utf-8') as f:
    #             svg_content = f.read()
            
    #         # Reduce decimal precision in coordinates (8 decimals → 4 decimals)
    #         # This regex finds numbers with many decimal places and rounds them
    #         # 4 decimals is enough for visual quality while reducing file size
    #         svg_content = re.sub(r'(\d+\.\d{5,})', lambda m: f'{float(m.group(1)):.4f}', svg_content)
            
    #         with open(fname, 'w', encoding='utf-8') as f:
    #             f.write(svg_content)
            
    #         import os
    #         size_kb = os.path.getsize(fname) / 1024
    #         print(f"Optimized SVG: {fname} ({size_kb:.1f} KB)")
    #     except Exception as e:
    #         print(f"SVG optimization failed: {e}")

    plt.savefig(fname, **save_kwargs)
    plt.show()
    plt.rcdefaults()


@lru_cache(maxsize=1)
def _coastline_segments():
    """
    Return projected coastline line segments for the north and south hemispheres.
    """

    filename = shpreader.natural_earth(
        resolution="50m",
        category="physical",
        name="coastline"
    )

    reader = shpreader.Reader(filename)

    north_segments = []
    south_segments = []

    for record in reader.records():

        geom = record.geometry

        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = geom.geoms
        else:
            continue

        for line in lines:

            coords = np.asarray(line.coords)

            current_north = []
            current_south = []

            for lon, lat in coords:

                hemi, x, y = _project_to_hemisphere(
                    np.deg2rad(lon),
                    np.deg2rad(lat),
                )

                if hemi == "north":
                    if current_south:
                        if len(current_south) > 1:
                            south_segments.append(np.asarray(current_south))
                        current_south = []

                    current_north.append((x, y))

                else:
                    if current_north:
                        if len(current_north) > 1:
                            north_segments.append(np.asarray(current_north))
                        current_north = []

                    current_south.append((x, y))

            if len(current_north) > 1:
                north_segments.append(np.asarray(current_north))

            if len(current_south) > 1:
                south_segments.append(np.asarray(current_south))

    return north_segments, south_segments

def _draw_curved_ra_label(
    ax,
    panel: str,
    center_ang_deg: float,
    radius: float,
    text: str,
    color: str,
    fontsize: float,
    alpha: float,
    char_spacing_deg: float | None = None,
    zorder: int = 9,
) -> None:
    """Draw `text` as individual characters curved tangent to the hemisphere
    circle, centered on `center_ang_deg` (same RA-angle convention used
    elsewhere: 0 deg = top, increasing clockwise for north / mirrored for
    south) at the given `radius` from the pole.

    Rotation is derived from the true direction of travel as RA increases
    (which differs between north and south due to south's mirrored x),
    not from radial position alone. Away from the 90/270 deg positions,
    the flip decision avoids upside-down text by keeping the reading
    direction within +/-90 deg of horizontal. Exactly at 90/270 deg the
    tangent is perfectly vertical, so there's no real "upside-down" to
    avoid there - both choices are equally sideways - so the tie is
    resolved instead by explicitly picking whichever orientation faces
    the pole (inward), which is otherwise inconsistent between panels
    due to south's mirrored geometry.
    """
    if char_spacing_deg is None:
        char_spacing_deg = fontsize * 0.1

    def _pos(ang_deg):
        ang = np.deg2rad(ang_deg)
        if panel == "north":
            return radius * np.sin(ang), radius * np.cos(ang)
        else:
            return -radius * np.sin(ang), radius * np.cos(ang)

    def _travel_angle_deg(ang_deg):
        # Direction of increasing RA (d/dang of position), which is what
        # "reads correctly" means - not the radial/position angle.
        ang = np.deg2rad(ang_deg)
        if panel == "north":
            dx, dy = np.cos(ang), -np.sin(ang)
        else:
            dx, dy = -np.cos(ang), -np.sin(ang)
        return np.degrees(np.arctan2(dy, dx))

    # Decide the flip once for the whole label, from its center, so
    # rotation and character order stay in agreement.
    # base_rot = _travel_angle_deg(center_ang_deg)
    # norm = ((base_rot + 180.0) % 360.0) - 180.0

    base_rot = _travel_angle_deg(center_ang_deg)
    norm = ((base_rot + 180.0) % 360.0) - 180.0

    # Flip only strictly between 90 and 270 degrees (through 180) -
    # never across 0. At exactly 90 or 270 degrees, don't flip by default.
    flipped = norm > 90.0 or norm < -90.0

    # norm == +90 corresponds to the 270 deg position for both panels
    # (base_rot is identical there since the cos(ang) term cancels out),
    # but only the south panel's orientation comes out wrong at that
    # specific spot - invert just for south there.
    if panel == "south" and np.isclose(norm, 90.0):
        flipped = not flipped

    n = len(text)
    offsets = (np.arange(n) - (n - 1) / 2.0) * char_spacing_deg
    if flipped:
        offsets = offsets[::-1]

    for char, off in zip(text, offsets):
        ang_deg = center_ang_deg + off
        x, y = _pos(ang_deg)
        rot = _travel_angle_deg(ang_deg)
        if flipped:
            rot += 180.0
        rot = ((rot + 180.0) % 360.0) - 180.0

        ax.text(
            x, y, char,
            color=color,
            fontsize=fontsize,
            ha="center",
            va="center",
            rotation=rot,
            rotation_mode="anchor",
            alpha=alpha,
            zorder=zorder,
        )