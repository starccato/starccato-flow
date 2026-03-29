"""Sky plotting utilities."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from . import set_plot_style

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


ASTROPY_RA_BACKSHIFT_DEG = 60.0

IMPORTANT_CONSTELLATIONS = {
    "Ori": "Orion",
    "Tau": "Taurus",
    "Gem": "Gemini",
    "CMa": "Canis Major",
    "UMa": "Ursa Major",
    "Cas": "Cassiopeia",
    "Sco": "Scorpius",
    "Sgr": "Sagittarius",
    "Aql": "Aquila",
    "Peg": "Pegasus",
    "Cru": "Crux",
    "Cen": "Centaurus"
}


def _backshift_astropy_ra_deg(ra_deg: np.ndarray | float, backshift_deg: float = ASTROPY_RA_BACKSHIFT_DEG) -> np.ndarray | float:
    """Shift Astropy-resolved RA values back by a fixed degree offset."""
    return np.mod(ra_deg - backshift_deg, 360.0)


def _get_betelgeuse_icrs_deg() -> tuple[float, float, str]:
    """Return Betelgeuse ICRS (RA, Dec) in degrees and a source label."""
    if not _ASTROPY_AVAILABLE:
        return np.nan, np.nan, "unavailable"

    try:
        coord = SkyCoord.from_name("Betelgeuse")
        ra_deg = float(_backshift_astropy_ra_deg(float(coord.ra.deg)))
        return ra_deg, float(coord.dec.deg), "astropy"
    except Exception:
        return np.nan, np.nan, "unavailable"


def _resolve_named_star_icrs_deg(name: str) -> tuple[float, float] | None:
    """Resolve a star name to ICRS RA/Dec degrees using Astropy only."""
    if not _ASTROPY_AVAILABLE:
        return None

    try:
        coord = SkyCoord.from_name(name)
        ra_deg = float(_backshift_astropy_ra_deg(float(coord.ra.deg)))
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
    rotation_rad: float,
) -> tuple[str, float, float]:
    """Project a single (RA, Dec) point into north/south hemisphere panel coordinates."""
    ra_use = _rotate_ra(ra_val, rotation_rad)
    if dec_val >= 0.0:
        rr = (np.pi / 2 - dec_val) / (np.pi / 2)
        xx = rr * np.sin(ra_use)
        yy = rr * np.cos(ra_use)
        return "north", float(xx), float(yy)

    rr = (np.pi / 2 + dec_val) / (np.pi / 2)
    xx = -rr * np.sin(ra_use)
    yy = rr * np.cos(ra_use)
    return "south", float(xx), float(yy)


def _rotate_ra(ra_rad: np.ndarray | float, rotation_rad: float) -> np.ndarray | float:
    """Apply the shared sky-view RA rotation used by all plotted objects."""
    return np.mod(ra_rad + rotation_rad, 2 * np.pi)


def _constellation_border_segments(
    rotation_rad: float,
    n_ra: int = 720,
    n_dec: int = 360,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected north/south line segments tracing constellation borders."""
    if not _ASTROPY_AVAILABLE:
        return np.empty((0, 2, 2)), np.empty((0, 2, 2))

    # Sample the sky and classify each point by constellation.
    ra_deg = np.linspace(0.0, 360.0, n_ra, endpoint=False)
    dec_deg = np.linspace(-89.5, 89.5, n_dec)
    ra_mesh, dec_mesh = np.meshgrid(ra_deg, dec_deg)

    sky = SkyCoord(ra=ra_mesh.ravel() * u.deg, dec=dec_mesh.ravel() * u.deg, frame="icrs")
    const_names = np.asarray(sky.get_constellation(short_name=True)).reshape(dec_mesh.shape)

    _, inv = np.unique(const_names, return_inverse=True)
    const_id = inv.reshape(const_names.shape)

    # Edge masks where neighboring cells differ in constellation id.
    dh = const_id[:, 1:] != const_id[:, :-1]
    dv = const_id[1:, :] != const_id[:-1, :]

    ra_mid_h = 0.5 * (ra_deg[1:] + ra_deg[:-1])
    dec_mid_v = 0.5 * (dec_deg[1:] + dec_deg[:-1])

    ra_h = np.broadcast_to(ra_mid_h, dh.shape)[dh]
    dec_h = np.broadcast_to(dec_deg[:, None], dh.shape)[dh]
    ra_v = np.broadcast_to(ra_deg, dv.shape)[dv]
    dec_v = np.broadcast_to(dec_mid_v[:, None], dv.shape)[dv]

    # Segment lengths based on local angular spacing, then projected per midpoint.
    d_ra = 360.0 / n_ra
    d_dec = 179.0 / (n_dec - 1)

    ra_h0 = ra_h
    dec_h0 = dec_h - 0.5 * d_dec
    ra_h1 = ra_h
    dec_h1 = dec_h + 0.5 * d_dec

    ra_v0 = ra_v - 0.5 * d_ra
    dec_v0 = dec_v
    ra_v1 = ra_v + 0.5 * d_ra
    dec_v1 = dec_v

    ra0 = np.concatenate([ra_h0, ra_v0])
    dec0 = np.concatenate([dec_h0, dec_v0])
    ra1 = np.concatenate([ra_h1, ra_v1])
    dec1 = np.concatenate([dec_h1, dec_v1])

    # Wrap RA and clamp Dec to valid ranges.
    ra0 = np.mod(ra0, 360.0)
    ra1 = np.mod(ra1, 360.0)
    dec0 = np.clip(dec0, -89.9, 89.9)
    dec1 = np.clip(dec1, -89.9, 89.9)

    # Avoid very long wrap-around segments near the RA seam.
    dra_abs = np.abs(ra1 - ra0)
    seam_cross = np.minimum(dra_abs, 360.0 - dra_abs) > 5.0
    keep = ~seam_cross

    ra0 = ra0[keep]
    dec0 = dec0[keep]
    ra1 = ra1[keep]
    dec1 = dec1[keep]

    if ra0.size == 0:
        return np.empty((0, 2, 2)), np.empty((0, 2, 2))

    ra0 = _backshift_astropy_ra_deg(ra0)
    ra1 = _backshift_astropy_ra_deg(ra1)

    ra0_rad = np.deg2rad(ra0)
    dec0_rad = np.deg2rad(dec0)
    ra1_rad = np.deg2rad(ra1)
    dec1_rad = np.deg2rad(dec1)

    ra0_use = np.mod(ra0_rad + rotation_rad, 2 * np.pi)
    ra1_use = np.mod(ra1_rad + rotation_rad, 2 * np.pi)

    p0_north = dec0_rad >= 0.0
    p1_north = dec1_rad >= 0.0
    north = p0_north & p1_north
    south = (~p0_north) & (~p1_north)

    def _proj(ra_use_val: np.ndarray, dec_val: np.ndarray, south_hemi: bool) -> tuple[np.ndarray, np.ndarray]:
        if south_hemi:
            rr = (np.pi / 2 + dec_val) / (np.pi / 2)
            x_val = -rr * np.sin(ra_use_val)
            y_val = rr * np.cos(ra_use_val)
        else:
            rr = (np.pi / 2 - dec_val) / (np.pi / 2)
            x_val = rr * np.sin(ra_use_val)
            y_val = rr * np.cos(ra_use_val)
        return x_val, y_val

    x0n, y0n = _proj(ra0_use[north], dec0_rad[north], south_hemi=False)
    x1n, y1n = _proj(ra1_use[north], dec1_rad[north], south_hemi=False)
    x0s, y0s = _proj(ra0_use[south], dec0_rad[south], south_hemi=True)
    x1s, y1s = _proj(ra1_use[south], dec1_rad[south], south_hemi=True)

    seg_n = np.stack([np.column_stack([x0n, y0n]), np.column_stack([x1n, y1n])], axis=1) if x0n.size else np.empty((0, 2, 2))
    seg_s = np.stack([np.column_stack([x0s, y0s]), np.column_stack([x1s, y1s])], axis=1) if x0s.size else np.empty((0, 2, 2))

    # Deduplicate overlapping segments so lines do not appear stacked.
    def _dedupe_segments(seg: np.ndarray) -> np.ndarray:
        if seg.size == 0:
            return seg
        a = np.round(seg[:, 0, :], 5)
        b = np.round(seg[:, 1, :], 5)
        flip = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
        p = np.where(flip[:, None], b, a)
        q = np.where(flip[:, None], a, b)
        key = np.hstack([p, q])
        _, idx = np.unique(key, axis=0, return_index=True)
        return seg[np.sort(idx)]

    seg_n = _dedupe_segments(seg_n)
    seg_s = _dedupe_segments(seg_s)

    return seg_n, seg_s


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


def plot_galactic_supernovae_polar_hemispheres(
    ccsn,
    fname: str = "plots/galactic_supernovae_polar_hemispheres.png",
    rotation_deg: float = 60.0,
    show_constellation_borders: bool = False,
    show_important_constellation_labels: bool = True,
    show: bool = True,
    dpi: int = 160,
    background: str = "black",
    font_family: str = "sans-serif",
    font_name: str = "Avenir",
    red_blob_mode: str = "middle_star",
) -> None:
    """Plot CCSN sky distribution as tangent north/south pole-centered hemispheres.

    Args:
        ccsn: Supernovae-like object exposing ``ra``, ``dec`` and
            ``get_galactic_center_direction()``.
        fname: Output image path.
        rotation_deg: Global RA view rotation in degrees.
        show_constellation_borders: If True, overlay IAU constellation boundaries.
        show_important_constellation_labels: If True and constellation borders are enabled,
            annotate key constellations with labels.
        show: If True, call ``plt.show()``.
        dpi: Image save DPI.
        background: Background color theme ("white" or "black").
        font_family: Font family to use.
        font_name: Specific font name.
        red_blob_mode: Red contour center mode. One of
            ``"middle_star"``, ``"density_peak"``, ``"true_center"``.
    """
    set_plot_style(background, font_family, font_name)
    ra = np.mod(np.asarray(ccsn.ra), 2 * np.pi)
    dec = np.asarray(ccsn.dec)

    rotation = np.deg2rad(rotation_deg)
    # Milky Way stars and Astropy-resolved objects use the same RA rotation.
    ra_rot = _rotate_ra(ra, rotation)

    fig = plt.figure(figsize=(12, 6.8), facecolor="black")
    # Keep a small canvas margin so boundary lines and circles are not clipped at image edges.
    ax_l = fig.add_axes([0.03, 0.03, 0.47, 0.94], facecolor="black")
    ax_r = fig.add_axes([0.50, 0.03, 0.47, 0.94], facecolor="black")

    north_mask = dec >= 0
    ra_n = ra_rot[north_mask]
    dec_n = dec[north_mask]
    r_n = (np.pi / 2 - dec_n) / (np.pi / 2)
    x_n = r_n * np.sin(ra_n)
    y_n = r_n * np.cos(ra_n)

    south_mask = dec <= 0
    ra_s = ra_rot[south_mask]
    dec_s = dec[south_mask]
    r_s = (np.pi / 2 + dec_s) / (np.pi / 2)
    x_s = -r_s * np.sin(ra_s)
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

    probs = [0.995, 0.80, 0.50, 0.25]
    combined_vals = np.concatenate([
        h_n_smooth.T[inside_circle],
        h_s_smooth.T[inside_circle],
    ])
    combined_vals = combined_vals[combined_vals > 0]
    if combined_vals.size == 0:
        thr_shared = [1.0 for _ in probs]
    else:
        vals = np.sort(combined_vals)[::-1]
        cdf = np.cumsum(vals) / np.sum(vals)
        thr_shared = []
        for p in probs:
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

    ax_l.contourf(xcenters, ycenters, h_n_plot, levels=fill_levels_shared, colors=fill_colors, antialiased=True)
    for r_lat in lat_radii:
        ax_l.plot(r_lat * np.cos(theta), r_lat * np.sin(theta), color="white", alpha=0.13, lw=0.75)
    ax_l.plot(np.cos(theta), np.sin(theta), color="white", lw=1.4)
    ax_l.axhline(0, color="white", alpha=0.18, lw=0.8)
    ax_l.axvline(0, color="white", alpha=0.18, lw=0.8)
    ax_l.text(
        0.0,
        0.0,
        "North\nPole",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
        multialignment="center",
        alpha=0.95,
    )

    ax_r.contourf(xcenters, ycenters, h_s_plot, levels=fill_levels_shared, colors=fill_colors, antialiased=True)
    for r_lat in lat_radii:
        ax_r.plot(r_lat * np.cos(theta), r_lat * np.sin(theta), color="white", alpha=0.13, lw=0.75)
    ax_r.plot(np.cos(theta), np.sin(theta), color="white", lw=1.4)
    ax_r.axhline(0, color="white", alpha=0.18, lw=0.8)
    ax_r.axvline(0, color="white", alpha=0.18, lw=0.8)
    ax_r.text(
        0.0,
        0.0,
        "South\nPole",
        color="white",
        fontsize=10,
        ha="center",
        va="center",
        multialignment="center",
        alpha=0.95,
    )

    # RA/Dec ticks for orientation in each hemisphere panel.
    ra_tick_deg = np.arange(0, 360, 30)
    ra_label_deg = np.arange(0, 360, 60)
    for ra_deg in ra_tick_deg:
        ang = np.deg2rad(float(ra_deg))
        x_in = 1.00 * np.sin(ang)
        y_in = 1.00 * np.cos(ang)
        x_out = 1.045 * np.sin(ang)
        y_out = 1.045 * np.cos(ang)
        for ax in (ax_l, ax_r):
            ax.plot([x_in, x_out], [y_in, y_out], color="white", alpha=0.45, lw=0.75, zorder=6)

    for ra_deg in ra_label_deg:
        ang = np.deg2rad(float(ra_deg))
        x_lbl = 1.085 * np.sin(ang)
        y_lbl = 1.085 * np.cos(ang)
        ra_hours = int((ra_deg // 15) % 24)
        label = f"{ra_hours}h"
        for ax in (ax_l, ax_r):
            ax.text(
                x_lbl,
                y_lbl,
                label,
                color="white",
                fontsize=7.0,
                ha="center",
                va="center",
                alpha=0.75,
                zorder=6,
            )

    dec_abs_ticks = [80, 60, 40, 20]
    # Place Dec ticks on the 0h/24h RA meridian.
    ux = 0.0
    uy = 1.0
    for dec_abs in dec_abs_ticks:
        r_tick = (90.0 - float(dec_abs)) / 90.0
        x0 = r_tick * ux
        y0 = r_tick * uy

        # North: positive Dec labels.
        ax_l.plot(
            [x0 - 0.012, x0 + 0.012],
            [y0, y0],
            color="white",
            alpha=0.45,
            lw=0.75,
            zorder=6,
        )
        ax_l.text(
            x0 + 0.020,
            y0,
            f"+{dec_abs}°",
            color="white",
            fontsize=7.0,
            ha="left",
            va="center",
            alpha=0.75,
            zorder=6,
        )

        # South: negative Dec labels.
        ax_r.plot(
            [x0 - 0.012, x0 + 0.012],
            [y0, y0],
            color="white",
            alpha=0.45,
            lw=0.75,
            zorder=6,
        )
        ax_r.text(
            x0 + 0.020,
            y0,
            f"-{dec_abs}°",
            color="white",
            fontsize=7.0,
            ha="left",
            va="center",
            alpha=0.75,
            zorder=6,
        )

    if show_constellation_borders:
        if _ASTROPY_AVAILABLE:
            seg_n, seg_s = _constellation_border_segments(rotation)
            if seg_n.size:
                ax_l.add_collection(
                    LineCollection(seg_n, colors="#e2e8f0", linewidths=0.36, alpha=0.34, zorder=4)
                )
            if seg_s.size:
                ax_r.add_collection(
                    LineCollection(seg_s, colors="#e2e8f0", linewidths=0.36, alpha=0.34, zorder=4)
                )

            if show_important_constellation_labels:
                centers = _constellation_centers_icrs_deg()
                for short_name, label in IMPORTANT_CONSTELLATIONS.items():
                    if short_name not in centers:
                        continue
                    ra_c_deg, dec_c_deg = centers[short_name]
                    ra_c_deg = float(_backshift_astropy_ra_deg(ra_c_deg))
                    panel, cx, cy = _project_to_hemisphere(np.deg2rad(ra_c_deg), np.deg2rad(dec_c_deg), rotation)
                    if cx * cx + cy * cy > 0.97 * 0.97:
                        continue
                    lbl_ax = ax_l if panel == "north" else ax_r

                    dx = 0.0
                    dy = 0.0
                    if short_name == "Ori":
                        dx = -0.022
                        dy = 0.062
                    elif short_name == "Tau":
                        dx = 0.085
                    elif short_name == "CMa":
                        dy = 0.024
                    elif short_name == "Cru":
                        dy = -0.060

                    lbl_ax.text(
                        cx + dx,
                        cy + dy,
                        label,
                        color="#e2e8f0",
                        fontsize=7.0,
                        ha="center",
                        va="center",
                        alpha=0.8,
                        zorder=6,
                    )
        else:
            print("Constellation borders requested, but astropy is not installed in this environment.")

    # Make circles touch at center seam while keeping extra margin on outer edges.
    ax_l.set_xlim(-1.03, 1.00)
    ax_r.set_xlim(-1.00, 1.03)

    for ax in (ax_l, ax_r):
        ax.set_aspect("equal", adjustable="box")
        ax.set_ylim(-1.03, 1.03)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    gc_ra, gc_dec = ccsn.get_galactic_center_direction()

    # Use the true galactic center for black hole visualization.
    true_gc_panel, true_gc_x, true_gc_y = _project_to_hemisphere(gc_ra, gc_dec, rotation)

    # Choose the red contour center for the accretion-blob style overlay.
    if red_blob_mode == "true_center":
        gc_panel, gc_x, gc_y = true_gc_panel, true_gc_x, true_gc_y
    elif red_blob_mode == "density_peak":
        # Use the highest posterior-density pixel across both hemispheres.
        n_plot = np.ma.array(h_n_smooth.T, mask=~inside_circle)
        s_plot = np.ma.array(h_s_smooth.T, mask=~inside_circle)
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
    else:
        # Default keeps legacy behavior: center red blob on the middle sample.
        star_idx = len(ra) // 2
        blob_ra = ra[star_idx]
        blob_dec = dec[star_idx]
        gc_panel, gc_x, gc_y = _project_to_hemisphere(blob_ra, blob_dec, rotation)

    # Resolve Betelgeuse via Astropy name resolution (no hardcoded coordinate fallback).
    betelgeuse_ra_deg, betelgeuse_dec_deg, betel_source = _get_betelgeuse_icrs_deg()
    if np.isfinite(betelgeuse_ra_deg) and np.isfinite(betelgeuse_dec_deg):
        betel_panel, betel_x, betel_y = _project_to_hemisphere(
            np.deg2rad(betelgeuse_ra_deg),
            np.deg2rad(betelgeuse_dec_deg),
            rotation,
        )
    else:
        betel_panel, betel_x, betel_y = None, np.nan, np.nan

    blob_sigma = 0.12
    blob_radius = 3.0 * blob_sigma
    dist2_gc = (xxc - gc_x) ** 2 + (yyc - gc_y) ** 2
    gc_blob = np.exp(-0.5 * dist2_gc / (blob_sigma**2))
    gc_blob[dist2_gc > blob_radius**2] = 0.0

    gc_mask = inside_circle & (gc_blob > 0.0)
    gc_thr = _hpd_thresholds(gc_blob, gc_mask, probs)
    gc_levels = np.sort(np.array(gc_thr, dtype=float))
    gc_top = max(gc_levels[-1] * 1.001, np.max(gc_blob[gc_mask]) * 1.001)
    gc_fill_levels = np.concatenate([gc_levels, [gc_top]])
    red_bases = ["#7f1d1d", "#dc2626", "#f87171", "#fecaca"]
    red_fill_colors = [
        to_rgba(red_bases[0], alpha=0.20),
        to_rgba(red_bases[1], alpha=0.40),
        to_rgba(red_bases[2], alpha=0.62),
        to_rgba(red_bases[3], alpha=0.88),
    ]

    gc_blob_plot = np.ma.array(gc_blob, mask=~inside_circle)
    if gc_panel == "north":
        ax_l.contourf(
            xcenters,
            ycenters,
            gc_blob_plot,
            levels=gc_fill_levels,
            colors=red_fill_colors,
            antialiased=True,
        )
    else:
        ax_r.contourf(
            xcenters,
            ycenters,
            gc_blob_plot,
            levels=gc_fill_levels,
            colors=red_fill_colors,
            antialiased=True,
        )

    gc_ax = ax_l if gc_panel == "north" else ax_r
    gc_ax.scatter(
        [gc_x],
        [gc_y],
        s=28,
        c="#fee2e2",
        edgecolors="#7f1d1d",
        linewidths=0.8,
        zorder=7,
    )

    # Black hole visualization at the true galactic center.
    bh_ax = ax_l if true_gc_panel == "north" else ax_r

    # Red accretion disk (outer ring).
    bh_disk_outer = Circle(
        (true_gc_x, true_gc_y), 0.015, color="#dc2626", alpha=0.8, zorder=8
    )
    bh_ax.add_patch(bh_disk_outer)

    # Black hole interior (event horizon).
    bh_interior = Circle(
        (true_gc_x, true_gc_y), 0.010, color="black", alpha=0.95, zorder=9
    )
    bh_ax.add_patch(bh_interior)

    if betel_panel is not None:
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
            color="#fde68a",
            fontsize=8.5,
            ha="right",
            va="center",
            zorder=8,
        )

        if betel_panel == gc_panel:
            betel_ax.plot(
                [gc_x, betel_x],
                [gc_y, betel_y],
                color="#f59e0b",
                alpha=0.55,
                lw=0.9,
                ls="--",
                zorder=6,
            )

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
        resolved = _resolve_named_star_icrs_deg(star_name)
        if resolved is None:
            continue
        star_ra_deg, star_dec_deg = resolved
        orion_proj[star_name] = _project_to_hemisphere(
            np.deg2rad(star_ra_deg),
            np.deg2rad(star_dec_deg),
            rotation,
        )

    for a_name, b_name in orion_edges:
        if a_name not in orion_proj or b_name not in orion_proj:
            continue
        a_panel, axx, ayy = orion_proj[a_name]
        b_panel, bxx, byy = orion_proj[b_name]
        
        if a_panel == b_panel:
            # Both stars on same hemisphere: simple connection
            orion_ax = ax_l if a_panel == "north" else ax_r
            orion_ax.plot(
                [axx, bxx],
                [ayy, byy],
                color="#e5e7eb",
                alpha=0.85,
                lw=1.0,
                zorder=8,
            )
        else:
            # Stars on different hemispheres: connect through the seam
            # North is on ax_l (left panel), South is on ax_r (right panel)
            # Seam at x=1.00 on left (north), x=-1.00 on right (south)
            seam_y = 0.5 * (ayy + byy)
            
            # Draw on north hemisphere from star to right edge seam
            if a_panel == "north":
                ax_l.plot(
                    [axx, 1.00],
                    [ayy, seam_y],
                    color="#e5e7eb",
                    alpha=0.85,
                    lw=1.0,
                    zorder=8,
                )
                # Draw on south hemisphere from left edge seam to star
                ax_r.plot(
                    [-1.00, bxx],
                    [seam_y, byy],
                    color="#e5e7eb",
                    alpha=0.85,
                    lw=1.0,
                    zorder=8,
                )
            else:
                # A is south, B is north: reverse
                ax_r.plot(
                    [axx, -1.00],
                    [ayy, seam_y],
                    color="#e5e7eb",
                    alpha=0.85,
                    lw=1.0,
                    zorder=8,
                )
                ax_l.plot(
                    [1.00, bxx],
                    [seam_y, byy],
                    color="#e5e7eb",
                    alpha=0.85,
                    lw=1.0,
                    zorder=8,
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
    taurus_edges = [
        ("Aldebaran", "Epsilon Tauri"),
        ("Epsilon Tauri", "Gamma Tauri"),
        ("Gamma Tauri", "Delta Tauri"),
        ("Aldebaran", "Delta Tauri"),
        ("Gamma Tauri", "Elnath"),
        ("Delta Tauri", "Zeta Tauri"),
    ]

    taurus_proj: dict[str, tuple[str, float, float]] = {}
    for star_name in taurus_star_names:
        resolved = _resolve_named_star_icrs_deg(star_name)
        if resolved is None:
            continue
        star_ra_deg, star_dec_deg = resolved
        taurus_proj[star_name] = _project_to_hemisphere(
            np.deg2rad(star_ra_deg),
            np.deg2rad(star_dec_deg),
            rotation,
        )

    for a_name, b_name in taurus_edges:
        if a_name not in taurus_proj or b_name not in taurus_proj:
            continue
        a_panel, axx, ayy = taurus_proj[a_name]
        b_panel, bxx, byy = taurus_proj[b_name]
        if a_panel != b_panel:
            continue
        taur_ax = ax_l if a_panel == "north" else ax_r
        taur_ax.plot(
            [axx, bxx],
            [ayy, byy],
            color="#fca5a5",
            alpha=0.9,
            lw=1.05,
            zorder=8,
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
        #     fontsize=8.2,
        #     ha="left",
        #     va="center",
        #     zorder=10,
        # )

    # Southern Cross (Crux), pointer stars, Achernar, and Pleiades.
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
        resolved = _resolve_named_star_icrs_deg(star_name)
        if resolved is None:
            continue
        star_ra_deg, star_dec_deg = resolved
        south_proj[star_name] = _project_to_hemisphere(
            np.deg2rad(star_ra_deg),
            np.deg2rad(star_dec_deg),
            rotation,
        )

    for a_name, b_name in scx_edges:
        if a_name not in south_proj or b_name not in south_proj:
            continue
        a_panel, axx, ayy = south_proj[a_name]
        b_panel, bxx, byy = south_proj[b_name]
        if a_panel != b_panel:
            continue
        scx_ax = ax_l if a_panel == "north" else ax_r
        scx_ax.plot(
            [axx, bxx],
            [ayy, byy],
            color="#86efac",
            alpha=0.92,
            lw=1.05,
            zorder=8,
        )

    if "Alpha Centauri" in south_proj and "Beta Centauri" in south_proj:
        pa_panel, pax, pay = south_proj["Alpha Centauri"]
        pb_panel, pbx, pby = south_proj["Beta Centauri"]
        if pa_panel == pb_panel:
            ptr_ax = ax_l if pa_panel == "north" else ax_r
            ptr_ax.plot(
                [pax, pbx],
                [pay, pby],
                color="#fcd34d",
                alpha=0.95,
                lw=1.2,
                zorder=8,
            )

            if "Acrux" in south_proj and south_proj["Acrux"][0] == pa_panel:
                _, acx, acy = south_proj["Acrux"]
                midx = 0.5 * (pax + pbx)
                midy = 0.5 * (pay + pby)
                ptr_ax.plot(
                    [midx, acx],
                    [midy, acy],
                    color="#fbbf24",
                    alpha=0.75,
                    lw=0.95,
                    ls="--",
                    zorder=7,
                )

    marker_styles = {
        "Acrux": ("#bbf7d0", 20),
        "Mimosa": ("#bbf7d0", 18),
        "Gacrux": ("#bbf7d0", 18),
        "Imai": ("#bbf7d0", 16),
        "Epsilon Crucis": ("#bbf7d0", 14),
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

    for label_name, label_color in (("Achernar", "#a5f3fc"), ("Pleiades", "#c4b5fd"), ("Acrux", "#bbf7d0"), ("Antares", "#fca5a5")):
        if label_name not in south_proj:
            continue
        panel, lx, ly = south_proj[label_name]
        lbl_ax = ax_l if panel == "north" else ax_r
        y_offset = 0.018
        if label_name == "Achernar":
            y_offset = 0.030
        if label_name == "Acrux":
            y_offset = 0.040
        lbl_ax.text(
            lx + 0.03,
            ly + y_offset,
            label_name,
            color=label_color,
            fontsize=8.2,
            ha="left",
            va="center",
            zorder=10,
        )

    # Additional visible-night-sky stars from Astropy name resolution, plotted without labels.
    extra_visible_star_names = [
        "Sirius",
        "Canopus",
        "Rigil Kentaurus",
        "Hadar",
        "Achernar",
        "Acrux",
        "Mimosa",
        "Gacrux",
        "Altair",
        "Aldebaran",
        "Antares",
        "Spica",
        "Regulus",
        "Deneb",
        "Polaris",
        "Alphard",
        "Alkaid",
        "Dubhe",
        "Merak",
        "Phecda",
        "Megrez",
        "Alioth",
        "Mizar",
        "Kochab",
        "Pherkad",
        "Mirfak",
        "Algol",
        "Hamal",
        "Almach",
        "Schedar",
        "Caph",
        "Ruchbah",
        "Segin",
        "Alpheratz",
        "Mirach",
        "Markab",
        "Scheat",
        "Enif",
        "Ankaa",
        "Menkar",
        "Diphda",
        "Fomalhaut",
        "Deneb Algedi",
        "Sadalmelik",
        "Sadalsuud",
        "Skat",
        "Nashira",
        "Albali",
        "Alnair",
        "Peacock",
        "Atria",
        "Avior",
        "Miaplacidus",
        "Alsephina",
        "Suhail",
        "Wezen",
        "Adhara",
        "Mirzam",
        "Bellatrix",
        "Saiph",
        "Rigel",
        "Mintaka",
        "Alnilam",
        "Alnitak",
        "Meissa",
        "Procyon",
        "Gomeisa",
        "Castor",
        "Pollux",
        "Alhena",
        "Elnath",
        "Capella",
        "Menkalinan",
        "Alnath",
        "Rasalhague",
        "Cebalrai",
        "Sabik",
        "Kaus Australis",
        "Nunki",
        "Ascella",
        "Alnasl",
        "Arkab Prior",
        "Arkab Posterior",
        "Vega",
        "Sheliak",
        "Sulafat",
        "Rasalgethi",
        "Kornephoros",
        "Izar",
        "Nekkar",
        "Seginus",
        "Arcturus",
        "Muphrid",
        "Porrima",
        "Zubenelgenubi",
        "Zubeneschamali",
        "Denebola",
        "Algieba",
        "Ras Elased Australis",
        "Rasalas",
        "Algenubi",
        "Chertan",
        "Alphard",
        "Cor Caroli",
        "Menkent",
        "Alphirk",
        "Errai",
        "Alderamin",
        "Alfirk",
        "Sadr",
        "Albireo",
        "Gienah",
        "Aljanah",
        "Rukh",
        "Tarazed",
        "Alshain",
        "Deneb Kaitos",
        "Mira",
        "Rigel Kentaurus",
        "Canopus",
        "Spica",
        "Vega",
        "Deneb",
        "Altair",
    ]
    excluded_named_stars = set(orion_star_names + taurus_star_names + scx_star_names + pointer_names + extra_names + ["Betelgeuse"])
    seen_extra_stars: set[str] = set()
    for star_name in extra_visible_star_names:
        if star_name in seen_extra_stars:
            continue
        seen_extra_stars.add(star_name)
        if star_name in excluded_named_stars:
            continue
        resolved = _resolve_named_star_icrs_deg(star_name)
        if resolved is None:
            continue
        star_ra_deg, star_dec_deg = resolved
        panel, sx, sy = _project_to_hemisphere(
            np.deg2rad(star_ra_deg),
            np.deg2rad(star_dec_deg),
            rotation,
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

    ax_r.plot(
        [],
        [],
        marker="o",
        linestyle="None",
        markersize=7,
        markerfacecolor="black",
        markeredgecolor="#dc2626",
        markeredgewidth=1.3,
        label="Galactic Center",
    )
    ax_r.legend(
        loc="lower right",
        frameon=False,
        labelcolor="white",
        fontsize=8.5,
    )

    plt.savefig(
        fname,
        dpi=dpi,
        facecolor="black",
        edgecolor="none",
        pad_inches=0,
        transparent=False,
        bbox_inches=None,
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    plt.rcdefaults()

    print(f"Plotted {len(ra)} supernovae stars from CCSN class coordinates.")
    print(f"Saved: {fname}")
    print(f"Applied view rotation: +{rotation_deg:.0f} deg to both hemispheres.")
    print("Rendered filled blue contours (lighter = denser) for shared 95%, 75%, 50%, and 25% regions across both hemispheres.")
    print("Added latitude guide circles every 10 degrees in both hemispheres.")
    print("Placed North Pole and South Pole labels at the true pole positions.")
    if red_blob_mode == "density_peak":
        print("Added a red density blob centered on the posterior peak with matching contour style.")
    elif red_blob_mode == "true_center":
        print("Added a red density blob centered on the true sky location with matching contour style.")
    else:
        print("Added a red density blob centered on the selected sample location with matching contour style.")
    if betel_source == "astropy":
        print(
            f"Plotted Betelgeuse at RA={betelgeuse_ra_deg:.3f} deg, Dec={betelgeuse_dec_deg:.3f} deg "
            f"in the same +{rotation_deg:.0f} deg rotated reference frame ({betel_panel} hemisphere)."
        )
        print("Betelgeuse coordinates resolved with astropy SkyCoord.from_name('Betelgeuse').")
    else:
        print("Betelgeuse name resolution unavailable; skipped Betelgeuse marker.")
    print(
        f"Orion stick figure drew {len(orion_proj)} resolved stars and "
        f"{sum(1 for a_name, b_name in orion_edges if a_name in orion_proj and b_name in orion_proj and orion_proj[a_name][0] == orion_proj[b_name][0])} edges."
    )
    print(
        f"Taurus stick figure drew {len(taurus_proj)} resolved stars and "
        f"{sum(1 for a_name, b_name in taurus_edges if a_name in taurus_proj and b_name in taurus_proj and taurus_proj[a_name][0] == taurus_proj[b_name][0])} edges."
    )
    print(
        f"Southern-sky overlay resolved {len(south_proj)} named targets "
        "(Southern Cross, pointers, Achernar, Pleiades)."
    )
    print(
        f"Applied Astropy RA backshift of {ASTROPY_RA_BACKSHIFT_DEG:.0f} deg before the shared sky-map rotation; "
        "Milky Way RA handling is unchanged."
    )
    print("South hemisphere mirrored in x so the shared seam RA aligns across both panels.")
    if show_constellation_borders:
        print("Rendered all IAU constellation boundaries.")
        if show_important_constellation_labels:
            print("Annotated important constellations with labels for orientation.")

