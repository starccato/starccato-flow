"""Sky plotting utilities."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

try:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


ASTROPY_RA_BACKSHIFT_DEG = 60.0


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


def _constellation_border_points(
    rotation_rad: float,
    n_ra: int = 720,
    n_dec: int = 360,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return projected north/south points that trace constellation borders."""
    if not _ASTROPY_AVAILABLE:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Sample the sky, classify each point by constellation, then keep only cell edges
    # where neighboring constellation IDs differ.
    ra_deg = np.linspace(0.0, 360.0, n_ra, endpoint=False)
    dec_deg = np.linspace(-89.5, 89.5, n_dec)
    ra_mesh, dec_mesh = np.meshgrid(ra_deg, dec_deg)

    sky = SkyCoord(ra=ra_mesh.ravel() * u.deg, dec=dec_mesh.ravel() * u.deg, frame="icrs")
    const_names = np.asarray(sky.get_constellation(short_name=True)).reshape(dec_mesh.shape)

    _, inv = np.unique(const_names, return_inverse=True)
    const_id = inv.reshape(const_names.shape)

    dh = const_id[:, 1:] != const_id[:, :-1]
    dv = const_id[1:, :] != const_id[:-1, :]

    ra_mid_h = 0.5 * (ra_deg[1:] + ra_deg[:-1])
    dec_mid_v = 0.5 * (dec_deg[1:] + dec_deg[:-1])

    ra_h = np.broadcast_to(ra_mid_h, dh.shape)[dh]
    dec_h = np.broadcast_to(dec_deg[:, None], dh.shape)[dh]
    ra_v = np.broadcast_to(ra_deg, dv.shape)[dv]
    dec_v = np.broadcast_to(dec_mid_v[:, None], dv.shape)[dv]

    ra_b = np.concatenate([ra_h, ra_v])
    dec_b = np.concatenate([dec_h, dec_v])

    ra_b = _backshift_astropy_ra_deg(ra_b)
    ra_b_rad = np.deg2rad(ra_b)
    dec_b_rad = np.deg2rad(dec_b)
    ra_use = np.mod(ra_b_rad + rotation_rad, 2 * np.pi)

    north = dec_b_rad >= 0.0
    south = ~north

    rr_n = (np.pi / 2 - dec_b_rad[north]) / (np.pi / 2)
    x_n = rr_n * np.sin(ra_use[north])
    y_n = rr_n * np.cos(ra_use[north])

    rr_s = (np.pi / 2 + dec_b_rad[south]) / (np.pi / 2)
    x_s = -rr_s * np.sin(ra_use[south])
    y_s = rr_s * np.cos(ra_use[south])

    return x_n, y_n, x_s, y_s


def plot_galactic_supernovae_polar_hemispheres(
    ccsn,
    fname: str = "plots/galactic_supernovae_polar_hemispheres.png",
    rotation_deg: float = 60.0,
    show_constellation_borders: bool = False,
    show: bool = True,
    dpi: int = 160,
) -> None:
    """Plot CCSN sky distribution as tangent north/south pole-centered hemispheres.

    Args:
        ccsn: Supernovae-like object exposing ``ra``, ``dec`` and
            ``get_galactic_center_direction()``.
        fname: Output image path.
        rotation_deg: Global RA view rotation in degrees.
        show_constellation_borders: If True, overlay IAU constellation boundaries.
        show: If True, call ``plt.show()``.
        dpi: Image save DPI.
    """
    ra = np.mod(np.asarray(ccsn.ra), 2 * np.pi)
    dec = np.asarray(ccsn.dec)

    rotation = np.deg2rad(rotation_deg)
    # Milky Way stars and Astropy-resolved objects use the same RA rotation.
    ra_rot = _rotate_ra(ra, rotation)

    fig = plt.figure(figsize=(12, 6.8), facecolor="black")
    ax_l = fig.add_axes([0.00, 0.00, 0.50, 1.00], facecolor="black")
    ax_r = fig.add_axes([0.50, 0.00, 0.50, 1.00], facecolor="black")

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

    blue_bases = ["#1d4ed8", "#3b82f6", "#60a5fa", "#bfdbfe"]
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

    if show_constellation_borders:
        if _ASTROPY_AVAILABLE:
            x_n, y_n, x_s, y_s = _constellation_border_points(rotation)
            ax_l.scatter(x_n, y_n, s=0.55, c="#f8fafc", alpha=0.62, linewidths=0, zorder=4)
            ax_r.scatter(x_s, y_s, s=0.55, c="#f8fafc", alpha=0.62, linewidths=0, zorder=4)
        else:
            print("Constellation borders requested, but astropy is not installed in this environment.")

    for ax in (ax_l, ax_r):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    gc_ra, gc_dec = ccsn.get_galactic_center_direction()
    gc_panel, gc_x, gc_y = _project_to_hemisphere(gc_ra, gc_dec, rotation)

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
            betel_x + 0.035,
            betel_y + 0.02,
            "Betelgeuse",
            color="#fde68a",
            fontsize=8.5,
            ha="left",
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
        if a_panel != b_panel:
            continue
        orion_ax = ax_l if a_panel == "north" else ax_r
        orion_ax.plot(
            [axx, bxx],
            [ayy, byy],
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
            s=14,
            c="#f8fafc",
            edgecolors="none",
            alpha=0.95,
            zorder=9,
        )

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
    extra_names = ["Achernar", "Pleiades"]

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

    for label_name, label_color in (("Achernar", "#a5f3fc"), ("Pleiades", "#c4b5fd"), ("Acrux", "#bbf7d0")):
        if label_name not in south_proj:
            continue
        panel, lx, ly = south_proj[label_name]
        lbl_ax = ax_l if panel == "north" else ax_r
        lbl_ax.text(
            lx + 0.03,
            ly + 0.018,
            label_name,
            color=label_color,
            fontsize=8.2,
            ha="left",
            va="center",
            zorder=10,
        )

    ax_r.plot([], [], color=fill_colors[0], lw=6, label="Blue 100% contour")
    ax_r.plot([], [], color=fill_colors[1], lw=6, label="Blue 75% contour")
    ax_r.plot([], [], color=fill_colors[2], lw=6, label="Blue 50% contour")
    ax_r.plot([], [], color=fill_colors[3], lw=6, label="Blue 25% contour")
    ax_r.plot([], [], color=red_fill_colors[2], lw=6, label="Red GC blob")
    # ax_r.legend(loc="lower right", facecolor="black", edgecolor="white", labelcolor="white", fontsize=8.5)

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

    print(f"Plotted {len(ra)} supernovae stars from CCSN class coordinates.")
    print(f"Saved: {fname}")
    print(f"Applied view rotation: +{rotation_deg:.0f} deg to both hemispheres.")
    print("Rendered filled blue contours (lighter = denser) for shared 95%, 75%, 50%, and 25% regions across both hemispheres.")
    print("Added latitude guide circles every 10 degrees in both hemispheres.")
    print("Placed North Pole and South Pole labels at the true pole positions.")
    print("Added a red density blob centered on the Galactic Center with matching contour style.")
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
        f"Southern-sky overlay resolved {len(south_proj)} named targets "
        "(Southern Cross, pointers, Achernar, Pleiades)."
    )
    print(
        f"Applied Astropy RA backshift of {ASTROPY_RA_BACKSHIFT_DEG:.0f} deg before the shared sky-map rotation; "
        "Milky Way RA handling is unchanged."
    )
    print("South hemisphere mirrored in x so the shared seam RA aligns across both panels.")

