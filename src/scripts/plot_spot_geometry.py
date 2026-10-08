"""
Generate a 3-panel figure illustrating the three cases of starspot
projected area from Kipping (2012):
  1. Fully visible  (beta < pi/2 - alpha)
  2. Partially visible (pi/2 - alpha < beta < pi/2 + alpha)
  3. Hidden (beta > pi/2 + alpha)

Convention:
  x toward observer, y right on sky, z up (north pole).
  Inclination I=90 deg = equator-on; I=0 deg = pole-on.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

title_fontsize = 22
label_fontsize = 18


def latlon_to_xyz(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def apply_inclination(x, y, z, inc):
    """Rotate around y by (inc - pi/2) so that at I=90 the equator
    faces the observer and the pole points up."""
    a = inc - np.pi / 2
    c, s = np.cos(a), np.sin(a)
    return c * x - s * z, y, s * x + c * z


def spot_boundary(lat0, lon0, alpha, inc, n=500):
    """Compute projected spot boundary."""
    t = np.linspace(0, 2 * np.pi, n)
    cx = np.cos(alpha) * np.ones_like(t)
    cy = np.sin(alpha) * np.cos(t)
    cz = np.sin(alpha) * np.sin(t)
    # R_y(+lat0) then R_z(lon0)
    cla, sla = np.cos(lat0), np.sin(lat0)
    x1 = cla * cx - sla * cz
    y1 = cy
    z1 = sla * cx + cla * cz
    clo, slo = np.cos(lon0), np.sin(lon0)
    x2 = clo * x1 - slo * y1
    y2 = slo * x1 + clo * y1
    z2 = z1
    xr, yr, zr = apply_inclination(x2, y2, z2, inc)
    return yr, zr, xr


def compute_centre(lat0, lon0, inc):
    xc, yc, zc = latlon_to_xyz(lat0, lon0)
    xr, yr, zr = apply_inclination(xc, yc, zc, inc)
    beta = np.arccos(np.clip(xr, -1, 1))
    return yr, zr, xr, beta


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_grid(ax, inc):
    n = 600
    for lat_deg in np.arange(-75, 90, 15):
        lat = np.radians(lat_deg)
        lon = np.linspace(-np.pi, np.pi, n)
        x, y, z = latlon_to_xyz(lat, lon)
        xr, yr, zr = apply_inclination(x, y, z, inc)
        yr = np.where(xr > 0, yr, np.nan)
        ax.plot(yr, zr, color='gray', lw=0.3, alpha=0.5)
    for lon_deg in np.arange(0, 180, 15):
        lon = np.radians(lon_deg)
        lat = np.linspace(-np.pi / 2, np.pi / 2, n)
        x, y, z = latlon_to_xyz(lat, lon)
        xr, yr, zr = apply_inclination(x, y, z, inc)
        yr = np.where(xr > 0, yr, np.nan)
        ax.plot(yr, zr, color='gray', lw=0.3, alpha=0.5)
    # Equator
    lon = np.linspace(-np.pi, np.pi, n)
    x, y, z = latlon_to_xyz(0, lon)
    xr, yr, zr = apply_inclination(x, y, z, inc)
    yr = np.where(xr > 0, yr, np.nan)
    ax.plot(yr, zr, color='gray', lw=0.7, alpha=0.7)


def fill_spot(ax, yp, zp, xr):
    vis = xr > 0
    if not np.any(vis):
        return
    vy, vz = yp[vis], zp[vis]
    if np.all(vis):
        ax.fill(vy, vz, color='#8B0000', alpha=0.55, zorder=2)
        ax.plot(yp, zp, color='darkred', lw=1.5, zorder=3)
        return
    ang = np.arctan2(vz, vy)
    order = np.argsort(ang)
    vy, vz, ang = vy[order], vz[order], ang[order]
    limb = np.linspace(ang[-1], ang[0], 80)
    poly_y = np.concatenate([vy, np.cos(limb)])
    poly_z = np.concatenate([vz, np.sin(limb)])
    ax.fill(poly_y, poly_z, color='#8B0000', alpha=0.55, zorder=2)
    yp2 = np.where(xr > 0, yp, np.nan)
    ax.plot(yp2, zp, color='darkred', lw=1.5, zorder=3)


def draw_coord_arrows(ax, inc):
    """Draw curved arrows on the sphere showing the Phi (latitude) and
    Lambda (longitude) coordinate directions."""
    # Place in the lower-left of the visible disk, away from the spot
    ref_lat = np.radians(-30)
    ref_lon = np.radians(-30)
    arc_len = np.radians(30)  # angular length of each arrow

    # --- Latitude arrow (Phi): along a meridian at ref_lon ---
    lats = np.linspace(ref_lat, ref_lat + arc_len, 80)
    x, y, z = latlon_to_xyz(lats, np.full_like(lats, ref_lon))
    xr, yr, zr = apply_inclination(x, y, z, inc)
    mask = xr > 0
    yr_vis = np.where(mask, yr, np.nan)
    ax.plot(yr_vis, zr, color='#2ca02c', lw=2.2, zorder=5, solid_capstyle='butt')
    # Arrowhead
    vi = np.where(mask)[0]
    if len(vi) > 3:
        ei = vi[-1]
        dy = yr[ei] - yr[ei - 2]
        dz = zr[ei] - zr[ei - 2]
        ax.annotate('', xy=(yr[ei] + dy * 0.5, zr[ei] + dz * 0.5),
                    xytext=(yr[ei], zr[ei]),
                    arrowprops=dict(arrowstyle='->', color='#2ca02c',
                                    lw=2.2, shrinkA=0, shrinkB=0),
                    zorder=5)
        ax.text(yr[ei] + 0.09, zr[ei] + 0.04,
                r'$\Phi$', fontsize=label_fontsize, color='#2ca02c',
                ha='left', va='center', zorder=5)

    # --- Longitude arrow (Lambda): along a latitude circle at ref_lat ---
    lons = np.linspace(ref_lon, ref_lon + arc_len, 80)
    x, y, z = latlon_to_xyz(np.full_like(lons, ref_lat), lons)
    xr, yr, zr = apply_inclination(x, y, z, inc)
    mask = xr > 0
    yr_vis = np.where(mask, yr, np.nan)
    ax.plot(yr_vis, zr, color='#9467bd', lw=2.2, zorder=5, solid_capstyle='butt')
    vi = np.where(mask)[0]
    if len(vi) > 3:
        ei = vi[-1]
        dy = yr[ei] - yr[ei - 2]
        dz = zr[ei] - zr[ei - 2]
        ax.annotate('', xy=(yr[ei] + dy * 0.5, zr[ei] + dz * 0.5),
                    xytext=(yr[ei], zr[ei]),
                    arrowprops=dict(arrowstyle='->', color='#9467bd',
                                    lw=2.2, shrinkA=0, shrinkB=0),
                    zorder=5)
        ax.text(yr[ei] + 0.06, zr[ei] - 0.09,
                r'$\Lambda$', fontsize=label_fontsize, color='#9467bd',
                ha='left', va='top', zorder=5)


def draw_panel(ax, lat0, lon0, alpha_k, inc, title, case_label,
               show_coords=False):
    # Disk
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.8)
    ax.fill(np.cos(th), np.sin(th), color='#FFF4E0', zorder=0)
    draw_grid(ax, inc)

    # Spot boundary and centre
    yp, zp, xr = spot_boundary(lat0, lon0, alpha_k, inc)
    ypc, zpc, xc_r, beta_k = compute_centre(lat0, lon0, inc)
    vis = xr > 0
    spot_on_disk = xc_r > 0

    # Fill spot
    fill_spot(ax, yp, zp, xr)

    # --- Annotations ---
    d = np.sqrt(ypc**2 + zpc**2)
    phi = np.arctan2(zpc, ypc)
    perp_y, perp_z = -np.sin(phi), np.cos(phi)

    # beta_k: dashed line from disk centre to spot centre (or to limb)
    line_end = min(d, 0.92) if spot_on_disk else 0.92
    ax.plot([0, line_end * np.cos(phi)],
            [0, line_end * np.sin(phi)],
            color='#0066CC', lw=1.2, ls='--', alpha=0.5, zorder=4)

    # beta_k label
    mid_r = line_end * 0.4
    ax.text(mid_r * np.cos(phi) + 0.11 * perp_y,
            mid_r * np.sin(phi) + 0.11 * perp_z,
            r'$\beta_k$', fontsize=label_fontsize, color='#0066CC',
            ha='center', va='center', zorder=5)

    # Spot centre marker
    if spot_on_disk:
        ax.plot(ypc, zpc, 'o', color='white', ms=6, mec='darkred',
                mew=1.5, zorder=7)

    # Hidden annotation
    if not spot_on_disk and not np.any(vis):
        ax.text(1.22 * np.cos(phi) - .2, 1.22 * np.sin(phi) - 0.5,
                '(spot behind\nstar)', fontsize=label_fontsize-2, color='darkred',
                alpha=0.6, ha='center', va='top',
                fontstyle='italic', zorder=5)

    # Coordinate system arrows (only on first panel)
    if show_coords:
        draw_coord_arrows(ax, inc)

    # Rotation axis
    ax.annotate('', xy=(0, 1.35), xytext=(0, 1.02),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
                zorder=1)
    ax.text(0.06, 1.4, r'$\Omega$', fontsize=label_fontsize, color='gray',
            ha='left', va='center')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.text(0.5, -0.04, case_label, fontsize=label_fontsize, color='#555555',
            ha='center', va='top', transform=ax.transAxes)
    ax.axis('off')


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(18, 5.5))
# 3 panels for the star + 1 narrow panel on the right for the legend
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.55], wspace=0.08)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_legend = fig.add_subplot(gs[0, 3])

inc = np.radians(90)
alpha = np.radians(25)

# beta values at I=90: beta = arccos(cos(Phi)*cos(Lambda))
# alpha = 25 deg => pi/2 - alpha = 65 deg, pi/2 + alpha = 115 deg
# Case 1: Phi=15, Lambda=45 => beta ~ 47 deg < 65
# Case 2: Phi=0,  Lambda=78 => beta = 78 deg, 65 < 78 < 115
# Case 3: Phi=-10, Lambda=150 => beta ~ 148 deg > 115

draw_panel(axes[0],
           lat0=np.radians(15), lon0=np.radians(45),
           alpha_k=alpha, inc=inc,
           title='Case 1: Fully Visible',
           case_label=r'$0 < \beta_k < \pi/2 - \alpha_k$',
           show_coords=True)

draw_panel(axes[1],
           lat0=np.radians(0), lon0=np.radians(78),
           alpha_k=alpha, inc=inc,
           title='Case 2: Partially Visible',
           case_label=r'$\pi/2 - \alpha_k < \beta_k < \pi/2 + \alpha_k$')

draw_panel(axes[2],
           lat0=np.radians(-10), lon0=np.radians(150),
           alpha_k=alpha, inc=inc,
           title='Case 3: Hidden',
           case_label=r'$\pi/2 + \alpha_k < \beta_k < \pi$')

# --- Legend panel: symbol definitions ---
ax_legend.axis('off')
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)

legend_entries = [
    ('gray',    r'$\Omega$',    ' = stellar rotation axis'),
    ('#2ca02c', r'$\Phi$',      ' = latitude'),
    ('#9467bd', r'$\Lambda$',   ' = longitude'),
    ('#0066CC', r'$\beta_k$',   ' = angle between spot normal\nand line of sight'),
    ('#8B0000', r'$\alpha_k$',  ' = angular radius of spot'),
]

# Draw box around the legend area
import matplotlib.patches as mpatches
box = mpatches.FancyBboxPatch(
    (-0.05, 0.05), 1.6, 0.85,
    boxstyle='round,pad=0.03', facecolor='white', edgecolor='#AAAAAA',
    lw=1.2, transform=ax_legend.transAxes, zorder=0,
    clip_on=False)
ax_legend.add_patch(box)

legend_fontsize = 14
y0 = 0.82
dy = 0.16
for i, (color, symbol, desc) in enumerate(legend_entries):
    y = y0 - i * dy
    ax_legend.text(0.05, y, symbol, fontsize=label_fontsize, color=color,
                   ha='left', va='top', transform=ax_legend.transAxes,
                   fontweight='bold')
    ax_legend.text(0.25, y - 0.005, desc, fontsize=legend_fontsize,
                   color='#333333',
                   ha='left', va='top', transform=ax_legend.transAxes)

fig.savefig(paths.figures / "spot_geometry.pdf", bbox_inches="tight", dpi=300)
