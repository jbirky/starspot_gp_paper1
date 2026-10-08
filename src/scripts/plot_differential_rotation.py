"""
Illustrate solar-like (kappa > 0) and anti-solar (kappa < 0)
differential rotation on a stellar sphere, showing velocity
arrows as a function of latitude.

The differential rotation law is:
    P(Phi) = P_eq / (1 - kappa * sin^2(Phi))

so the angular velocity is:
    omega(Phi) = omega_eq * (1 - kappa * sin^2(Phi))

Convention: x toward observer, y right on sky, z up (north pole).
Inclination I = 70 deg so the pole is tilted toward the viewer.
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

title_fontsize = 25
label_fontsize = 25

KAPPA_SOLAR = 1.0
KAPPA_ANTISOLAR = -1.0


# ── Geometry helpers (reused from spot_geometry.py) ─────────────

def latlon_to_xyz(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z


def apply_inclination(x, y, z, inc):
    a = inc - np.pi / 2
    c, s = np.cos(a), np.sin(a)
    return c * x - s * z, y, s * x + c * z


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


# ── Velocity arrows ─────────────────────────────────────────────

def draw_velocity_arrows(ax, inc, kappa, P_eq=3.0):
    """
    Draw arced arrows on the visible hemisphere showing the rotation
    frequency at each latitude. Arc angular extent is proportional to
    omega(Phi) = (2pi/P_eq) * (1 - kappa * sin^2(Phi)).
    """
    omega_eq = 2 * np.pi / P_eq

    lat_degs = np.arange(-75, 76, 15)

    # Compute max |omega| across latitudes for normalization
    omegas = omega_eq * (1 - kappa * np.sin(np.radians(lat_degs)) ** 2)
    omega_max = np.max(np.abs(omegas))

    # Max arc half-extent in longitude (radians) for the fastest latitude
    arc_max = np.radians(50)

    for lat_deg in lat_degs:
        lat = np.radians(lat_deg)
        omega_phi = omega_eq * (1 - kappa * np.sin(lat) ** 2)

        # Arc extent proportional to |omega|
        arc_extent = arc_max * abs(omega_phi) / omega_max
        if arc_extent < np.radians(3):
            continue

        # Arc centered at lon=0, from -arc_extent/2 to +arc_extent/2
        n_pts = 80
        sign = np.sign(omega_phi)
        lons = np.linspace(-sign * arc_extent / 2, sign * arc_extent / 2, n_pts)

        # Compute projected arc
        x, y, z = latlon_to_xyz(lat, lons)
        xr, yr, zr = apply_inclination(x, y, z, inc)

        # Only draw visible portion
        vis = xr > 0.02
        if not np.any(vis):
            continue

        yr_vis = np.where(vis, yr, np.nan)
        ax.plot(yr_vis, zr, color='#CC3300', lw=1.8, solid_capstyle='butt',
                zorder=5)

        # Arrowhead at the end of the arc
        vi = np.where(vis)[0]
        if len(vi) < 3:
            continue
        ei = vi[-1]
        dy = yr[ei] - yr[ei - 2]
        dz = zr[ei] - zr[ei - 2]
        ax.annotate('',
                    xy=(yr[ei] + dy * 0.5, zr[ei] + dz * 0.5),
                    xytext=(yr[ei], zr[ei]),
                    arrowprops=dict(arrowstyle='->', color='#CC3300',
                                    lw=1.8, shrinkA=0, shrinkB=0),
                    zorder=5)


# ── Draw panel ──────────────────────────────────────────────────

def draw_panel(ax, inc, kappa, title):
    # Disk
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), 'k-', lw=1.8)
    ax.fill(np.cos(th), np.sin(th), color='#FFF4E0', zorder=0)
    draw_grid(ax, inc)

    # Velocity arrows
    draw_velocity_arrows(ax, inc, kappa)

    # Rotation axis arrow
    # The pole direction projected: apply_inclination to (0,0,1)
    xp, yp, zp = apply_inclination(0, 0, 1, inc)
    pole_len = 0.35
    ax.annotate('', xy=(yp * (1 + pole_len), zp * (1 + pole_len)),
                xytext=(yp * 1.02, zp * 1.02),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
                zorder=1)
    ax.text(yp * (1 + pole_len) + 0.06, zp * (1 + pole_len) + 0.04,
            r'$\Omega$', fontsize=label_fontsize, color='gray',
            ha='left', va='center')

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=title_fontsize, pad=12)
    ax.axis('off')


# ── Figure ──────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

inc = np.radians(70)

draw_panel(axes[0], inc, kappa=KAPPA_SOLAR,
           title=r'Solar-like ($\kappa = %.1f$)' % KAPPA_SOLAR)
draw_panel(axes[1], inc, kappa=KAPPA_ANTISOLAR,
           title=r'Anti-solar ($\kappa = %.1f$)' % KAPPA_ANTISOLAR)

# Subtitle labels
axes[0].text(0.5, -0.02, 'Equator faster than poles',
             fontsize=label_fontsize - 2, color='#555555',
             ha='center', va='top', transform=axes[0].transAxes)
axes[1].text(0.5, -0.02, 'Poles faster than equator',
             fontsize=label_fontsize - 2, color='#555555',
             ha='center', va='top', transform=axes[1].transAxes)

fig.subplots_adjust(wspace=0.1)
fig.savefig(paths.figures / "differential_rotation.pdf",
            bbox_inches="tight", dpi=300)
