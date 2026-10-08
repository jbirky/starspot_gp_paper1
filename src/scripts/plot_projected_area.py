"""
Visualize the projected area function A(beta) for a starspot.

Left panel:  Stellar disk with a spot of angular radius alpha_max
             centered at beta=0 (disk center), showing the spot size
             relative to the star.
Right panel: Normalized projected area A(beta) / (pi * sin^2(alpha))
             vs beta, showing the exact piecewise function (Eq. 8)
             and the small-spot approximation max{cos(beta), 0}.

Output:
    figures/fig_projected_area.pdf
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

label_fs = 18
tick_fs = 14
legend_fs = 14
title_fs = 20

# ── Projected area function (exact, Eq. 8) ──────────────────────

def projected_area_exact(beta, alpha):
    """Exact projected area A(beta, alpha) from Kipping (2012) Eq. 8.

    Returns A / pi (normalized by the stellar disk area).
    """
    beta = np.asarray(beta, dtype=float)
    alpha = float(alpha)
    A = np.zeros_like(beta)

    # Case 1: fully visible (0 < beta < pi/2 - alpha)
    case1 = beta < (np.pi / 2 - alpha)
    A[case1] = np.pi * np.sin(alpha)**2 * np.cos(beta[case1])

    # Case 2: partially visible (pi/2 - alpha < beta < pi/2 + alpha)
    case2 = (beta >= (np.pi / 2 - alpha)) & (beta < (np.pi / 2 + alpha))
    b2 = beta[case2]
    term1 = np.arccos(np.cos(alpha) / np.sin(b2))
    term2 = np.cos(b2) * np.sin(alpha)**2 * np.arccos(-1.0 / np.tan(alpha) / np.tan(b2))
    term3 = np.cos(alpha) * np.sin(b2) * np.sqrt(1 - np.cos(alpha)**2 / np.sin(b2)**2)
    A[case2] = term1 + term2 - term3

    # Case 3: hidden (beta > pi/2 + alpha)
    # A = 0 (already initialized)

    return A


# ── Parameters ───────────────────────────────────────────────────
alpha_values = [np.radians(5), np.radians(15), np.radians(30)]
alpha_labels = [r'$\alpha_{\max} = 5^\circ$',
                r'$\alpha_{\max} = 15^\circ$',
                r'$\alpha_{\max} = 30^\circ$']
alpha_colors = ['C0', 'C1', 'C2']

beta_arr = np.linspace(0, np.pi, 1000)

# ── Figure ───────────────────────────────────────────────────────
fig, (ax_disk, ax_area) = plt.subplots(1, 2, figsize=(14, 6),
                                        gridspec_kw={'width_ratios': [1, 1.3]})

# ── Left panel: stellar disk with spot at beta=0 ────────────────
th = np.linspace(0, 2 * np.pi, 500)
ax_disk.plot(np.cos(th), np.sin(th), 'k-', lw=2)
ax_disk.fill(np.cos(th), np.sin(th), color='#FFF4E0', zorder=0)

# Draw grid lines (latitude/longitude circles projected)
for lat_deg in np.arange(-75, 90, 15):
    lat = np.radians(lat_deg)
    lon = np.linspace(-np.pi / 2, np.pi / 2, 500)
    y_proj = np.cos(lat) * np.sin(lon)
    z_proj = np.sin(lat)
    ax_disk.plot(y_proj, np.full_like(y_proj, z_proj),
                 color='gray', lw=0.3, alpha=0.4)

for lon_deg in np.arange(-75, 90, 15):
    lon = np.radians(lon_deg)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 500)
    y_proj = np.cos(lat) * np.sin(lon)
    z_proj = np.sin(lat)
    ax_disk.plot(y_proj, z_proj, color='gray', lw=0.3, alpha=0.4)

# Draw spots at disk center (beta=0) for each alpha_max
for alpha, label, color in zip(alpha_values, alpha_labels, alpha_colors):
    spot_th = np.linspace(0, 2 * np.pi, 300)
    # At beta=0, the spot is at disk center; projected as a circle
    # with angular radius alpha (in radians) = sin(alpha) in projected coords
    r_spot = np.sin(alpha)
    ax_disk.plot(r_spot * np.cos(spot_th), r_spot * np.sin(spot_th),
                 color=color, lw=2.5, label=label, zorder=3)
    ax_disk.fill(r_spot * np.cos(spot_th), r_spot * np.sin(spot_th),
                 color=color, alpha=0.15, zorder=2)

ax_disk.set_xlim(-1.3, 1.3)
ax_disk.set_ylim(-1.3, 1.3)
ax_disk.set_aspect('equal')
ax_disk.set_title('Spot at disk center ($\\beta = 0$)', fontsize=title_fs)
ax_disk.legend(fontsize=legend_fs, loc='lower left',
               framealpha=0.9, edgecolor='#AAAAAA')
ax_disk.axis('off')

# ── Right panel: A(beta) vs beta ────────────────────────────────
for alpha, label, color in zip(alpha_values, alpha_labels, alpha_colors):
    A_exact = projected_area_exact(beta_arr, alpha)
    # Normalize by the maximum (= pi * sin^2(alpha) at beta=0)
    A_max = np.pi * np.sin(alpha)**2
    ax_area.plot(np.degrees(beta_arr), A_exact / A_max,
                 color=color, lw=2.5, label=label)

# Small-spot approximation: max{cos(beta), 0}
ax_area.plot(np.degrees(beta_arr), np.maximum(np.cos(beta_arr), 0),
             'k--', lw=1.5, alpha=0.6,
             label=r'Small-spot approx. $\max\{\cos\beta, 0\}$')

# Mark the transition regions
ax_area.axvline(90, color='gray', ls=':', lw=0.8, alpha=0.5)
ax_area.text(91, 0.5, r'$\beta = \pi/2$', fontsize=tick_fs,
             color='gray', va='center', rotation=90)

ax_area.set_xlabel(r'$\beta$ [degrees]', fontsize=label_fs)
ax_area.set_ylabel(r'$A(\beta)\, /\, A(0)$', fontsize=label_fs)
ax_area.set_title('Normalized projected area', fontsize=title_fs)
ax_area.legend(fontsize=legend_fs - 1, loc='upper right')
ax_area.set_xlim(0, 180)
ax_area.set_ylim(-0.05, 1.1)
ax_area.tick_params(labelsize=tick_fs)
ax_area.minorticks_on()

fig.tight_layout()
outpath = paths.figures / "fig_projected_area.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved {outpath}")
