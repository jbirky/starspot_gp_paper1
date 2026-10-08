"""
Visibility function Pi(t) = max{cos(beta(t)), 0} over three rotation periods.

Left panel:  vary spot latitude Phi at fixed edge-on inclination I = 90 deg.
Right panel: vary stellar inclination I at fixed latitude Phi = 60 deg.

Output: figures/fig_visibility.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = ['Computer Modern Roman', 'cmr10']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

label_fs = 20
title_fs = 18
tick_fs = 18
legend_fs = 16

# ── Parameters ──────────────────────────────────────────────────
P = 5.0                 # rotation period [days]
omega0 = 2 * np.pi / P


# ── Visibility function ─────────────────────────────────────────
def visibility(t, inc, Phi, Lambda0=0.0):
    """Pi(t) = max{cos(beta(t)), 0}."""
    cos_beta = (np.cos(inc) * np.sin(Phi)
                + np.sin(inc) * np.cos(Phi) * np.cos(Lambda0 + omega0 * t))
    return np.maximum(cos_beta, 0)


# ── Time array (3 rotation periods) ─────────────────────────────
t = np.linspace(0, 3 * P, 1000)

# ── Figure ──────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

# Left panel: vary latitude at fixed I = 90 deg
phi_vals = [20, 45, 70, 90]
colors_left = ['C0', 'C1', 'C2', 'C3']
for phi_deg, col in zip(phi_vals, colors_left):
    Pi = visibility(t, np.pi / 2, np.radians(phi_deg))
    ax1.plot(t / P, Pi, color=col, lw=1.3,
             label=rf'$\Phi = {phi_deg}^\circ$')

ax1.set_title(r'Vary latitude ($I = 90^\circ$)', fontsize=title_fs)
ax1.set_ylabel(r'$\mathcal{V}(t)$', fontsize=label_fs + 4)
ax1.set_xlim(min(t) / P, max(t) / P)
ax1.legend(fontsize=legend_fs)

# Right panel: vary inclination at fixed Phi = 60 deg
inc_vals = [0, 30, 60, 90]
colors_right = ['C0', 'C1', 'C2', 'C3']
for inc_deg, col in zip(inc_vals, colors_right):
    Pi = visibility(t, np.radians(inc_deg), np.radians(60))
    ax2.plot(t / P, Pi, color=col, lw=1.3,
             label=rf'$I = {inc_deg}^\circ$')

ax2.set_title(r'Vary inclination ($\Phi = 60^\circ$)', fontsize=title_fs)
ax2.legend(fontsize=legend_fs)
ax2.set_xlim(min(t) / P, max(t) / P)

# Common formatting
for ax in (ax1, ax2):
    ax.set_xlabel(r'$t\,/\,P$', fontsize=label_fs + 4)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.3)
    ax.tick_params(labelsize=tick_fs)
    ax.minorticks_on()

fig.tight_layout()

fig.savefig(paths.figures / "fig_visibility.pdf", bbox_inches="tight", dpi=300)