"""
Annotated diagram of the visibility function cos(beta(t)) = a0 + a1*cos(Lambda0 + omega0*t),
illustrating the parameters a0, a1, theta_vis, omega0, and Lambda0.

Output: figures/derivation/fig_visibility_annotated.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

# matplotlib.rcParams['font.family'] = 'serif'
# matplotlib.rcParams['font.serif'] = ['Computer Modern Roman', 'cmr10']
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True

label_fs = 20
tick_fs = 18
ann_fs = 18

# ── Parameters (chosen so a0 is clearly nonzero) ────────────────
I_deg = 70.0        # inclination [deg]
Phi_deg = 50.0      # spot latitude [deg]
Lambda0_deg = 50.0  # initial longitude [deg]

I = np.radians(I_deg)
Phi = np.radians(Phi_deg)
Lambda0 = np.radians(Lambda0_deg)

a0 = np.cos(I) * np.sin(Phi)
a1 = np.sin(I) * np.cos(Phi)
theta_vis = np.arccos(np.clip(-a0 / a1, -1, 1))

P = 5.0
omega0 = 2 * np.pi / P

# ── Plot in terms of phase theta = omega0*t ─────────────────────
theta = np.linspace(-0.5, 3 * np.pi, 2000)
cos_beta = a0 + a1 * np.cos(Lambda0 + theta)
Pi_func = np.maximum(cos_beta, 0)

# Peak location (where cos(Lambda0 + theta) = 1, i.e. theta = -Lambda0 + 2*pi*k)
theta_peak = -Lambda0 + 2 * np.pi
peak_val = a0 + a1
trough_val = a0 - a1

# Visibility window edges
center = theta_peak
left_edge = center - theta_vis
right_edge = center + theta_vis

fig, ax = plt.subplots(figsize=(10, 5))

# Shade visibility windows
ax.fill_between(theta, 0, Pi_func, where=(Pi_func > 0),
                color='C0', alpha=0.12)

# cos(beta) curve
ax.plot(theta, cos_beta, 'C0', lw=2.0, label=r'$\cos\beta(\theta)$')

# Pi(t) thick on top where visible
ax.plot(theta, Pi_func, 'C0', lw=2.8, alpha=0.4,
        label=r'$\mathcal{V}(\theta) = \max\{\cos\beta,\;0\}$')

# Zero line
ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.4)

# ── a0: dashed line at the DC offset ───────────────────────────
ax.axhline(a0, color='C3', ls='--', lw=1.2, alpha=0.8, zorder=1)
# Vertical arrow from 0 to a0 on the right side
x_a0 = 2.85 * np.pi
ax.annotate('', xy=(x_a0, a0), xytext=(x_a0, 0),
            arrowprops=dict(arrowstyle='<->', color='C3', lw=1.5,
                            shrinkA=0, shrinkB=0))
ax.text(x_a0 + 0.12, a0 / 2, r'$b_0$',
        fontsize=ann_fs + 3, color='C3', va='center', ha='left',
        fontweight='bold')

# ── a1: amplitude from a0 to peak ──────────────────────────────
x_a1 = theta_peak + 0.25
ax.annotate('', xy=(x_a1, peak_val), xytext=(x_a1, a0),
            arrowprops=dict(arrowstyle='<->', color='C2', lw=1.5,
                            shrinkA=0, shrinkB=0))
ax.text(x_a1 + 0.12, (a0 + peak_val) / 2, r'$b_1$',
        fontsize=ann_fs + 3, color='C2', va='center', ha='left',
        fontweight='bold')

# ── theta_vis: half-width of visibility window ──────────────────
y_tv = -0.12
ax.annotate('', xy=(center, y_tv), xytext=(right_edge, y_tv),
            arrowprops=dict(arrowstyle='<->', color='C1', lw=1.5,
                            shrinkA=0, shrinkB=0))
ax.text((center + right_edge) / 2, y_tv - 0.07,
        r'$\theta_\mathrm{vis}$',
        fontsize=ann_fs + 3, color='C1', ha='center', va='top',
        fontweight='bold')

# Mark zero-crossing dots
for edge in [left_edge, right_edge]:
    ax.plot(edge, 0, 'o', color='C1', ms=6, zorder=5)

# Thin vertical guide at center
ax.axvline(center, color='gray', ls=':', lw=0.6, alpha=0.4)

# ── omega0 / period: bracket spanning one full cycle ────────────
y_per = trough_val - 0.12
p_start = theta_peak - np.pi
p_end = theta_peak + np.pi
ax.annotate('', xy=(p_start, y_per), xytext=(p_end, y_per),
            arrowprops=dict(arrowstyle='<->', color='C4', lw=1.5,
                            shrinkA=0, shrinkB=0))
ax.text((p_start + p_end) / 2, y_per - 0.06,
        r'$2\pi/\omega_0 = P$',
        fontsize=ann_fs, color='C4', ha='center', va='top')

# ── Lambda0: phase offset from 2*pi to the peak ────────────────
y_lam = peak_val + 0.07
ref = 2 * np.pi
ax.annotate('', xy=(theta_peak, y_lam), xytext=(ref, y_lam),
            arrowprops=dict(arrowstyle='<->', color='C5', lw=1.5,
                            shrinkA=0, shrinkB=0))
ax.text((ref + theta_peak) / 2, y_lam + 0.05,
        r'$\Lambda_0$',
        fontsize=ann_fs + 3, color='C5', ha='center', va='bottom',
        fontweight='bold')
# Guide lines at reference and peak
ax.plot([ref, ref], [peak_val - 0.02, y_lam], color='C5',
        ls=':', lw=0.7, alpha=0.5)
ax.plot([theta_peak, theta_peak], [peak_val, y_lam], color='C5',
        ls=':', lw=0.7, alpha=0.5)

# ── Formatting ──────────────────────────────────────────────────
ax.set_xlabel(r'$\theta = \omega_0 t$', fontsize=label_fs + 2)
ax.set_ylabel(r'$\cos\beta(t)$', fontsize=label_fs + 2)
ax.set_xlim(-0.3, 3 * np.pi)
ax.set_ylim(trough_val - 0.35, peak_val + 0.25)

ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi,
               5 * np.pi / 2, 3 * np.pi])
ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$',
                     r'$2\pi$', r'$5\pi/2$', r'$3\pi$'])
ax.tick_params(labelsize=tick_fs)
ax.minorticks_on()
ax.legend(fontsize=tick_fs + 1, loc='upper left')

fig.tight_layout()

fig.savefig(paths.figures / "fig_visibility_annotated.pdf", bbox_inches="tight", dpi=300)
