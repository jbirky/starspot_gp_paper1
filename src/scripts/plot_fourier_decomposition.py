"""
Fourier decomposition of the single-spot signal A(t) = Gamma(t) * Pi(t).

Left panel:  A(t) in the time domain for several (I, Phi) configurations.
Right panel: First ~5 Fourier coefficients |c_n|^2 for each configuration.

Corresponds to Eq. (A_hat) in Section 3.5.

Output: figures/fig_fourier_decomposition.pdf
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

label_fs = 20
title_fs = 18
tick_fs = 18
legend_fs = 14

# ── Parameters ──────────────────────────────────────────────────
P = 5.0                  # rotation period [days]
omega0 = 2 * np.pi / P
alpha_max = 0.1          # peak spot angular radius [rad]
lspot = 40.0             # plateau duration [days]
tau_spot = 10.0          # rise/decay timescale [days]

N_harmonics = 5          # number of harmonics to show


# ── Trapezoidal envelope alpha(t) ──────────────────────────────
def trapezoidal_alpha(t, lspot, tau_spot, alpha_max):
    """Symmetric trapezoid centered at t=0."""
    half_plat = lspot / 2.0
    abst = np.abs(t)
    alpha = np.where(
        abst <= half_plat,
        alpha_max,
        np.where(
            abst <= half_plat + tau_spot,
            alpha_max * (1.0 - (abst - half_plat) / tau_spot),
            0.0,
        ),
    )
    return alpha


# ── Normalized squared envelope Gamma(t) = alpha^2(t) / alpha_max^2 ──
def Gamma(t, lspot, tau_spot, alpha_max):
    return (trapezoidal_alpha(t, lspot, tau_spot, alpha_max) / alpha_max) ** 2


# ── Visibility function Pi(t) = max{cos(beta), 0} ─────────────
def visibility(t, inc, Phi):
    a0 = np.cos(inc) * np.sin(Phi)
    a1 = np.sin(inc) * np.cos(Phi)
    cos_beta = a0 + a1 * np.cos(omega0 * t)
    return np.maximum(cos_beta, 0.0)


# ── Single-spot signal A(t) = Gamma(t) * Pi(t) ────────────────
def A_signal(t, inc, Phi, lspot, tau_spot, alpha_max):
    return Gamma(t, lspot, tau_spot, alpha_max) * visibility(t, inc, Phi)


# ── Fourier coefficients c_n (general inclination, Eq. cn_general) ──
def cn_coefficients(inc, Phi, n_max):
    """Compute c_n for n = 0, 1, ..., n_max."""
    a0 = np.cos(inc) * np.sin(Phi)
    a1 = np.sin(inc) * np.cos(Phi)

    # Visibility angle
    ratio = -a0 / a1
    if np.abs(ratio) >= 1.0:
        if ratio <= -1.0:
            theta_vis = np.pi  # always visible
        else:
            theta_vis = 0.0    # never visible
    else:
        theta_vis = np.arccos(ratio)

    if theta_vis == 0.0:
        return np.zeros(n_max + 1)

    cn = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        if n == 0:
            cn[n] = (a0 * theta_vis + a1 * np.sin(theta_vis)) / np.pi
        elif n == 1:
            cn[n] = (a0 * np.sin(theta_vis)
                      + a1 * (theta_vis + np.sin(theta_vis) * np.cos(theta_vis)) / 2
                      ) / np.pi
        else:
            term1 = a0 * np.sin(n * theta_vis) / n
            nm1 = n - 1
            np1 = n + 1
            term2 = a1 / 2 * (np.sin(nm1 * theta_vis) / nm1
                               + np.sin(np1 * theta_vis) / np1)
            cn[n] = (term1 + term2) / np.pi

    return cn


# ── Configurations to plot ─────────────────────────────────────
# Span different visibility regimes: edge-on equatorial (half-wave),
# tilted with moderate latitude (asymmetric duty cycle),
# and nearly pole-on with high-latitude spot (always visible, large c_0).
phi_fixed = 30  # fixed latitude for all configurations
configs = [
    (90, phi_fixed, rf'$I=90^\circ,\;\Phi={phi_fixed}^\circ$'),
    (60, phi_fixed, rf'$I=60^\circ,\;\Phi={phi_fixed}^\circ$'),
    (30, phi_fixed, rf'$I=30^\circ,\;\Phi={phi_fixed}^\circ$'),
]

colors = ['C0', 'C1', 'C2']

# ── Time array ─────────────────────────────────────────────────
t = np.linspace(-0.5 * (lspot + 2 * tau_spot), 0.5 * (lspot + 2 * tau_spot), 4000)

# ── Figure ─────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ns = np.arange(N_harmonics + 1)  # 0, 1, 2, 3, 4, 5
bar_width = 0.22
offsets = np.arange(len(configs)) - (len(configs) - 1) / 2

for i, (inc_deg, phi_deg, label) in enumerate(configs):
    inc = np.radians(inc_deg)
    Phi = np.radians(phi_deg)

    # Left panel: A(t)
    At = A_signal(t, inc, Phi, lspot, tau_spot, alpha_max)
    ax1.plot(t, At, color=colors[i], lw=1.5, label=label)

    # Right panel: |c_n|^2  — normalize so sum |c_n|^2 over plotted n = 1
    cn = cn_coefficients(inc, Phi, N_harmonics)
    cn_sq = cn ** 2
    total = cn_sq[0] + 2 * np.sum(cn_sq[1:])
    if total > 0:
        cn_sq /= total
    ax2.bar(ns + offsets[i] * bar_width, cn_sq, width=bar_width,
            color=colors[i], alpha=0.85, label=label, edgecolor='k', linewidth=0.3)

# Dashed envelope on left panel
env = Gamma(t, lspot, tau_spot, alpha_max)
ax1.plot(t, env, color='gray', lw=1.2, ls='--', alpha=0.6, label=r'$\Gamma(t)$')

# ── Left panel formatting ──────────────────────────────────────
ax1.set_xlabel(r'Time [days]', fontsize=label_fs)
ax1.set_ylabel(r'$A(t)$', fontsize=label_fs + 2)
ax1.set_title(r'Single-spot signal', fontsize=title_fs)
ax1.legend(fontsize=legend_fs, loc='lower left')
ax1.tick_params(labelsize=tick_fs)
ax1.minorticks_on()
ax1.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.3)

# ── Right panel formatting ─────────────────────────────────────
ax2.set_xlabel(r'Harmonic $n$', fontsize=label_fs)
ax2.set_ylabel(r'$|c_n|^2\;/\;\sum |c_n|^2$', fontsize=label_fs + 2)
ax2.set_title(r'Fourier coefficients (normalized)', fontsize=title_fs)
ax2.set_xticks(ns)
ax2.set_xticklabels([rf'${n}$' for n in ns])
ax2.legend(fontsize=legend_fs, loc='upper right')
ax2.tick_params(labelsize=tick_fs)
ax2.minorticks_on()

fig.tight_layout()
fig.savefig(paths.figures / "fig_fourier_decomposition.pdf",
            bbox_inches="tight", dpi=300)

# ── Second figure: harmonic decomposition for each case ───────
fig2, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

harmonic_colors = ['C3', 'C4', 'C5']
harmonic_labels = [r'$n=0$: $c_0\,\Gamma(t)$',
                   r'$n=1$: $2c_1\,\Gamma(t)\cos(\omega_0 t)$',
                   r'$n=2$: $2c_2\,\Gamma(t)\cos(2\omega_0 t)$']

for i, (inc_deg, phi_deg, label) in enumerate(configs):
    ax = axes[i]
    inc = np.radians(inc_deg)
    Phi = np.radians(phi_deg)

    cn = cn_coefficients(inc, Phi, N_harmonics)
    env = Gamma(t, lspot, tau_spot, alpha_max)

    # Harmonic components
    comp0 = cn[0] * env
    comp1 = 2 * cn[1] * env * np.cos(omega0 * t)
    comp2 = 2 * cn[2] * env * np.cos(2 * omega0 * t)
    reconstructed = comp0 + comp1 + comp2

    # Full signal for reference
    At = A_signal(t, inc, Phi, lspot, tau_spot, alpha_max)

    ax.plot(t, At, color='k', lw=1.5, alpha=0.4, label=r'$A(t)$ (exact)')
    ax.plot(t, comp0, color=harmonic_colors[0], lw=1.3, label=harmonic_labels[0])
    ax.plot(t, comp1, color=harmonic_colors[1], lw=1.3, label=harmonic_labels[1])
    ax.plot(t, comp2, color=harmonic_colors[2], lw=1.3, label=harmonic_labels[2])
    ax.plot(t, reconstructed, color='k', lw=1.5, ls='--',
            label=r'$\sum_{n=0}^{2}$ (reconstruction)')

    ax.set_xlabel(r'Time [days]', fontsize=label_fs)
    ax.set_title(label, fontsize=title_fs)
    ax.legend(fontsize=legend_fs - 2, loc='upper right')
    ax.tick_params(labelsize=tick_fs)
    ax.minorticks_on()
    ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.3)

axes[0].set_ylabel(r'$A(t)$', fontsize=label_fs + 2)

fig2.tight_layout()
fig2.savefig(paths.figures / "fig_fourier_decomposition_harmonics.pdf",
             bbox_inches="tight", dpi=300)
