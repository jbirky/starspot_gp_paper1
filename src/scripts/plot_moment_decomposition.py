"""
Decompose a single-spot signal into 1st and 2nd moments of the GP.

Top panel:  80 realizations of A(t) = Gamma(t) * V(t) for random longitudes,
            with the ensemble mean <A>_Lambda = c_0 * Gamma(t) (1st moment).
Bottom panel: residuals A(t) - c_0*Gamma(t) and their standard deviation
              envelope (2nd moment).

Output: figures/fig_moment_decomposition.pdf
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
tick_fs = 18
legend_fs = 14
title_fs = 18

# ── Parameters ───────────────────────────────────────────────────
P = 5.0
omega0 = 2 * np.pi / P
alpha_max = 0.1
ell = 15.0
tau_s = 3.0
inc_deg = 60
phi_deg = 30
inc = np.radians(inc_deg)
phi = np.radians(phi_deg)

n_real = 80


# ── Helper functions ─────────────────────────────────────────────

def trapezoidal_Gamma(t, ell, tau_s, alpha_max):
    """Squared trapezoidal envelope, centered at t=0."""
    half_plat = ell / 2.0
    abst = np.abs(t)
    alpha = np.where(
        abst <= half_plat,
        alpha_max,
        np.where(
            abst <= half_plat + tau_s,
            alpha_max * (1.0 - (abst - half_plat) / tau_s),
            0.0,
        ),
    )
    return alpha ** 2


def visibility(t, inc, phi, Lambda):
    """V(t) = max{cos(beta), 0} for a given longitude phase."""
    a0 = np.cos(inc) * np.sin(phi)
    a1 = np.sin(inc) * np.cos(phi)
    return np.maximum(a0 + a1 * np.cos(omega0 * t - Lambda), 0.0)


def compute_c0(inc, phi):
    """Zeroth Fourier coefficient of the visibility function."""
    a0 = np.cos(inc) * np.sin(phi)
    a1 = np.sin(inc) * np.cos(phi)
    ratio = -a0 / a1
    if ratio <= -1.0:
        return a0
    if ratio >= 1.0:
        return 0.0
    theta_vis = np.arccos(ratio)
    return (a0 * theta_vis + a1 * np.sin(theta_vis)) / np.pi


# ── Compute signals ──────────────────────────────────────────────
dur = ell + 2 * tau_s
t = np.linspace(-0.5 * dur - 1, 0.5 * dur + 1, 3000)

Gamma = trapezoidal_Gamma(t, ell, tau_s, alpha_max)
c0 = compute_c0(inc, phi)

np.random.seed(42)
Lambdas = np.random.uniform(0, 2 * np.pi, n_real)
sigs = np.array([Gamma * visibility(t, inc, phi, L) for L in Lambdas])
mean_sig = Gamma * c0
resids = sigs - mean_sig
std = np.std(resids, axis=0)


# ── Figure ───────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                gridspec_kw={'hspace': 0.08})

# Panel 1: realizations + mean (1st moment)
for s in sigs:
    ax1.plot(t, s, color='steelblue', alpha=0.08, lw=0.6)
ax1.plot(t, Gamma, 'k--', lw=1.5, alpha=0.35, label=r'Envelope $\Gamma(t)$')
ax1.plot(t, mean_sig, 'C3-', lw=2.5, zorder=10,
         label=r'1st moment: $\langle A \rangle_\Lambda = c_0\,\Gamma(t)$')
ax1.set_ylabel(r'$A(t)$', fontsize=label_fs)
ax1.legend(loc='upper right', framealpha=0.9, fontsize=legend_fs)
ax1.set_title(
    rf'Single-spot signal for {n_real} random longitudes'
    rf'  ($I={inc_deg}^\circ$, $\Phi={phi_deg}^\circ$, $P={P:.0f}$\,d)',
    fontsize=title_fs)
ax1.tick_params(labelsize=tick_fs)
ax1.minorticks_on()

ax1.text(0.02, 0.08,
         r'Mean $\propto\, c_0 \cdot \int\!\Gamma\,dt$'
         r'  (linear in $c_0$, 1st moment of $\Gamma$)',
         transform=ax1.transAxes, fontsize=legend_fs, color='C3',
         bbox=dict(fc='white', ec='none', alpha=0.8))

# Panel 2: residuals (2nd moment)
for r in resids:
    ax2.plot(t, r, color='C1', alpha=0.08, lw=0.6)
ax2.fill_between(t, -std, std, alpha=0.25, color='C1',
                 label=r'$\pm\sigma(t)$ --- 2nd moment (spread around mean)')
ax2.axhline(0, color='gray', ls='-', lw=0.4)
ax2.set_ylabel(r'Residual: $A(t) - c_0\,\Gamma(t)$', fontsize=label_fs)
ax2.set_xlabel(r'Time [days]', fontsize=label_fs)
ax2.legend(loc='upper right', framealpha=0.9, fontsize=legend_fs)
ax2.tick_params(labelsize=tick_fs)
ax2.minorticks_on()

ax2.text(0.02, 0.08,
         r'Covariance $\propto\, |c_n|^2 \cdot R_\Gamma(\tau)$'
         r'  (quadratic in $c_n$, 2nd moment of $\Gamma$)',
         transform=ax2.transAxes, fontsize=legend_fs, color='C1',
         bbox=dict(fc='white', ec='none', alpha=0.8))

fig.savefig(paths.figures / "fig_moment_decomposition.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_moment_decomposition.pdf")
