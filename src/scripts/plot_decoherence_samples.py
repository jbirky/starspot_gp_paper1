"""
Show how differential rotation causes decoherence in GP lightcurve samples.

For each kappa value, draw GP samples from the analytic kernel and plot them
alongside the kernel ACF and its upper envelope. The periodic signal loses
coherence at large lags for kappa != 0.

Output:
    figures/fig_decoherence_samples.pdf
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import matplotlib.pyplot as plt
import matplotlib
import paths

from spotgp.analytic_kernel import AnalyticKernel

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

# ── Parameters ───────────────────────────────────────────────────
P_eq = 5.0
inc = np.pi / 2
lspot = 30.0
tau_s = 3.0
sigma_k = 1.0

kappa_values = [0.0, 0.5, 1.0]
n_samples = 3
rng = np.random.default_rng(42)

# Time arrays
t_lc = np.linspace(0, 60, 600)      # lightcurve time grid
tau_acf = np.linspace(0, 40, 800)    # ACF lag grid

# ── Figure ───────────────────────────────────────────────────────
label_fs = 14
tick_fs = 12
legend_fs = 11
title_fs = 15
lw = 1.5

n_kappa = len(kappa_values)
fig, axes = plt.subplots(n_kappa, 2, figsize=(14, 3.2 * n_kappa),
                         gridspec_kw={'wspace': 0.25, 'hspace': 0.15,
                                      'width_ratios': [1.6, 1]})

colors_sample = ['C0', 'C4', 'C3']

for i, kappa in enumerate(kappa_values):
    ax_lc = axes[i, 0]
    ax_acf = axes[i, 1]

    # Build kernel
    hparam = dict(peq=P_eq, kappa=kappa, inc=inc,
                  lspot=lspot, tau_spot=tau_s, sigma_k=sigma_k)
    ak = AnalyticKernel(hparam, n_harmonics=2, n_lat=64)

    # ── ACF panel ────────────────────────────────────────────────
    k_acf = np.asarray(ak.kernel(tau_acf), dtype=np.float64)
    k_acf_norm = k_acf / k_acf[0] if k_acf[0] > 0 else k_acf

    # Upper envelope: R_Gamma(tau) / R_Gamma(0)
    R_env = np.asarray(ak.R_Gamma(tau_acf), dtype=np.float64)
    R_env_norm = R_env / R_env[0] if R_env[0] > 0 else R_env

    ax_acf.plot(tau_acf, k_acf_norm, color='k', lw=lw, label='Kernel')
    ax_acf.plot(tau_acf, R_env_norm, color='k', lw=1.2, ls='--', alpha=0.5,
                label=r'$R_\Gamma(\tau)/R_\Gamma(0)$')
    ax_acf.plot(tau_acf, -R_env_norm, color='k', lw=1.2, ls='--', alpha=0.5)
    ax_acf.axhline(0, color='gray', ls='-', lw=0.4)

    # Mark rotation period multiples
    for m in range(1, int(tau_acf[-1] / P_eq) + 1):
        ax_acf.axvline(m * P_eq, color='gray', ls=':', lw=0.4, alpha=0.3)

    ax_acf.set_xlim(0, tau_acf[-1])
    ax_acf.set_ylim(-0.05, 1.05)
    ax_acf.set_ylabel(r'$K(\tau)/K(0)$', fontsize=label_fs)
    ax_acf.tick_params(labelsize=tick_fs)
    ax_acf.minorticks_on()

    # Kappa annotation
    ax_acf.text(0.97, 0.92, rf'$\kappa = {kappa}$',
                transform=ax_acf.transAxes, fontsize=16,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

    if i == 0:
        ax_acf.legend(fontsize=legend_fs, loc='upper left')
    if i == n_kappa - 1:
        ax_acf.set_xlabel(r'Time lag $\tau$ [days]', fontsize=label_fs)

    # ── Lightcurve samples panel ─────────────────────────────────
    # Build covariance matrix
    dt_matrix = np.abs(t_lc[:, None] - t_lc[None, :])
    # Evaluate kernel on unique lags for efficiency
    dt_unique = np.unique(dt_matrix)
    k_unique = ak.kernel(dt_unique)
    # Map back to full matrix
    sorter = np.searchsorted(dt_unique, dt_matrix.ravel())
    K_matrix = np.asarray(k_unique[sorter], dtype=np.float64).reshape(dt_matrix.shape)
    # Add jitter for numerical stability
    K_matrix += 1e-6 * np.eye(len(t_lc))

    # Draw samples
    L = np.linalg.cholesky(K_matrix)
    for j in range(n_samples):
        z = rng.standard_normal(len(t_lc))
        sample = L @ z
        # Normalize for visual comparison
        sample = sample / np.std(sample)
        offset = (n_samples - 1 - j) * 3.5
        ax_lc.plot(t_lc, sample + offset, color=colors_sample[j],
                   lw=0.8, alpha=0.85)

    # Mark rotation period multiples
    for m in range(1, int(t_lc[-1] / P_eq) + 1):
        ax_lc.axvline(m * P_eq, color='gray', ls=':', lw=0.4, alpha=0.15)

    ax_lc.set_xlim(0, t_lc[-1])
    ax_lc.set_ylabel('Flux (offset)', fontsize=label_fs)
    ax_lc.tick_params(labelsize=tick_fs, left=False, labelleft=False)
    ax_lc.minorticks_on()
    ax_lc.tick_params(which='minor', left=False)

    # Kappa annotation
    ax_lc.text(0.01, 0.92, rf'$\kappa = {kappa}$',
               transform=ax_lc.transAxes, fontsize=16,
               va='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='gray', alpha=0.9))

    if i == n_kappa - 1:
        ax_lc.set_xlabel('Time [days]', fontsize=label_fs)

# Column titles
axes[0, 0].set_title('GP lightcurve samples', fontsize=title_fs)
axes[0, 1].set_title('Kernel autocorrelation', fontsize=title_fs)

# Parameter annotation
param_text = (rf'$P_{{\rm eq}} = {P_eq:.0f}$\,d, '
              rf'$I = {np.degrees(inc):.0f}^\circ$, '
              rf'$\ell_{{\rm spot}} = {lspot:.0f}$\,d, '
              rf'$\tau_{{\rm spot}} = {tau_s:.0f}$\,d')
fig.text(0.5, 0.02, param_text, ha='center', fontsize=16,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.savefig(paths.figures / "fig_decoherence_samples.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_decoherence_samples.pdf")
