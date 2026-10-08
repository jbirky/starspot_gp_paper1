"""
Illustrate the banded structure of the covariance matrix arising from the
compact support of the starspot kernel.

Left:   Kernel K(tau) showing the compact support at tau = lspot + 2*tau_spot.
Center: Dense covariance matrix with entries colored by magnitude.
Right:  Banded covariance matrix with zero entries masked out, showing the
        bandwidth b = (lspot + 2*tau_spot) / dt.

Output:
    figures/fig_banded_covariance.pdf
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch
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
lspot = 15.0
tau_s = 3.0
sigma_k = 1.0
kappa = 0.0

tau_max = lspot + 2 * tau_s   # compact support boundary

# Observation grid: cadence and duration chosen so the banded
# structure is visually clear
dt = 0.5          # cadence [days]
T_obs = 60.0      # total baseline [days]
t_obs = np.arange(0, T_obs, dt)
N = len(t_obs)
bandwidth = int(np.floor(tau_max / dt))

print(f"N = {N},  bandwidth b = {bandwidth},  "
      f"fill fraction = {(2*bandwidth+1)*N / N**2:.2f}")

# ── Build kernel and covariance matrix ───────────────────────────
hparam = dict(peq=P_eq, kappa=kappa, inc=inc,
              lspot=lspot, tau_spot=tau_s, sigma_k=sigma_k)
ak = AnalyticKernel(hparam, n_harmonics=2, n_lat=64)

# ACF curve
tau_arr = np.linspace(0, T_obs * 0.7, 1000)
k_acf = np.asarray(ak.kernel(tau_arr), dtype=np.float64)
k_acf_norm = k_acf / k_acf[0]

# Covariance matrix
dt_matrix = np.abs(t_obs[:, None] - t_obs[None, :])
dt_unique = np.unique(dt_matrix)
k_unique = np.asarray(ak.kernel(dt_unique), dtype=np.float64)
sorter = np.searchsorted(dt_unique, dt_matrix.ravel())
K_full = k_unique[sorter].reshape(N, N)
K_full_norm = K_full / K_full[0, 0]

# Banded version: zero out entries beyond bandwidth
K_banded = K_full_norm.copy()
K_banded[dt_matrix > tau_max] = 0.0

# ── Figure ───────────────────────────────────────────────────────
label_fs = 15
tick_fs = 12
title_fs = 16

fig = plt.figure(figsize=(16, 5.0))

# Use gridspec for precise control
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1, 1],
                      wspace=0.35, left=0.06, right=0.97,
                      bottom=0.14, top=0.88)

# ── Left panel: Kernel ACF ───────────────────────────────────────
ax0 = fig.add_subplot(gs[0])
ax0.plot(tau_arr, k_acf_norm, color='k', lw=1.8)
ax0.axvline(tau_max, color='C3', ls='--', lw=1.5, alpha=0.8)
ax0.axhline(0, color='gray', ls='-', lw=0.4)

# Shade the compact support region
ax0.axvspan(tau_max, tau_arr[-1], color='C3', alpha=0.06)

# Annotate compact support
ax0.annotate(rf'$\ell_{{\rm spot}} + 2\tau_{{\rm spot}} = {tau_max:.0f}$\,d',
             xy=(tau_max, 0.5), xytext=(tau_max + 3, 0.65),
             fontsize=13, color='C3',
             arrowprops=dict(arrowstyle='->', color='C3', lw=1.2))

ax0.text(tau_max + (tau_arr[-1] - tau_max) / 2, 0.15,
         r'$K(\tau) = 0$', fontsize=14, color='C3',
         ha='center', va='center', alpha=0.8)

# Mark rotation period multiples
for m in range(1, int(tau_arr[-1] / P_eq) + 1):
    ax0.axvline(m * P_eq, color='gray', ls=':', lw=0.3, alpha=0.3)

ax0.set_xlim(0, tau_arr[-1])
ax0.set_ylim(-0.15, 1.05)
ax0.set_xlabel(r'Time lag $\tau$ [days]', fontsize=label_fs)
ax0.set_ylabel(r'$K(\tau)\,/\,K(0)$', fontsize=label_fs)
ax0.set_title('Kernel (compact support)', fontsize=title_fs)
ax0.tick_params(labelsize=tick_fs)
ax0.minorticks_on()

# ── Center panel: Dense covariance matrix ────────────────────────
ax1 = fig.add_subplot(gs[1])
im1 = ax1.imshow(K_full_norm, cmap='RdBu_r', vmin=-0.15, vmax=1.0,
                  aspect='equal', interpolation='nearest',
                  extent=[0, T_obs, T_obs, 0])
ax1.set_xlabel('Time [days]', fontsize=label_fs)
ax1.set_ylabel('Time [days]', fontsize=label_fs)
ax1.set_title('Dense covariance matrix', fontsize=title_fs)
ax1.tick_params(labelsize=tick_fs)

# ── Right panel: Banded covariance matrix ────────────────────────
ax2 = fig.add_subplot(gs[2])

# Show the banded matrix; mask zeros for visual clarity
K_display = np.ma.masked_where(K_banded == 0, K_banded)
ax2.imshow(np.where(dt_matrix > tau_max, np.nan, 0),
           cmap='Greys', vmin=0, vmax=1, aspect='equal',
           extent=[0, T_obs, T_obs, 0], alpha=0.08)
im2 = ax2.imshow(K_display, cmap='RdBu_r', vmin=-0.15, vmax=1.0,
                  aspect='equal', interpolation='nearest',
                  extent=[0, T_obs, T_obs, 0])
ax2.set_xlabel('Time [days]', fontsize=label_fs)
ax2.set_title('Banded covariance matrix', fontsize=title_fs)
ax2.tick_params(labelsize=tick_fs)
ax2.set_yticklabels([])

# Annotate bandwidth
# Draw lines showing the band edges
band_days = tau_max
ax2.plot([0, T_obs - band_days], [band_days, T_obs],
         color='C3', ls='--', lw=1.2, alpha=0.7)
ax2.plot([band_days, T_obs], [0, T_obs - band_days],
         color='C3', ls='--', lw=1.2, alpha=0.7)

# Bandwidth annotation
mid = T_obs / 2
ax2.annotate('', xy=(mid + band_days / 2 + 0.5, mid + 0.5),
             xytext=(mid - 0.5, mid - band_days / 2 - 0.5),
             arrowprops=dict(arrowstyle='<->', color='C3', lw=1.5))
ax2.text(mid + band_days / 2 + 2, mid - band_days / 2 + 1,
         rf'$b = {bandwidth}$',
         fontsize=14, color='C3', rotation=-45,
         ha='left', va='top')

# Shared colorbar
cbar_ax = fig.add_axes([0.37, 0.04, 0.30, 0.025])
cb = fig.colorbar(im2, cax=cbar_ax, orientation='horizontal')
cb.set_label(r'$K_{ij}\,/\,K(0)$', fontsize=13)
cb.ax.tick_params(labelsize=tick_fs)

# Parameter annotation
param_text = (rf'$P_{{\rm eq}} = {P_eq:.0f}$\,d, '
              rf'$\ell_{{\rm spot}} = {lspot:.0f}$\,d, '
              rf'$\tau_{{\rm spot}} = {tau_s:.0f}$\,d, '
              rf'$\Delta t = {dt}$\,d, '
              rf'$N = {N}$')
fig.text(0.5, 0.97, param_text, ha='center', fontsize=13, va='top',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.savefig(paths.figures / "fig_banded_covariance.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_banded_covariance.pdf")
