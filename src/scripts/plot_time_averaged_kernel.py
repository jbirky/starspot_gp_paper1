"""
Compare the time-averaged kernel to the Wiener--Khinchin (ensemble) kernel.

The analytic kernel K(tau) is the ensemble covariance about the true mean
flux.  A sample ACF is a time average over one window of length T about that
window's own mean, whose expectation is the time-averaged kernel

    K_T(tau) = K(tau) - (1/T) int K(s) ds,

a uniform downward shift equal to the variance of the window mean.  Each K_T
curve is checked against the mean sample ACF of spotgp lightcurves with a
Poisson number of spots.

Output:
    figures/fig_time_averaged_kernel.pdf
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

from spotgp.analytic_kernel import AnalyticKernel
from spotgp.lightcurve import LightcurveModel

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}\usepackage{amssymb}'

# ── Parameters ───────────────────────────────────────────────────
params = dict(
    peq=5.0,
    kappa=0.0,
    inc=np.pi / 2,
    lspot=15.0,
    tau_spot=3.0,
    alpha_max=0.1,
    fspot=0.0,
)
nspot_rate = 0.5           # spot emergence rate [spots/day]
TSAMP = 0.05               # cadence [days]
T_VALUES = [50.0, 100.0, 200.0]
NSIM = 1000                # simulated lightcurves per baseline
N_HARMONICS = 8
TMAX_PLOT = 30.0           # plotted lag range [days]
SEED = 42

# ── Wiener--Khinchin kernel, normalized to K(0) = 1 ──────────────
lag = np.arange(0, TMAX_PLOT + TSAMP / 2, TSAMP)
ak = AnalyticKernel(dict(params, nspot=10), n_harmonics=N_HARMONICS)
K = np.array(ak.kernel(lag))
K /= K[0]

# int K ds over all lags (K is even and vanishes beyond lspot + 2 tau_spot)
K_int = 2 * np.trapezoid(K, lag)
print(f"int K ds / K(0) = {K_int:.3f} d")


# ── Monte Carlo sample ACFs ──────────────────────────────────────
def sample_acfs(T, rng):
    """Sample ACF of each lightcurve about its own mean, 1/(n-k) normalized,
    in units of the ensemble variance K(0)."""
    W = T + params["lspot"] + 2 * params["tau_spot"]
    fluxes = []
    for _ in range(NSIM):
        nspot = max(1, rng.poisson(nspot_rate * W))
        np.random.seed(rng.integers(2**31))   # LightcurveModel draws from np.random
        lc = LightcurveModel(nspot=nspot, tsim=T, tsamp=TSAMP, **params)
        fluxes.append(lc.flux)
    fluxes = np.array(fluxes)
    n = fluxes.shape[1]
    nlag = len(lag)

    d = fluxes - fluxes.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(d, 2 * n, axis=1))**2
    acf = np.fft.irfft(power, axis=1)[:, :nlag] / (n - np.arange(nlag))
    var_ens = fluxes.var(axis=0).mean()      # unbiased for Poisson spot counts
    return acf / var_ens


rng = np.random.default_rng(SEED)
mc = {}
for T in T_VALUES:
    acf = sample_acfs(T, rng)
    mc[T] = (acf.mean(axis=0), acf.std(axis=0) / np.sqrt(NSIM))
    tail = lag > params["lspot"] + 2 * params["tau_spot"] + 1
    print(f"T = {T:5.0f} d: predicted offset {K_int / T:.4f},"
          f"  MC offset {-mc[T][0][tail].mean():.4f}"
          f"  (max |MC - K_T| = {np.abs(mc[T][0] - (K - K_int / T)).max():.4f})")

# ── Figure ───────────────────────────────────────────────────────
label_fs = 14
tick_fs = 12
legend_fs = 11
colors = ['C0', 'C1', 'C2']

fig, (ax, ax_off) = plt.subplots(
    1, 2, figsize=(13, 4.4), gridspec_kw={'width_ratios': [2.1, 1], 'wspace': 0.28})

ax.axhline(0, color='0.6', lw=0.8)
ax.plot(lag, K, color='k', ls='--', lw=2, zorder=5,
        label=r'Wiener--Khinchin $\mathcal{K}(\tau)$')
step = int(round(0.5 / TSAMP))
for T, c in zip(T_VALUES, colors):
    ax.plot(lag, K - K_int / T, color=c, lw=1.6,
            label=rf'$\mathcal{{K}}_T(\tau)$, $T = {T:.0f}$ d')
    ax.plot(lag[::step], mc[T][0][::step], 'o', ms=3.5, color=c, alpha=0.8)
ax.plot([], [], 'o', ms=3.5, color='0.4', label='Mean sample ACF (simulations)')
for k in range(1, int(TMAX_PLOT / params["peq"]) + 1):
    ax.axvline(k * params["peq"], color='0.85', lw=0.8, zorder=0)

ax.set_xlim(0, TMAX_PLOT)
ax.set_xlabel(r'Time lag $\tau$ [days]', fontsize=label_fs)
ax.set_ylabel(r'Covariance / $\mathcal{K}(0)$', fontsize=label_fs)
ax.tick_params(labelsize=tick_fs)
ax.legend(fontsize=legend_fs, frameon=False, loc='upper right')

# Offset vs baseline: the variance of the window mean, (1/T) int K ds
T_grid = np.geomspace(20, 1000, 200)
ax_off.loglog(T_grid, K_int / T_grid, color='k', lw=1.6,
              label=r'$T^{-1}\!\int \mathcal{K}(s)\,ds$')
tail = lag > params["lspot"] + 2 * params["tau_spot"] + 1
for T, c in zip(T_VALUES, colors):
    off = -mc[T][0][tail].mean()
    err = mc[T][1][tail].mean()
    ax_off.errorbar(T, off, yerr=err, fmt='o', ms=6, color=c, capsize=3)
ax_off.plot([], [], 'o', ms=6, color='0.4', label='Simulations')
for T_ref, name in [(27.4, 'TESS sector'), (90.0, 'Kepler quarter')]:
    ax_off.axvline(T_ref, color='0.75', ls=':', lw=1, zorder=0)
    ax_off.text(T_ref * 1.08, 6e-3, name, rotation=90, va='bottom', ha='left',
                fontsize=10, color='0.45')
ax_off.set_xlim(15, 1000)
ax_off.set_ylim(5e-3, 0.6)
ax_off.set_xlabel(r'Baseline $T$ [days]', fontsize=label_fs)
ax_off.set_ylabel(r'Offset / $\mathcal{K}(0)$', fontsize=label_fs)
ax_off.tick_params(labelsize=tick_fs)
ax_off.legend(fontsize=legend_fs, frameon=False, loc='upper right')

fig.savefig(paths.figures / "fig_time_averaged_kernel.pdf", bbox_inches='tight')
plt.close(fig)
