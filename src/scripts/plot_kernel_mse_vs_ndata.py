"""
Kernel MSE vs. number of lightcurve datapoints for different noise levels.

For each noise level (sigma_noise), simulates NAVG random lightcurves with
varying duration (fixed cadence), computes the sample ACF for each, and
measures the MSE against the analytic kernel.  Plots the mean MSE with
1-sigma errorbars.  One colored line per noise level.

Output:
    figures/fig_kernel_mse_vs_ndata.pdf
"""
import sys
import numpy as np

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import matplotlib.pyplot as plt
import matplotlib
import paths

from spotgp.analytic_kernel import AnalyticKernel
from spotgp.lightcurve import LightcurveModel

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

# ── Parameters ────────────────────────────────────────────────────
params = dict(
    peq=5.0,
    kappa=0.0,
    inc=np.pi / 2,
    nspot_rate=0.25,
    lspot=15.0,
    tau_spot=3.0,
    alpha_max=0.1,
    fspot=0.0,
)

TSAMP = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0  # sampling cadence [days]
N_HARMONICS = 3
TMAX_ACF = params["lspot"] + 2*params["tau_spot"]     # max lag for MSE computation [days]

# Simulation durations [days] — ndata = tsim / tsamp
TSIM_VALUES = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

# Gaussian noise standard deviations to compare (in relative flux units)
SIGMA_NOISE = [1e-4, 1e-3, 1e-2, 1e-1]

NAVG = 10           # number of random lightcurves to average over
SEED = 42


def sample_acf(flux, nlags):
    """Compute the sample autocorrelation function of a single time series.

    Uses the standard biased estimator (1/N normalization) which is
    guaranteed non-negative at lag 0 and consistent for stationary processes.
    """
    flux = flux - np.mean(flux)
    n = len(flux)
    acf = np.correlate(flux, flux, mode='full')[n - 1:]  # lags 0..n-1
    acf = acf / acf[0] if acf[0] != 0 else acf
    return acf[:nlags]


# ── Compute analytic kernel (ground truth) ────────────────────────
ak = AnalyticKernel(params, n_harmonics=N_HARMONICS)

# ── Compute MSE for each (tsim, sigma_noise), averaged over NAVG trials ──
ndata_values = (TSIM_VALUES / TSAMP).astype(int)
# Store arrays of shape (len(TSIM_VALUES), NAVG) per noise level
results_mean = {sigma: [] for sigma in SIGMA_NOISE}
results_std = {sigma: [] for sigma in SIGMA_NOISE}
rng = np.random.default_rng(SEED)

# Analytic ACF on lag grid (same for all tsim)
nlags_acf = int(TMAX_ACF / TSAMP) + 1
tau_arr = np.arange(nlags_acf) * TSAMP
k_ana = ak.kernel(tau_arr)
acf_ana = k_ana / k_ana[0] if k_ana[0] != 0 else k_ana

for tsim in TSIM_VALUES:
    ndata = int(tsim / TSAMP)
    print(f"tsim={tsim:6g} d  (ndata={ndata})")

    # Collect MSE over NAVG random lightcurves
    mse_trials = {sigma: np.zeros(NAVG) for sigma in SIGMA_NOISE}

    for j in range(NAVG):
        # Seed numpy global RNG so each LightcurveModel draws different spots
        np.random.seed(rng.integers(0, 2**31))
        nspot = max(1, int(params["nspot_rate"] * tsim))
        lc = LightcurveModel(
            peq=params["peq"], kappa=params["kappa"], inc=params["inc"],
            nspot=nspot,
            tem=params["tau_spot"], tdec=params["tau_spot"],
            alpha_max=params["alpha_max"], fspot=params["fspot"],
            lspot=params["lspot"],
            tsim=tsim, tsamp=TSAMP,
        )
        flux_clean = lc.flux

        for sigma in SIGMA_NOISE:
            if sigma > 0:
                flux = flux_clean + rng.normal(0, sigma, flux_clean.shape)
            else:
                flux = flux_clean

            acf_num = sample_acf(flux, nlags_acf)
            mse_trials[sigma][j] = np.mean((acf_num - acf_ana) ** 2)

    for sigma in SIGMA_NOISE:
        mean_mse = np.mean(mse_trials[sigma])
        std_mse = np.std(mse_trials[sigma])
        results_mean[sigma].append(mean_mse)
        results_std[sigma].append(std_mse)
        print(f"  sigma_noise={sigma:.1e}, MSE={mean_mse:.3e} +/- {std_mse:.3e}")

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(SIGMA_NOISE)))

for i, sigma in enumerate(SIGMA_NOISE):
    label = (r"$\sigma_{\rm noise} = 0$" if sigma == 0
             else rf"$\sigma_{{\rm noise}} = {sigma:.0e}$")
    mean = np.array(results_mean[sigma])
    std = np.array(results_std[sigma])
    ax.errorbar(ndata_values, mean, yerr=std, fmt='o-', color=colors[i],
                lw=2, markersize=5, capsize=3, label=label)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Number of datapoints in lightcurve', fontsize=18)
ax.set_ylabel('MSE (analytic vs. numerical ACF)', fontsize=18)
ax.set_title(f"Sampling Cadence = {TSAMP} d", fontsize=20)
ax.legend(fontsize=14, title=r'Gaussian noise $\sigma$', title_fontsize=14, loc="lower left")
ax.tick_params(labelsize=14)
ax.minorticks_on()
ax.grid(True, alpha=0.4, which='both')
ax.set_ylim(1e-4, 3e-1)
ax.set_xlim(ndata_values[0] / 1.5, ndata_values[-1] * 1.5)

fig.tight_layout()
outpath = paths.figures / f"fig_kernel_mse_vs_ndata_{str(TSAMP).replace('.', 'p')}d.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"\nSaved figure to {outpath}")
