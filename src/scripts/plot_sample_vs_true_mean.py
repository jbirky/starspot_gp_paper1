"""
Compare the sample mean of one simulated lightcurve to the true mean flux.

Draws a single spotgp lightcurve with a Poisson number of spots and marks two
horizontal levels: the true ensemble mean E[F] from Eq. (mean_flux), about
which the analytic kernel is defined, and the sample time mean of this one
lightcurve, about which a sample ACF is computed. The true E[F] is also
checked against the ensemble mean of many independent simulations.

Output:
    figures/fig_sample_vs_true_mean.pdf
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

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
nspot_rate = 0.5      # spot emergence rate [spots/day]
TSIM = 100.0          # lightcurve duration [days]
TSAMP = 0.05          # cadence [days]
NSIM_CHECK = 2000     # simulations for the ensemble-mean check
SEED = 3

# Spots emerge uniformly over the window that overlaps [0, TSIM]
# (LightcurveModel._tmax_window), so the expected spot count is rate * W.
tau = params["tau_spot"]
W = TSIM + params["lspot"] + 2 * tau
nspot_mean = nspot_rate * W


def simulate(rng):
    """One lightcurve with N ~ Poisson(rate * W) spots."""
    nspot = max(1, rng.poisson(nspot_mean))
    np.random.seed(rng.integers(2**31))   # LightcurveModel draws from np.random
    return LightcurveModel(nspot=nspot, tsim=TSIM, tsamp=TSAMP, **params)


# ── True mean flux, Eq. (mean_flux) ──────────────────────────────
# int Gamma dt for the squared trapezoid: plateau plus two parabolic ramps
int_gamma = params["lspot"] + 2 * tau / 3

# Latitude-averaged c_0 = Vbar(Phi, I), Eq. (mean_visibility), uniform p(Phi)
phi = np.linspace(-np.pi / 2, np.pi / 2, 4001)
b0 = np.cos(params["inc"]) * np.sin(phi)
b1 = np.sin(params["inc"]) * np.cos(phi)
theta_vis = np.arccos(np.clip(-b0 / np.maximum(b1, 1e-15), -1, 1))
vbar = (b0 * theta_vis + b1 * np.sin(theta_vis)) / np.pi
vbar_avg = np.trapezoid(vbar, phi) / np.pi

EF_true = 1 - nspot_rate * (1 - params["fspot"]) * params["alpha_max"]**2 \
    * int_gamma * vbar_avg

# ── Simulations ──────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
lc = simulate(rng)
t, flux = lc.t, lc.flux
EF_sample = flux.mean()

ens = np.array([simulate(rng).flux for _ in range(NSIM_CHECK)])
EF_ens = ens.mean()
EF_ens_err = ens.mean(axis=1).std() / np.sqrt(NSIM_CHECK)
sigma_F = ens.std()

print(f"N_spot in plotted lightcurve: {lc.nspot}  (expected {nspot_mean:.1f})")
print(f"True E[F], Eq. mean_flux:     1 - {1 - EF_true:.4e}")
print(f"Ensemble mean ({NSIM_CHECK} sims):     1 - {1 - EF_ens:.4e}"
      f"  +/- {EF_ens_err:.1e}")
print(f"Sample mean (this LC):        1 - {1 - EF_sample:.4e}"
      f"  ({(EF_sample - EF_true) / sigma_F:+.2f} sigma_F from E[F])")

# ── Figure ───────────────────────────────────────────────────────
label_fs = 14
tick_fs = 12
legend_fs = 11

ppt = lambda f: (f - 1) * 1e3

fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(t, ppt(flux), color='k', lw=1.0, label='Simulated lightcurve')
ax.axhline(ppt(EF_true), color='C0', ls='--', lw=2,
           label=r'True mean $\mathbb{E}[F]$')
ax.axhline(ppt(EF_sample), color='C3', ls='-', lw=2,
           label=r'Sample mean $\bar{F} = \frac{1}{T}\int_0^T F\,dt$')
ax.axhspan(ppt(min(EF_true, EF_sample)), ppt(max(EF_true, EF_sample)),
           color='C3', alpha=0.12, lw=0)

ax.set_xlim(0, TSIM)
ax.set_xlabel('Time [days]', fontsize=label_fs)
ax.set_ylabel(r'$(F - 1) \times 10^3$', fontsize=label_fs)
ax.tick_params(labelsize=tick_fs)
ax.legend(fontsize=legend_fs, loc='lower center', bbox_to_anchor=(0.5, 1.0),
          ncol=3, frameon=False)

fig.tight_layout()
fig.savefig(paths.figures / "fig_sample_vs_true_mean.pdf", bbox_inches='tight')
plt.close(fig)
