"""
Single-row, two-panel figure showing the harmonic decomposition of the
analytic kernel (left) and its power spectral density (right) for one
fiducial parameter set:

    k(tau) = sigma_k^2 R_Gamma(tau) < |c_0|^2 + 2 sum_n |c_n|^2 cos(n w0 tau) >_phi
    S(w)   = sigma_k^2 < |c_0|^2 Gh(w)^2
                         + sum_n |c_n|^2 [Gh(w - n w0)^2 + Gh(w + n w0)^2] >_phi

where <.>_phi is the latitude average and Gh = |FT[Gamma]|.  The n = 0, 1, 2
contributions are shown individually along with the total, and the kernel
panel is annotated with each harmonic's fractional contribution to k(0).

The decomposition uses the same trapezoid latitude quadrature as
spotgp.AnalyticKernel, and the summed components are checked against the
package totals at runtime.

Output:
    figures/fig_kernel_psd_harmonic_decomposition_single.pdf
"""
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import matplotlib.pyplot as plt
import matplotlib
import jax
import jax.numpy as jnp
import paths

jax.config.update("jax_enable_x64", True)

from spotgp import (
    TrapezoidSymmetricEnvelope,
    VisibilityFunction,
    SpotEvolutionModel,
)
from spotgp.analytic_kernel import AnalyticKernel

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

# ── Configuration ────────────────────────────────────────────────
N_HARMONICS = 2
N_LAT = 96
N_TAU = 1200
N_FREQ = 1600

# Fiducial model parameters.  The closed-form R_Gamma assumes
# tau_spot < lspot/2 — keep values inside that.
PARAMS = dict(peq=5.0, kappa=0.2, inc=np.radians(60.0),
              lspot=15.0, tau_spot=3.0, sigma_k=0.05)

# Explicit Okabe-Ito hexes (CVD-safe): importing spotgp switches matplotlib to
# the 'classic' style, so 'C0'/'C1'/'C2' would resolve to blue/green/red.
COLOR_TOTAL = 'k'
COLORS_N = {0: '#0072B2', 1: '#E69F00', 2: '#CC79A7'}
LABELS_N = {0: r'$n=0$', 1: r'$n=1$', 2: r'$n=2$'}

label_fs = 15
tick_fs = 12
legend_fs = 12
ann_fs = 14


# ── Model / decomposition helpers ────────────────────────────────

def build_kernel(p):
    """AnalyticKernel for a parameter dict p."""
    envelope = TrapezoidSymmetricEnvelope(lspot=p["lspot"], tau_spot=p["tau_spot"])
    visibility = VisibilityFunction(peq=p["peq"], kappa=p["kappa"], inc=p["inc"])
    model = SpotEvolutionModel(envelope=envelope, visibility=visibility,
                               sigma_k=p["sigma_k"])
    return AnalyticKernel(model, n_harmonics=N_HARMONICS, n_lat=N_LAT,
                          quadrature="trapezoid")


def harmonic_components(ak, tau, omega):
    """
    Per-harmonic contributions to the latitude-averaged kernel and PSD.

    Replicates AnalyticKernel._kernel_stationary / compute_psd term by term
    (same phi grid, same rectangle-sum/trapezoid-norm quadrature) so that the
    components sum exactly to the package totals.

    Returns
    -------
    k_parts : list of (n_tau,) arrays, one per harmonic n = 0..N_HARMONICS
    S_parts : list of (n_freq,) arrays
    """
    phi_min, phi_max = ak.lat_range
    phi = np.linspace(phi_min, phi_max, ak.n_lat)
    dphi = phi[1] - phi[0]
    lat_dist = ak.spot_model.latitude_distribution
    w = np.array([float(lat_dist(float(p))) for p in phi])
    norm = np.trapezoid(w, phi)
    lat_w = w * dphi / norm

    cn_sq = np.array([np.asarray(ak.cn_squared(p)) for p in phi])  # (n_lat, N+1)
    w0 = np.asarray(ak.omega0(jnp.asarray(phi)))                   # (n_lat,)
    R = np.asarray(ak.R_Gamma(jnp.asarray(tau)))                   # (n_tau,)
    s2 = ak.sigma_k ** 2

    def Gh_sq(om):
        return np.asarray(ak.envelope.Gamma_hat(jnp.asarray(om))) ** 2

    k_parts, S_parts = [], []
    for n in range(ak.n_harmonics + 1):
        if n == 0:
            c0_avg = np.sum(lat_w * cn_sq[:, 0])
            k_parts.append(s2 * R * c0_avg)
            S_parts.append(s2 * c0_avg * Gh_sq(omega))
        else:
            wn = lat_w * cn_sq[:, n]
            cosine = np.cos(n * w0[:, None] * tau[None, :])
            k_parts.append(s2 * R * 2.0 * np.sum(wn[:, None] * cosine, axis=0))
            sidebands = (Gh_sq(omega[None, :] - n * w0[:, None])
                         + Gh_sq(omega[None, :] + n * w0[:, None]))
            S_parts.append(s2 * np.sum(wn[:, None] * sidebands, axis=0))
    return k_parts, S_parts


# ── Compute ──────────────────────────────────────────────────────

ak = build_kernel(PARAMS)

tau_max = 1.02 * (PARAMS["lspot"] + 2 * PARAMS["tau_spot"])
f_max = 2.6 / PARAMS["peq"]
tau = np.linspace(0.0, tau_max, N_TAU)
freq = np.linspace(0.0, f_max, N_FREQ)
omega = 2 * np.pi * freq

k_parts, S_parts = harmonic_components(ak, tau, omega)
k_total = sum(k_parts)
S_total = sum(S_parts)

# Verify against the package totals
k_pkg = np.asarray(ak.kernel(jnp.asarray(tau)))
_, S_pkg = ak.compute_psd(jnp.asarray(omega))
err_k = np.max(np.abs(k_total - k_pkg)) / np.max(np.abs(k_pkg))
err_S = np.max(np.abs(S_total - np.asarray(S_pkg))) / np.max(np.abs(S_pkg))
print(f"Max relative deviation of summed harmonics from spotgp totals: "
      f"{max(err_k, err_S):.2e}")

# Fractional contribution of each harmonic to the variance k(0)
fracs = [k_parts[n][0] / k_total[0] for n in range(N_HARMONICS + 1)]

# ── Figure ───────────────────────────────────────────────────────

fig, (ax_k, ax_S) = plt.subplots(1, 2, figsize=(13, 4.4),
                                 gridspec_kw={'wspace': 0.22})

# Left panel: kernel
ax_k.plot(tau, k_total, color=COLOR_TOTAL, lw=2.2, label=r'total $k(\tau)$')
for n in range(N_HARMONICS + 1):
    ax_k.plot(tau, k_parts[n], color=COLORS_N[n], lw=1.7, label=LABELS_N[n])
ax_k.fill_between(tau, 0, k_parts[0], color=COLORS_N[0], alpha=0.10)

for m in range(1, int(tau_max / PARAMS["peq"]) + 1):
    ax_k.axvline(m * PARAMS["peq"], color='gray', ls=':', lw=0.8, alpha=0.3)
ax_k.axhline(0, color='gray', ls='-', lw=0.4)

x_ann, y_ann, dy = 0.97, 0.95, 0.10
for n in range(N_HARMONICS + 1):
    ax_k.text(x_ann, y_ann - n * dy, rf'$f_{n} = {fracs[n]:.1%}$'.replace('%', r'\%'),
              transform=ax_k.transAxes, fontsize=ann_fs, va='top', ha='right',
              color=COLORS_N[n])

ax_k.set_xlim(0, tau_max)
ax_k.set_xlabel(r'Time lag $\tau$ [days]', fontsize=label_fs)
ax_k.set_ylabel(r'$k(\tau)$', fontsize=label_fs)
ax_k.tick_params(labelsize=tick_fs)
ax_k.minorticks_on()
ax_k.legend(fontsize=legend_fs, loc='lower right', framealpha=0.9)

# Right panel: PSD
ax_S.plot(freq, S_total, color=COLOR_TOTAL, lw=2.2, label=r'total $S(f)$')
for n in range(N_HARMONICS + 1):
    ax_S.plot(freq, S_parts[n], color=COLORS_N[n], lw=1.7, label=LABELS_N[n])

for n in (1, 2):
    ax_S.axvline(n / PARAMS["peq"], color='gray', ls=':', lw=0.8, alpha=0.3)

ax_S.set_yscale('log')
ax_S.set_xlim(0, f_max)
ax_S.set_ylim(S_total.max() * 10**-4.5, S_total.max() * 4)
ax_S.set_xlabel(r'Frequency $f$ [day$^{-1}$]', fontsize=label_fs)
ax_S.set_ylabel(r'$S(f)$', fontsize=label_fs)
ax_S.tick_params(labelsize=tick_fs)
ax_S.minorticks_on()
ax_S.legend(fontsize=legend_fs, loc='upper right', framealpha=0.9)

# Parameter annotation
param_text = (rf'$P_{{\rm eq}} = {PARAMS["peq"]:g}$\,d, '
              rf'$\kappa = {PARAMS["kappa"]:g}$, '
              rf'$i = {np.degrees(PARAMS["inc"]):.0f}^\circ$, '
              rf'$\ell_{{\rm spot}} = {PARAMS["lspot"]:g}$\,d, '
              rf'$\tau_{{\rm spot}} = {PARAMS["tau_spot"]:g}$\,d, '
              rf'$\sigma_k = {PARAMS["sigma_k"]:g}$')
fig.text(0.5, -0.12, param_text, ha='center', fontsize=13,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.savefig(paths.figures / "fig_kernel_psd_harmonic_decomposition_single.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_kernel_psd_harmonic_decomposition_single.pdf")
