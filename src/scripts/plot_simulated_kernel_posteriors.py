"""
Posterior predictive kernel and PSD from the dynesty fits to simulated
lightcurves, comparing 1 year and 4 years of data at each true inclination.

Input:
    data/results_simulated_dynesty_mean/tsim_{365.0,1460.0}_tsamp_1.0_inc_{30.0,90.0}_peq_5.0/dynesty_results.npz
    data/results_simulated_dynesty_mean/tsim_{365.0,1460.0}_tsamp_1.0_inc_{30.0,90.0}_peq_5.0/simulated_data.npz
    (from mcmc_simulated_dynesty.py)

Output:
    figures/simulated_kernel_psd_posterior_90.pdf
    figures/simulated_kernel_psd_posterior_30.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import paths

from spotgp import (TimeSeriesData, TrapezoidSymmetricEnvelope, VisibilityFunction,
                    SpotEvolutionModel, GPSolver)


def kernel_posterior(tsim, inc_deg, n_draw=2000, seed=0, tsamp=1.0, peq=5.0,
                     results_dir=paths.data / "results_simulated_dynesty_mean",
                     tlags_kernel=np.linspace(0, 40, 401),
                     tlags_psd=np.arange(0, 200, 0.1)):
    """Kernel and PSD evaluated at the truth and at random posterior draws."""
    subdir = results_dir / f"tsim_{tsim}_tsamp_{tsamp}_inc_{inc_deg}_peq_{peq}"
    res = np.load(subdir / "dynesty_results.npz", allow_pickle=True)
    sim = np.load(subdir / "simulated_data.npz", allow_pickle=True)
    tp = res["true_params"].item()

    # rebuild the GP as mcmc_simulated_dynesty.py fit it
    model = SpotEvolutionModel(
        envelope=TrapezoidSymmetricEnvelope(lspot=tp["lspot"], tau_spot=tp["tau_spot"]),
        visibility=VisibilityFunction(peq=tp["peq"], kappa=tp["kappa"], inc=tp["inc"]),
        nspot_rate=tp["nspot_rate"], fspot=tp["fspot"], alpha_max=tp["alpha_max"])
    bounds = {
        "peq":      (0.9 * tp["peq"], 1.1 * tp["peq"]),
        "kappa":    (-1.0, 1.0),
        "inc":      (tp["inc"] - np.pi/6, tp["inc"] + np.pi/6),
        "lspot":    (10.0, 30.0),
        "tau_spot": (0.0, 10.0),
        "sigma_k":  (1e-3, 1e-2),
    }
    flux = sim["flux_obs"] - np.mean(sim["flux_obs"])
    ts = TimeSeriesData(x=sim["time"], y=flux, yerr=sim["yerr"], normalize=False)
    gp = GPSolver(ts, model, bounds)
    assert list(gp.param_keys) == [str(k) for k in res["param_keys"]]

    # dynesty samples are equally weighted; inc is in radians
    samples = res["samples"]
    idx = np.random.default_rng(seed).choice(len(samples), min(n_draw, len(samples)), replace=False)
    truth = [tp[k] for k in gp.param_keys]
    theta = np.vstack([truth, samples[idx]])

    tau, K = gp.compute_kernel_samples(theta, tlags=tlags_kernel)
    freq, P = gp.compute_kernel_samples(theta, kind="psd", tlags=tlags_psd)
    return dict(tau=tau, K_true=K[0], K=K[1:], freq=freq, P_true=P[0], P=P[1:])


def plot_band(ax, x, Y, color, label=None):
    """Posterior median with pointwise 68% and 95% intervals."""
    lo2, lo1, med, hi1, hi2 = np.percentile(Y, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(x, lo2, hi2, color=color, alpha=0.15, lw=0)
    ax.fill_between(x, lo1, hi1, color=color, alpha=0.35, lw=0)
    ax.plot(x, med, color=color, lw=1.5, label=label)


names = {365.0: "1 year of data", 1460.0: "4 years of data"}
f_nyq = 0.5 / 1.0  # data Nyquist frequency for tsamp = 1 d


def plot_kernel_psd_posterior(post, inc):
    """K(tau)/K(0) (left) and PSD (right), with residuals from the truth below."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 7), sharex="col",
                             gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05, wspace=0.25))
    (ax_k, ax_p), (ax_kr, ax_pr) = axes
    for tsim, color in zip((365.0, 1460.0), ("C0", "C1")):
        p = post[(inc, tsim)]
        acf = p["K"] / p["K"][:, :1]  # each draw normalized by its own K(0)
        acf_true = p["K_true"] / p["K_true"][0]
        plot_band(ax_k, p["tau"], acf, color, names[tsim])
        plot_band(ax_kr, p["tau"], acf - acf_true, color)
        plot_band(ax_p, p["freq"], p["P"], color, names[tsim])
        plot_band(ax_pr, p["freq"], p["P"] / p["P_true"] - 1, color)
    # both baselines are simulated from the same true parameters
    ax_k.plot(p["tau"], acf_true, "k--", lw=1.5, label="True")
    ax_p.plot(p["freq"], p["P_true"], "k--", lw=1.5, label="True")
    for ax in (ax_kr, ax_pr):
        ax.axhline(0, color="k", ls="--", lw=1.5)

    ax_kr.set_xlim(0, p["tau"][-1])
    ax_pr.set_xlim(0, f_nyq)
    ax_p.set_yscale("log")
    ax_p.set_ylim(1e-5 * p["P_true"].max(), 3 * p["P_true"].max())
    # the fractional residual spikes where P_true -> 0 at the PSD nulls
    ax_pr.set_ylim(-1, 1)
    ax_k.set_ylabel(r"$K(\tau)/K(0)$", fontsize=20)
    ax_p.set_ylabel("PSD", fontsize=20)
    ax_kr.set_ylabel(r"$\Delta\,K/K(0)$", fontsize=20)
    ax_pr.set_ylabel(r"$P/P_{\rm true} - 1$", fontsize=20)
    ax_kr.set_xlabel(r"$\tau$ [d]", fontsize=20)
    ax_pr.set_xlabel(r"Frequency [d$^{-1}$]", fontsize=20)
    for ax in axes.ravel():
        ax.tick_params(labelsize=14)
    for ax in (ax_kr, ax_pr):
        ax.yaxis.set_major_locator(MaxNLocator(3, symmetric=True))
    ax_k.legend(fontsize=16, frameon=False)
    fig.suptitle(rf"$i_{{\rm true}} = {inc:.0f}^\circ$", fontsize=24, y=0.96)
    return fig


post = {(inc, tsim): kernel_posterior(tsim, inc)
        for inc in (90.0, 30.0) for tsim in (365.0, 1460.0)}

for inc in (90.0, 30.0):
    fig = plot_kernel_psd_posterior(post, inc)
    fig.savefig(paths.figures / f"simulated_kernel_psd_posterior_{inc:.0f}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
