"""
Predictive mean and variance of the GP at the posterior-mean hyperparameters of the
dynesty fits to simulated lightcurves, for true inclinations of 90 and 30 deg with 1 and
4 years of data.

The predictive distribution is that of the noise-free flux, conditioned on all the data
of each fit. Every panel shows the same window, so the four fits are compared at the
same resolution, with the observations. Below each, the residual of the predictive mean
from the true (noise-free) simulated flux, in units of the predictive standard deviation,
at the observed times.
The coverage of the true flux by the 1 and 2 sigma intervals over the whole baseline,
printed for each fit, tests whether the predictive variance is calibrated.

Input:
    data/results_simulated_dynesty_mean/tsim_{365.0,1460.0}_tsamp_1.0_inc_{30.0,90.0}_peq_5.0/dynesty_results.npz
    data/results_simulated_dynesty_mean/tsim_{365.0,1460.0}_tsamp_1.0_inc_{30.0,90.0}_peq_5.0/simulated_data.npz
    (from compute_mcmc_simulated_dynesty.py)

Output:
    figures/simulated_predictive.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import paths

from spotgp import (TimeSeriesData, TrapezoidSymmetricEnvelope, VisibilityFunction,
                    SpotEvolutionModel, GPSolver)

RESULTS = paths.data / "results_simulated_dynesty_mean"
FITS = [(90.0, 365.0), (90.0, 1460.0), (30.0, 365.0), (30.0, 1460.0)]  # (true inclination [deg], baseline [d])
WINDOW = (200.0, 260.0)   # days shown, inside both baselines
DT_PRED = 0.05            # spacing of the prediction grid [d]
CHUNK = 800               # prediction points per solve, to bound the memory of the cross-covariance
PPT = 1e3                 # plotted flux unit: parts per thousand
names = {365.0: "1 year of data", 1460.0: "4 years of data"}
colors = {365.0: "C0", 1460.0: "C1"}


def load_fit(tsim, inc_deg, tsamp=1.0, peq=5.0):
    """The GP as compute_mcmc_simulated_dynesty.py fit it, the simulated data, and the
    posterior mean of the hyperparameters."""
    subdir = RESULTS / f"tsim_{tsim}_tsamp_{tsamp}_inc_{inc_deg}_peq_{peq}"
    res = np.load(subdir / "dynesty_results.npz", allow_pickle=True)
    sim = np.load(subdir / "simulated_data.npz", allow_pickle=True)
    tp = res["true_params"].item()

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
    offset = np.mean(sim["flux_obs"])  # the fit was to the mean-subtracted flux
    ts = TimeSeriesData(x=sim["time"], y=sim["flux_obs"] - offset, yerr=sim["yerr"], normalize=False)
    gp = GPSolver(ts, model, bounds)
    assert list(gp.param_keys) == [str(k) for k in res["param_keys"]]

    # dynesty samples are equally weighted, in the solver's parameter order (inc in radians)
    theta = res["samples"].mean(axis=0)
    data = dict(time=sim["time"], flux=sim["flux_obs"] - offset, yerr=sim["yerr"],
                flux_true=sim["flux_true"] - offset)
    return gp, data, theta


def predict(gp, theta, x):
    """Predictive mean and variance of the noise-free flux at x, in chunks."""
    parts = [gp.predict_at(theta, x[i:i + CHUNK]) for i in range(0, len(x), CHUNK)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# one block per fit (rows: inclination, columns: baseline): flux above its residuals
fig = plt.figure(figsize=(16, 16))
blocks = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.05)
axes = []
for block in blocks:
    grid = block.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    axes += [fig.add_subplot(grid[i]) for i in range(2)]
axes = np.array(axes)
# every panel shares the time axis; the flux panels share one flux range, the residuals another
for ax in axes[1:]:
    ax.sharex(axes[0])
for panels in (axes[0::2], axes[1::2]):
    for ax in panels[1:]:
        ax.sharey(panels[0])
xpred = np.arange(WINDOW[0], WINDOW[1] + DT_PRED / 2, DT_PRED)

for i, ((inc_deg, tsim), ax, ax_res) in enumerate(zip(FITS, axes[0::2], axes[1::2])):
    left, bottom = i % 2 == 0, i // 2 == 1  # FITS fills the 2x2 grid row by row
    gp, data, theta = load_fit(tsim, inc_deg)
    color = colors[tsim]

    # in the window, on a fine grid, and at every observed time for the residuals and coverage
    mu, var = predict(gp, theta, xpred)
    mu_obs, var_obs = predict(gp, theta, data["time"])
    sd, sd_obs = np.sqrt(np.clip(var, 0, None)), np.sqrt(np.clip(var_obs, 0, None))
    z = (mu_obs - data["flux_true"]) / sd_obs
    within1, within2 = np.mean(np.abs(z) < 1), np.mean(np.abs(z) < 2)

    ax.fill_between(xpred, PPT * (mu - 2 * sd), PPT * (mu + 2 * sd), color=color, alpha=0.15, lw=0,
                    label=r"$\pm 2\sigma$")
    ax.fill_between(xpred, PPT * (mu - sd), PPT * (mu + sd), color=color, alpha=0.35, lw=0,
                    label=r"$\pm 1\sigma$")
    ax.plot(xpred, PPT * mu, color=color, lw=1.2, label="predictive mean")
    shown = (data["time"] >= WINDOW[0]) & (data["time"] <= WINDOW[1])
    ax.errorbar(data["time"][shown], PPT * data["flux"][shown], yerr=PPT * data["yerr"][shown], fmt="o", ms=3.5,
                mfc="none", mec="k", mew=0.8, ecolor="k", elinewidth=0.8, capsize=0, label="observed")
    if left:
        ax.set_ylabel("Flux [ppt]", fontsize=20)
    else:
        ax.tick_params(labelleft=False)
    names_theta = dict(zip(gp.param_keys, theta))
    ax.set_title(rf"$I = {inc_deg:.0f}^\circ$, {names[tsim]}" + "\n" + rf"posterior mean $P_{{\rm eq}}$ = "
                 rf"{names_theta['peq']:.3f} d, $\kappa$ = {names_theta['kappa']:.2f}, "
                 rf"$I$ = {np.degrees(names_theta['inc']):.1f}$^\circ$", fontsize=18, loc="left")
    ax.tick_params(labelbottom=False)

    ax_res.axhspan(-2, 2, color=color, alpha=0.15, lw=0)
    ax_res.axhspan(-1, 1, color=color, alpha=0.35, lw=0)
    ax_res.axhline(0, color=color, lw=1)
    ax_res.plot(data["time"][shown], z[shown], "o", ms=3, color="k")
    ax_res.set_ylim(-4, 4)
    ax_res.set_yticks([-2, 0, 2])
    if left:
        ax_res.set_ylabel(r"$\Delta / \sigma$", fontsize=20)
    else:
        ax_res.tick_params(labelleft=False)
    if bottom:
        ax_res.set_xlabel("Time [days]", fontsize=20)
    else:
        ax_res.tick_params(labelbottom=False)
    print(f"I={inc_deg:.0f}, tsim={tsim:.0f}: within 1 sigma {within1:.1%}, 2 sigma {within2:.1%}, "
          f"median predictive sd {np.median(sd_obs):.2e} at observed times, {np.median(sd):.2e} on the grid")

fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=5,
           fontsize=16, frameon=False)
axes[-1].set_xlim(*WINDOW)
fig.savefig(paths.figures / "simulated_predictive.pdf", bbox_inches="tight")
