"""
KDE corner plots of the dynesty posteriors on simulated lightcurves, comparing
1 year and 4 years of data at each true inclination.

Input:
    data/results_simulated_dynesty_mean/tsim_{365.0,1460.0}_tsamp_1.0_inc_{30.0,90.0}_peq_5.0/dynesty_results.npz
    (from mcmc_simulated_dynesty.py)

Output:
    figures/simulated_corner_dynesty_kde_90.pdf
    figures/simulated_corner_dynesty_kde_30.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
import paths


def kde_corner(samples, labels, truths=None, levels=(0.393, 0.865),
               ngrid=100, bw_method=None, max_samples=5000, colors=None,
               tcolor="k", names=None, seed=0, labelsize=20, ticksize=14):
    """Corner plot with Gaussian-KDE contours and marginals.
    """
    if isinstance(samples, np.ndarray):
        samples, truths = [samples], [truths]
        names = None if names is None else [names]
    nset = len(samples)
    truths = [None] * nset if truths is None else truths
    colors = [f"C{k}" for k in range(nset)] if colors is None else colors

    rng = np.random.default_rng(seed)
    sets = []
    for s in samples:
        s = np.asarray(s)
        if len(s) > max_samples:  # KDE cost ~ N_samples * ngrid^2 per panel
            s = s[rng.choice(len(s), max_samples, replace=False)]
        sets.append(s)
    ndim = sets[0].shape[1]

    # shared axis limits covering every sample set
    lo = np.min([np.percentile(s, 0.5, axis=0) for s in sets], axis=0)
    hi = np.max([np.percentile(s, 99.5, axis=0) for s in sets], axis=0)
    pad = 0.1 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    grids = [np.linspace(lo[k], hi[k], ngrid) for k in range(ndim)]

    fig, axes = plt.subplots(ndim, ndim, figsize=(2.2 * ndim, 2.2 * ndim), squeeze=False)

    for s, tru, color in zip(sets, truths, colors):
        fills = [to_rgba(color, a) for a in np.linspace(0.3, 0.6, len(levels))]
        for i in range(ndim):
            for j in range(i + 1):
                ax = axes[i, j]
                if i == j:  # 1-D marginal
                    x = grids[i]
                    ax.plot(x, gaussian_kde(s[:, i], bw_method)(x), color=color)
                    if tru is not None:
                        ax.axvline(tru[i], color=tcolor, ls="--", lw=1)
                else:       # 2-D joint
                    X, Y = np.meshgrid(grids[j], grids[i])
                    kde = gaussian_kde(s[:, [j, i]].T, bw_method)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                    # density thresholds enclosing each probability mass
                    zs = np.sort(Z.ravel())[::-1]
                    cdf = np.cumsum(zs) / zs.sum()
                    thr = [zs[np.searchsorted(cdf, m)] for m in sorted(levels, reverse=True)]
                    ax.contourf(X, Y, Z, levels=thr + [Z.max()], colors=fills)
                    ax.contour(X, Y, Z, levels=thr, colors=color, linewidths=1)
                    if tru is not None:
                        ax.axvline(tru[j], color=tcolor, ls="--", lw=1)
                        ax.axhline(tru[i], color=tcolor, ls="--", lw=1)
                        ax.plot(tru[j], tru[i], "s", color=tcolor, ms=4)

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.set_ylim(bottom=0)
                ax.set_yticks([])
            else:
                ax.set_ylim(lo[i], hi[i])
            ax.set_xlim(lo[j], hi[j])
            ax.tick_params(labelsize=ticksize)
            ax.xaxis.set_major_locator(MaxNLocator(4, prune="lower"))
            if i != j:
                ax.yaxis.set_major_locator(MaxNLocator(4, prune="lower"))
            # axis labels only on the outer edge
            if i == ndim - 1:
                ax.set_xlabel(labels[j], fontsize=labelsize)
                ax.tick_params(axis="x", labelrotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=labelsize)
                ax.tick_params(axis="y", labelrotation=45)
            elif j > 0:
                ax.set_yticklabels([])

    if names is not None:
        handles = [Line2D([], [], color=c, label=n) for c, n in zip(colors, names)]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.9, 0.88),
                   fontsize=24, frameon=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    return fig


label_dict = {'peq': r"$P_{\rm eq}$ [d]",
              'kappa': r"$\kappa$",
              'inc': r"$I$ [deg]",
              'lspot': r"$\ell_{\rm spot}$ [d]",
              'tau_spot': r"$\tau_{\rm spot}$ [d]",
              'sigma_k': r"$\sigma_k$"}


def get_truths(res, param_keys):
    """True values in sample order (sigma_k is sampled as log10)."""
    ref = res["true_params"].item()
    ref["inc"] = np.rad2deg(ref["inc"])
    return [ref[k.rpartition(".")[2]] for k in param_keys]


results_dir = paths.data / "results_simulated_dynesty_mean"

for inc in [90, 30]:
    # allow_pickle: true_params is stored as a dict
    res1 = np.load(results_dir / f"tsim_365.0_tsamp_1.0_inc_{inc:.1f}_peq_5.0/dynesty_results.npz", allow_pickle=True)
    res2 = np.load(results_dir / f"tsim_1460.0_tsamp_1.0_inc_{inc:.1f}_peq_5.0/dynesty_results.npz", allow_pickle=True)

    samples1 = res1["samples"]
    samples2 = res2["samples"]
    # convert rad to deg
    samples1.T[2] = np.rad2deg(samples1.T[2])
    samples2.T[2] = np.rad2deg(samples2.T[2])

    param_keys = [str(k) for k in res1["param_keys"]]
    assert param_keys == [str(k) for k in res2["param_keys"]]
    labels = [label_dict[k] for k in param_keys]

    truths1, truths2 = get_truths(res1, param_keys), get_truths(res2, param_keys)

    fig = kde_corner([samples1, samples2], labels,
                     truths=[truths1, truths2],
                     names=["1 year of data", "4 years of data"])
    plt.text(0.3, 0.85, r"$i_{{\rm true}} = {}^\circ$".format(inc), ha="left", va="center", color="r", transform=fig.transFigure, fontsize=28)
    fig.savefig(paths.figures / f"simulated_corner_dynesty_kde_{inc}.pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
