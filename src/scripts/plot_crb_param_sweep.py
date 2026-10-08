"""
Corner plots of the fractional Cramer-Rao bound across the Sobol parameter sweep.

Each panel bins the sweep in a pair of the derived variables and colors it by
the mean of log10(sigma_crb / |true|) for one parameter.

Input:
    data/crb_sobol_sweep_365_4096.csv   (from crb_compute_parameter_sweep.py)

Output:
    figures/crb_sweep_corner_contour_kappa.png
    figures/crb_sweep_corner_contour_inc.png
    figures/crb_sweep_corner_contour_peq.png
    figures/crb_sweep_corner_contour_sigma_k.png
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
import paths

# ``import paths`` imports spotgp, which switches matplotlib to the 'classic'
# style; these figures were made with the default style.
plt.rcdefaults()
plt.rcParams.update({
    "font.size":        16,   # base font size
    "axes.titlesize":   20,   # axes title
    "axes.labelsize":   22,   # x/y axis labels
    "xtick.labelsize":  16,   # x tick labels
    "ytick.labelsize":  16,   # y tick labels
    "legend.fontsize":  14,
    "figure.titlesize": 22,   # suptitle
    "axes.formatter.useoffset": False,  # disable scientific notation offset
    "text.usetex": True,
    "font.family": "serif",
})


def corner_variables(sims):
    """Derived quantities shown on the corner plot axes."""
    return [
        (r"$P_{\rm eq}$",                      sims["true_peq"]),
        (r"$l_{\rm spot} + 2\tau_{\rm spot}$", sims["true_lspot"] + 2*sims["true_tau_spot"]),
        (r"$i\ \rm (deg)$",                    sims["true_inc"] * 180/np.pi),
        (r"$\kappa$",                          sims["true_kappa"]),
        (r"$\sigma_k$",                        sims["true_sigma_k"]),
    ]


def plot_crb_corner(sims, param, title, cbar_label, savename,
                    bins=20, vmin=-2, vmax=1, smooth=1.5, minorticks=False):
    """Corner plot of the fractional CRB uncertainty on ``param``."""
    variables = corner_variables(sims)
    n = len(variables)
    z = np.log10(sims[f"sigma_{param}"] / np.abs(sims[f"true_{param}"]))
    levels = np.linspace(vmin, vmax, 20)

    tick_labels    = [1, 10, 100, 1000]
    tick_positions = [np.log10(v / 100) for v in tick_labels]

    fig, axes = plt.subplots(n-1, n-1, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    last_cf = None
    for row in range(n-1):
        for col in range(n-1):
            ax = axes[row, col]
            if col <= row:
                x_label, x_data = variables[col]
                y_label, y_data = variables[row+1]
                stat, xedges, yedges, _ = binned_statistic_2d(
                    x_data, y_data, z, statistic='mean', bins=bins
                )
                stat_smooth = gaussian_filter(
                    np.where(np.isnan(stat), np.nanmean(stat), stat), sigma=smooth
                )
                # bin centres for contour
                xc = 0.5 * (xedges[:-1] + xedges[1:])
                yc = 0.5 * (yedges[:-1] + yedges[1:])
                last_cf = ax.contourf(xc, yc, stat_smooth.T, levels=levels,
                                      cmap="RdBu_r", vmin=vmin, vmax=vmax, extend="both")
                ax.contour(xc, yc, stat_smooth.T, levels=levels,
                           colors="k", linewidths=0.4, alpha=0.4)
                if col == 0:
                    ax.set_ylabel(y_label)
                else:
                    ax.tick_params(labelleft=False)
                if row == n-2:
                    ax.set_xlabel(x_label)
                else:
                    ax.tick_params(labelbottom=False)
            else:
                ax.set_visible(False)

    plt.suptitle(title, y=0.95, fontsize=24)
    cbar_ax = fig.add_axes([0.8, 0.45, 0.02, 0.4])
    cbar = fig.colorbar(last_cf, cax=cbar_ax, label=cbar_label)
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels([str(v) for v in tick_labels])
    if minorticks:
        cbar_ax.minorticks_on()
    fig.savefig(paths.figures / savename, dpi=300, bbox_inches='tight')
    plt.close(fig)


sims = pd.read_csv(paths.data / "crb_sobol_sweep_365_4096.csv")

corner_panels = [
    dict(param="kappa",
         title=r"$\kappa$ uncertainty limit",
         cbar_label=r'$\% \, \rm uncertainty = \kappa_{crb} / |\kappa_{\rm true}| \times 100$',
         savename="crb_sweep_corner_contour_kappa.png"),
    dict(param="inc",
         title=r"$i$ uncertainty limit",
         cbar_label=r'$\% \, \rm uncertainty = i_{\rm crb} / |i_{\rm true}| \times 100$',
         savename="crb_sweep_corner_contour_inc.png"),
    dict(param="peq",
         title=r"$P_{\rm eq}$ uncertainty limit",
         cbar_label=r'$\% \, \rm uncertainty = P_{\rm eq, crb} / |P_{\rm eq, true}| \times 100$',
         savename="crb_sweep_corner_contour_peq.png"),
    dict(param="sigma_k",
         title=r"$\sigma_k$ uncertainty limit",
         cbar_label=r'$\% \, \rm uncertainty = \sigma_{\rm k, crb} / |\sigma_{\rm k, true}| \times 100$',
         savename="crb_sweep_corner_contour_sigma_k.png",
         minorticks=True),
]

for panel in corner_panels:
    plot_crb_corner(sims, **panel)
