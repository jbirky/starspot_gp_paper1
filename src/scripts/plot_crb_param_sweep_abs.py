"""
Corner plots of the absolute Cramer-Rao bound across the Sobol parameter sweep.

Companion to plot_crb_param_sweep.py, which shows the fractional uncertainty
sigma_crb / |true|.  The fractional uncertainty diverges as the true value
approaches zero (kappa ~ 0, i ~ 0), so here each panel is colored by the mean of
log10(sigma_crb) in the parameter's own units.  The two sparse end bins of
l_spot + 2 tau_spot are left out.  The smoothing uses only bins that contain
samples; empty bins are masked and drawn gray.

Input:
    data/crb_sobol_sweep_365_4096.csv   (from compute_crb_compute_parameter_sweep.py)

Output:
    figures/crb_sweep_corner_contour_abs_kappa.png
    figures/crb_sweep_corner_contour_abs_inc.png
    figures/crb_sweep_corner_contour_abs_peq.png
    figures/crb_sweep_corner_contour_abs_sigma_k.png
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
    """Derived quantities shown on the corner plot axes, and whether to trim
    their end bins.  l_spot + 2 tau_spot is a sum of two uniform draws, so its
    density falls to zero at both ends and its end bins are nearly empty."""
    return [
        (r"$P_{\rm eq}$",                      sims["true_peq"],                             False),
        (r"$l_{\rm spot} + 2\tau_{\rm spot}$", sims["true_lspot"] + 2*sims["true_tau_spot"], True),
        (r"$i\ \rm (deg)$",                    sims["true_inc"] * 180/np.pi,                 False),
        (r"$\kappa$",                          sims["true_kappa"],                           False),
        (r"$\sigma_k$",                        sims["true_sigma_k"],                         False),
    ]


def bin_edges(data, bins, trim):
    """Equal-width bin edges over the data range; ``trim`` drops the two end bins."""
    edges = np.linspace(data.min(), data.max(), bins + 1)
    return edges[1:-1] if trim else edges


def plot_crb_corner(sims, param, title, cbar_label, savename, vmin, vmax,
                    scale=1.0, bins=20, smooth=1.5, minorticks=False):
    """Corner plot of the absolute CRB uncertainty on ``param``, times ``scale``."""
    variables = corner_variables(sims)
    n = len(variables)
    z = np.log10(scale * sims[f"sigma_{param}"])
    levels = np.linspace(vmin, vmax, 20)

    tick_positions = np.arange(np.ceil(vmin), np.floor(vmax) + 1)
    tick_labels    = [f"{10.0**p:g}" for p in tick_positions]

    fig, axes = plt.subplots(n-1, n-1, figsize=(12, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    last_cf = None
    for row in range(n-1):
        for col in range(n-1):
            ax = axes[row, col]
            if col <= row:
                x_label, x_data, x_trim = variables[col]
                y_label, y_data, y_trim = variables[row+1]
                stat, xedges, yedges, _ = binned_statistic_2d(
                    x_data, y_data, z, statistic='mean',
                    bins=[bin_edges(x_data, bins, x_trim), bin_edges(y_data, bins, y_trim)]
                )
                # smooth over the occupied bins only, then mask the empty ones
                filled = np.isfinite(stat)
                num = gaussian_filter(np.where(filled, stat, 0.0), sigma=smooth)
                den = gaussian_filter(filled.astype(float), sigma=smooth)
                stat_smooth = np.ma.masked_array(num / den, mask=~filled)
                # bin centres for contour
                xc = 0.5 * (xedges[:-1] + xedges[1:])
                yc = 0.5 * (yedges[:-1] + yedges[1:])
                ax.set_facecolor("0.8")   # empty bins, distinct from the white mid-color
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
    cbar.set_ticklabels(tick_labels)
    if minorticks:
        cbar_ax.minorticks_on()
    fig.savefig(paths.figures / savename, dpi=300, bbox_inches='tight')
    plt.close(fig)


sims = pd.read_csv(paths.data / "crb_sobol_sweep_365_4096.csv")

# Color ranges span roughly the 1st-99th percentiles of each parameter's sigma_crb.
corner_panels = [
    dict(param="kappa", vmin=-3, vmax=1,
         title=r"$\kappa$ absolute uncertainty limit",
         cbar_label=r"$\kappa_{\rm crb}$",
         savename="crb_sweep_corner_contour_abs_kappa.png"),
    dict(param="inc", vmin=-1, vmax=2, scale=180/np.pi,
         title=r"$i$ absolute uncertainty limit",
         cbar_label=r"$i_{\rm crb}$ (deg)",
         savename="crb_sweep_corner_contour_abs_inc.png"),
    dict(param="peq", vmin=-4, vmax=2,
         title=r"$P_{\rm eq}$ absolute uncertainty limit",
         cbar_label=r"$P_{\rm eq, crb}$ (d)",
         savename="crb_sweep_corner_contour_abs_peq.png"),
    dict(param="sigma_k", vmin=-3, vmax=-1,
         title=r"$\sigma_k$ absolute uncertainty limit",
         cbar_label=r"$\sigma_{\rm k, crb}$",
         savename="crb_sweep_corner_contour_abs_sigma_k.png",
         minorticks=True),
]

for panel in corner_panels:
    plot_crb_corner(sims, **panel)
