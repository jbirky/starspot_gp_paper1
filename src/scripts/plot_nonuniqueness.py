"""
Non-uniqueness: identical stellar and spot population parameters produce
different lightcurves due to the random positions and emergence times of
individual spots.  Each realization draws spot longitudes, latitudes, and
reference times from the same distributions, yet the resulting lightcurves
differ — motivating a statistical (GP) approach.

Output: figures/fig_nonuniqueness.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths
from spotgp.lightcurve import LightcurveModel

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True

label_fs = 20
tick_fs = 18
legend_fs = 15
title_fs = 18

# ── Shared parameters ───────────────────────────────────────────
params = dict(
    peq=5.0,
    kappa=0.0,
    inc=np.radians(80),
    nspot=10,
    tem=3,
    tdec=3,
    alpha_max=0.06,
    fspot=0,
    lspot=8,
    long=[0, 2 * np.pi],
    lat=[np.pi / 6, np.pi / 2],
    tsim=40,
    tsamp=0.02,
)

n_realizations = 4
seeds = [0, 1, 2, 3]
colors = ['C0', 'C1', 'C2', 'C3']

# ── Generate lightcurves ────────────────────────────────────────
lightcurves = []
for seed in seeds:
    np.random.seed(seed)
    lc = LightcurveModel(**params)
    lightcurves.append(lc)

# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(n_realizations, 1, figsize=(12, 8), sharex=True)

for i, (lc, ax, col) in enumerate(zip(lightcurves, axes, colors)):
    ax.plot(lc.t, lc.flux, color=col, lw=1.0)
    ax.set_yticks([])
    ax.tick_params(labelsize=tick_fs - 2)
    ax.minorticks_on()
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.text(0.98, 0.92, f'Realization {i + 1}',
            transform=ax.transAxes, fontsize=legend_fs,
            ha='right', va='top', color=col)
    ax.set_ylim(0.993, 1.001)

axes[-1].set_xlabel('Time [days]', fontsize=label_fs)
axes[0].set_title('Same parameters, different spot configurations', fontsize=title_fs)

fig.supylabel('Normalized Flux', fontsize=label_fs)

fig.tight_layout()
fig.subplots_adjust(hspace=0.08, left=0.07)

outdir = paths.figures
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "fig_nonuniqueness.pdf", bbox_inches="tight", dpi=300)
