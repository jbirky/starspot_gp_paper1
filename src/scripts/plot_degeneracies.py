"""
Illustrate degeneracies and non-uniqueness of starspot lightcurves.

Left panel  -- Degeneracy: different spot size / contrast combinations produce
               nearly identical single-spot lightcurves.
Right panel -- Non-uniqueness: 1 large spot vs 2 smaller spots yield similar
               lightcurves.

Output: figures/fig_degeneracies.pdf
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
legend_fs = 16
title_fs = 18

# Common settings
tsim = 30
tsamp = 0.02
peq = 5.0
inc = np.pi / 2  # edge-on

# Helper: build a model and override tmax / long / lat for precise control
def make_lc(nspot, alpha_max, fspot, tem, tdec, lspot, longs, lats, tmaxs,
            peq=peq, kappa=0, inc=inc, tsim=tsim, tsamp=tsamp):
    np.random.seed(0)
    lc = LightcurveModel(
        peq=peq, kappa=kappa, inc=inc, nspot=nspot,
        tem=tem, tdec=tdec, alpha_max=alpha_max, fspot=fspot, lspot=lspot,
        tsim=tsim, tsamp=tsamp,
    )
    # Override random spot positions and reference times
    lc.long = np.atleast_1d(longs)
    lc.lat = np.atleast_1d(lats)
    lc.tmax = np.atleast_1d(tmaxs)
    # Recompute lightcurve with the overridden parameters
    lc.flux = lc.Flux(lc.t)
    return lc


# ── Left panel: Degeneracy (spot size vs contrast) ──────────────
# Config A: larger spot, moderate contrast (fspot=0.3 → 70% dark)
# Config B: smaller spot, fully dark (fspot=0 → 100% dark)
# Matched so alpha^2 * (1-fspot) is the same:
#   A: 0.08^2 * 0.7 = 0.00448
#   B: alpha^2 * 1.0 = 0.00448 → alpha = 0.0669
tmax_single = tsim / 2
lc_a = make_lc(
    nspot=1, alpha_max=0.08, fspot=0.3, tem=3, tdec=3, lspot=10,
    longs=[np.pi], lats=[np.pi / 3], tmaxs=[tmax_single],
)
lc_b = make_lc(
    nspot=1, alpha_max=0.0669, fspot=0.0, tem=3, tdec=3, lspot=10,
    longs=[np.pi], lats=[np.pi / 3], tmaxs=[tmax_single],
)

# ── Right panel: Non-uniqueness (1 spot vs 2 spots) ────────────
inc2 = np.radians(80)
lc_c = make_lc(
    nspot=1, alpha_max=0.10, fspot=0, tem=2, tdec=2, lspot=15,
    longs=[np.pi], lats=[np.pi / 2.5], tmaxs=[tmax_single], inc=inc2,
)
lc_d = make_lc(
    nspot=2, alpha_max=0.07, fspot=0, tem=2, tdec=2, lspot=15,
    longs=[np.pi - 0.3, np.pi + 0.3],
    lats=[np.pi / 2.5, np.pi / 2.5],
    tmaxs=[tmax_single, tmax_single], inc=inc2,
)

# ── Figure (plot flux directly, no offset) ──────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(lc_a.t, lc_a.flux, 'C0', lw=1.5,
         label=r'$\alpha_{\max}=0.08$, $f_{\rm spot}=0.3$')
ax1.plot(lc_b.t, lc_b.flux, 'C1', lw=2, ls='--',
         label=r'$\alpha_{\max}=0.067$, $f_{\rm spot}=0$')
ax1.set_title(r'Degeneracy: spot size vs.\ contrast', fontsize=title_fs)
ax1.set_xlabel('Time [days]', fontsize=label_fs)
ax1.set_ylabel('Relative flux', fontsize=label_fs)
ax1.legend(fontsize=legend_fs, loc='lower left')
ax1.tick_params(labelsize=tick_fs)
ax1.ticklabel_format(useOffset=False)
ax1.minorticks_on()

ax2.plot(lc_c.t, lc_c.flux, 'C0', lw=1.5,
         label=r'1 spot ($\alpha_{\max}=0.10$)')
ax2.plot(lc_d.t, lc_d.flux, 'C1', lw=2, ls='--',
         label=r'2 spots ($\alpha_{\max}=0.07$ each)')
ax2.set_title(r'Degeneracy: 1 spot vs.\ 2 spots', fontsize=title_fs)
ax2.set_xlabel('Time [days]', fontsize=label_fs)
ax2.set_ylabel('Relative flux', fontsize=label_fs)
ax2.legend(fontsize=legend_fs, loc='lower left')
ax2.tick_params(labelsize=tick_fs)
ax2.ticklabel_format(useOffset=False)
ax2.minorticks_on()

fig.tight_layout()

outdir = paths.figures
outdir.mkdir(parents=True, exist_ok=True)
fig.savefig(outdir / "fig_degeneracies.pdf", bbox_inches="tight", dpi=300)
