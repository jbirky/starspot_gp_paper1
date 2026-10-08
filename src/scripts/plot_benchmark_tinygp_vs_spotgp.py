"""
Plot the GP solver benchmark: tinygp (celerite O(N) solver) vs george (full
O(N^3) solver) vs spotgp (banded and full Cholesky) for a one-sided exponential
envelope starspot kernel.

Input:
    data/benchmark_tinygp_vs_spotgp.npz
    (from compute_benchmark_tinygp_vs_spotgp.py, which needs a GPU)

Output:
    figures/fig_benchmark_tinygp_vs_spotgp.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

bench = np.load(paths.data / "benchmark_tinygp_vs_spotgp.npz")
print(f"Timings from {bench['date']} on {bench['gpu']} / {bench['cpu']}: {bench['versions']}")
TSIM_VALUES, TSAMP = bench["tsim"], float(bench["tsamp"])
n_harmonics = int(bench["n_harmonics"])
ndata_values = bench["ndata"]
times_tinygp = bench["times_tinygp"]
times_george = bench["times_george"]
times_spotgp_banded = bench["times_spotgp_banded"]
times_spotgp_full = bench["times_spotgp_full"]
bandwidths_banded = bench["bandwidths_banded"]

# ── Compute matrix storage memory ────────────────────────────────
BYTES_PER_FLOAT = 8
_MB = 1e6
J_CELERITE = 1 + 2 * n_harmonics

times_george = np.array(times_george, dtype=float)
times_spotgp_full = np.array(times_spotgp_full, dtype=float)
bandwidths_banded = np.array(bandwidths_banded, dtype=float)

mem_tinygp_mb = ndata_values * (J_CELERITE**2 + 3 * J_CELERITE + 1) * BYTES_PER_FLOAT / _MB
mem_george_mb = np.where(~np.isnan(times_george),
                         ndata_values.astype(float)**2 * BYTES_PER_FLOAT / _MB, np.nan)
mem_spotgp_banded_mb = ndata_values * (2 * bandwidths_banded + 1) * BYTES_PER_FLOAT / _MB
mem_spotgp_full_mb = np.where(~np.isnan(times_spotgp_full),
                              ndata_values.astype(float)**2 * BYTES_PER_FLOAT / _MB, np.nan)

# ── Plot ──────────────────────────────────────────────────────────
label_fs = 18
tick_fs = 14
legend_fs = 14
title_fs = 20
lw = 2.5

fig, (ax, ax_mem, ax2) = plt.subplots(3, 1, figsize=(8, 11),
                                       gridspec_kw={'height_ratios': [3, 1.5, 1]},
                                       sharex=True)

ax.loglog(ndata_values, times_tinygp, 'o-', color='g', lw=lw,
          markersize=7, label=r'tinygp (c\'{e}lerite, $\mathcal{O}(N)$)')

mask_george = ~np.isnan(times_george)
if np.any(mask_george):
    ax.loglog(ndata_values[mask_george],
              np.array(times_george)[mask_george],
              '^-', color='r', lw=lw, markersize=7,
              label=r'george ($\mathcal{O}(N^3)$)')

ax.loglog(ndata_values, times_spotgp_banded, 's-', color='darkblue', lw=lw,
          markersize=7, label=r'spotgp banded ($\mathcal{O}(Nb^2)$)')

# mask_eo = ~np.isnan(times_spotgp_edgeon_banded)
# if np.any(mask_eo):
#     ax.loglog(ndata_values[mask_eo],
#               np.array(times_spotgp_edgeon_banded)[mask_eo],
#               'P-', color='steelblue', lw=lw, markersize=7,
#               label=r'spotgp edge-on banded ($\mathcal{O}(Nb^2)$)')

# Only plot full Cholesky where we have data
mask_full = ~np.isnan(times_spotgp_full)
if np.any(mask_full):
    ax.loglog(ndata_values[mask_full],
              np.array(times_spotgp_full)[mask_full],
              'D-', color='teal', lw=lw, markersize=7,
              label=r'spotgp full ($\mathcal{O}(N^3)$)')

# Reference scaling lines
n_ref = ndata_values.astype(float)
t_ref = times_tinygp[0] * (n_ref / n_ref[0])
ax.loglog(n_ref, t_ref, ':', color='gray', lw=1, label=r'$\propto N$')

# Use george or spotgp full for N^3 reference (whichever has data)
t_ref3_base = times_george[0] if not np.isnan(times_george[0]) else times_spotgp_full[0]
t_ref3 = t_ref3_base * (n_ref / n_ref[0])**3
ax.loglog(n_ref[n_ref <= 4000], t_ref3[n_ref <= 4000],
          '--', color='gray', lw=1, label=r'$\propto N^3$')

ax.axhline(1.0, color='gray', ls='-', lw=0.8, alpha=0.4)
ax.text(ndata_values[-1] * 1.2, 1.0, '1 s', fontsize=tick_fs, va='center', color='darkgray')
ax.axhline(60.0, color='gray', ls='-', lw=0.8, alpha=0.4)
ax.text(ndata_values[-1] * 1.2, 60.0, '1 min', fontsize=tick_fs, va='center', color='darkgray')

ax.set_ylabel('Log-likelihood evaluation time [s]', fontsize=label_fs)
ax.set_title('GP Solver Speed Comparison', fontsize=title_fs)
ax.legend(fontsize=legend_fs, loc='upper left')
ax.tick_params(labelsize=tick_fs)
ax.grid(True, alpha=0.3, which='both')
ax.minorticks_on()

# ── Memory usage subplot ─────────────────────────────────────────
ax_mem.loglog(ndata_values, mem_tinygp_mb, 'o-', color='g', lw=lw,
              markersize=7, label=r"tinygp (c\'{e}lerite)")

mask_george_mem = ~np.isnan(mem_george_mb)
if np.any(mask_george_mem):
    ax_mem.loglog(ndata_values[mask_george_mem],
                  mem_george_mb[mask_george_mem],
                  '^-', color='r', lw=lw, markersize=7, label='george')

ax_mem.loglog(ndata_values, mem_spotgp_banded_mb, 's-', color='darkblue', lw=lw,
              markersize=7, label='spotgp banded')

mask_full_mem = ~np.isnan(mem_spotgp_full_mb)
if np.any(mask_full_mem):
    ax_mem.loglog(ndata_values[mask_full_mem],
                  mem_spotgp_full_mb[mask_full_mem],
                  'D-', color='teal', lw=lw, markersize=7, label='spotgp full')

ax_mem.set_ylabel('Matrix storage [MB]', fontsize=label_fs)
ax_mem.legend(fontsize=legend_fs - 2, loc='upper left')
ax_mem.tick_params(labelsize=tick_fs)
ax_mem.grid(True, alpha=0.3, which='both')
ax_mem.minorticks_on()

# ── Sparsity fraction subplot ────────────────────────────────────
bandwidths_banded = np.array(bandwidths_banded, dtype=float)
# bandwidths_edgeon = np.array(bandwidths_edgeon, dtype=float)

mask_b = ~np.isnan(bandwidths_banded)
sparsity_banded = (2 * bandwidths_banded[mask_b] + 1) / ndata_values[mask_b]
ax2.semilogx(ndata_values[mask_b], sparsity_banded, 's-', color='darkblue',
             lw=lw, markersize=7, label='spotgp banded')

# mask_eo_b = ~np.isnan(bandwidths_edgeon)
# sparsity_edgeon = (2 * bandwidths_edgeon[mask_eo_b] + 1) / ndata_values[mask_eo_b]
# ax2.semilogx(ndata_values[mask_eo_b], sparsity_edgeon, 'P-', color='steelblue',
#              lw=lw, markersize=7, label='spotgp edge-on banded')

ax2.axhline(1.0, color='gray', ls='-', lw=0.8, alpha=0.4)
ax2.set_ylim(-0.05, 1.1)
ax2.set_xlabel('Number of datapoints $N$', fontsize=label_fs)
ax2.set_ylabel('Sparsity fraction \n$(2b+1)/N$', fontsize=label_fs)
ax2.legend(fontsize=legend_fs - 2, loc='upper right')
ax2.tick_params(labelsize=tick_fs)
ax2.grid(True, alpha=0.3, which='both')
ax2.minorticks_on()

ax2.set_xlim(TSIM_VALUES[0] / TSAMP * 0.8, TSIM_VALUES[-1] / TSAMP * 1.2)

fig.tight_layout()
outpath = paths.figures / "fig_benchmark_tinygp_vs_spotgp.pdf"
fig.savefig(outpath, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"\nSaved figure to {outpath}")
