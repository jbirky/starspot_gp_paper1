"""
Analytic vs numerical kernel comparison across a parameter grid.

For each swept parameter (Peq, kappa, I, lspot, tau_spot), test 3 different
values while holding the others at their baseline.  Produces:

  1. A LaTeX table of MSE values with and without limb darkening.
  2. A single figure: a 2-column by 3-row grid with one cell per parameter
     sweep, each cell showing the ACF comparison (numerical solid, analytic
     dashed) above its residual.  The spare cell carries the line-style
     legend.  No PSD panels.

Output:
    figures/fig_kernel_comparison_grid.pdf
    output/kernel_accuracy_benchmark.csv
    output/kernel_accuracy_benchmark.tex
"""
import numpy as np

# Compat shim: np.trapezoid was added in NumPy 2.0; older versions have np.trapz
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import paths

from spotgp.analytic_kernel import AnalyticKernel
from spotgp.numerical_kernel import NumericalKernel, generate_sims, avg_covariance_tlag

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

# ── Baseline parameters ─────────────────────────────────────────
baseline = dict(
    peq=5.0,
    kappa=0.0,
    inc=np.pi / 2,
    nspot=10,
    lspot=15.0,
    tau_spot=3.0,
    alpha_max=0.1,
    fspot=0.0,
)

# Simulation settings
TSIM = 200
TSAMP = 0.05
NSIM = 500
N_HARMONICS = 3
TMAX_ACF = 20  # plotted lag range [days]

# ── Parameter sweeps ────────────────────────────────────────────
# Each entry: (key, [3 values], latex label, ACF tmax multiplier)
sweeps = [
    ("peq",   [3.0, 5.0, 10.0],                              r"$P_{\rm eq}$",          8),
    ("kappa", [-1.0, 0., 1.0],                                r"$\kappa$",              8),
    ("inc",   [np.radians(30), np.radians(60), np.radians(90)], r"$I$",                 8),
    ("lspot", [5.0, 15.0, 30.0],                              r"$\ell_{\rm spot}$",     8),
    ("tau_spot", [1.0, 3.0, 8.0],                              r"$\tau_{\rm spot}$",     8),
]

# Nice value labels for the legend
def _val_label(key, val):
    if key == "inc":
        return rf"${np.degrees(val):.0f}^\circ$"
    elif key == "kappa":
        return rf"${val:.1f}$"
    elif key == "alpha_max":
        return rf"${val:.2f}$"
    else:
        return rf"${val:g}$"


def _compute_numerical_acf_with_limb_darkening(params, tsim, tsamp, nsim):
    """Compute numerical ACF with limb darkening enabled.

    Uses generate_sims directly since NumericalKernel does not expose
    the limb_darkening flag.
    """
    peq = params["peq"]
    kappa = params["kappa"]
    inc = params["inc"]
    nspot = params["nspot"]
    lspot = params["lspot"]
    tau = params["tau_spot"]
    alpha_max = params["alpha_max"]

    theta = np.array([peq, kappa, inc, nspot])
    fluxes = generate_sims(
        theta, nsim=nsim,
        tem=tau, tdec=tau, alpha_max=alpha_max,
        lspot=lspot, tsim=tsim, tsamp=tsamp,
        limb_darkening=True,
    )
    cov_matrix = np.cov(fluxes.T)
    autocov = avg_covariance_tlag(cov_matrix)
    autocor = autocov / autocov[0] if autocov[0] != 0 else autocov
    tarr = np.arange(0, tsim, tsamp)
    return tarr, autocor


colors = ['C0', 'C1', 'C2']
label_fs = 14
tick_fs = 12
legend_fs = 11
title_fs = 15

# ── Run all comparisons ────────────────────────────────────────
rows = []  # for CSV / LaTeX
results = []  # for plotting

n_sweeps = len(sweeps)

for key, vals, latex, tmax_mult in sweeps:
    sweep_results = []
    for val in vals:
        params = baseline.copy()
        params[key] = val

        peq = params["peq"]
        tmax_acf = tmax_mult * peq

        print(f"  {key} = {val}  (peq={peq})")

        # --- Numerical kernel (no limb darkening) ---
        print(f"    Computing numerical kernel (nsim={NSIM})...")
        nk = NumericalKernel(params, tsim=TSIM, tsamp=TSAMP, nsim=NSIM, verbose=False)
        t_num, acf_num = nk.get_acf()

        # --- Numerical kernel (with limb darkening) ---
        print(f"    Computing numerical kernel with limb darkening (nsim={NSIM})...")
        t_num_ld, acf_num_ld = _compute_numerical_acf_with_limb_darkening(
            params, tsim=TSIM, tsamp=TSAMP, nsim=NSIM
        )

        # --- Analytic kernel ---
        print(f"    Computing analytic kernel...")
        ak = AnalyticKernel(params, n_harmonics=N_HARMONICS)
        tau_arr = np.arange(0, TSIM, TSAMP)
        k_ana = ak.kernel(tau_arr)
        acf_ana = k_ana / k_ana[0] if k_ana[0] != 0 else k_ana

        # MSE over plotted range (no limb darkening)
        idx = tau_arr < tmax_acf
        mse = np.mean((acf_num[idx] - acf_ana[idx]) ** 2)

        # MSE over plotted range (with limb darkening)
        mse_ld = np.mean((acf_num_ld[idx] - acf_ana[idx]) ** 2)

        print(f"    MSE (no LD) = {mse:.2e},  MSE (LD) = {mse_ld:.2e}")

        # Save row
        row = {
            "peq": params["peq"],
            "kappa": params["kappa"],
            "inc_deg": np.degrees(params["inc"]),
            "nspot": params["nspot"],
            "lspot": params["lspot"],
            "tau_spot": params["tau_spot"],
            "alpha_max": params["alpha_max"],
            "varied_param": key,
            "mse_acf": mse,
            "mse_acf_ld": mse_ld,
        }
        rows.append(row)

        sweep_results.append(dict(
            val=val,
            label=_val_label(key, val),
            t_num=t_num, acf_num=acf_num,
            tau_arr=tau_arr, acf_ana=acf_ana,
            tmax_acf=tmax_acf, peq=peq, mse=mse, mse_ld=mse_ld,
        ))

    results.append((key, latex, sweep_results))

# ── Save CSV ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = paths.output / "kernel_accuracy_benchmark.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False, float_format="%.6g")
print(f"\nSaved CSV to {csv_path}")

# ── Save LaTeX table ────────────────────────────────────────────

# Column display names for the varied parameter
_col_latex = {
    "peq": r"$P_{\rm eq}$",
    "kappa": r"$\kappa$",
    "inc": r"$I$",
    "lspot": r"$\ell_{\rm spot}$",
    "tau_spot": r"$\tau_{\rm spot}$",
    "alpha_max": r"$\alpha_{\rm max}$",
}


def _mse_to_latex(val):
    """Format a float in scientific notation as LaTeX: $1.23 \\times 10^{-4}$."""
    if val == 0:
        return "$0$"
    exp = int(np.floor(np.log10(np.abs(val))))
    coeff = val / 10**exp
    return rf"${coeff:.2f} \times 10^{{{exp}}}$"


tex_lines = []
tex_lines.append(r"\begin{table}[ht!]")
tex_lines.append(r"\centering")
tex_lines.append(r"\caption{Mean squared error (MSE) of the analytic kernel ACF compared to numerical Monte Carlo simulations, including stellar limb darkening.}")
tex_lines.append(r"\label{tab:kernel_mse}")
tex_lines.append(r"\begin{tabular}{llc}")
tex_lines.append(r"\hline")
tex_lines.append(r"Varied param. & Value & MSE \\")
tex_lines.append(r"\hline")

# Map sweep key -> CSV column name
_key_to_col = {"inc": "inc_deg"}

prev_key = None
for _, row in df.iterrows():
    key = row["varied_param"]
    col = _key_to_col.get(key, key)
    val = row[col]

    # Format the value (inc_deg is already in degrees)
    if key == "inc":
        val_str = rf"${val:.0f}^\circ$"
    elif key == "kappa":
        val_str = rf"${val:.1f}$"
    elif key == "alpha_max":
        val_str = rf"${val:.2f}$"
    else:
        val_str = rf"${val:g}$"

    param_label = _col_latex.get(key, key) if key != prev_key else ""
    mse_ld_str = _mse_to_latex(row["mse_acf_ld"])

    # Add midrule between parameter groups
    if prev_key is not None and key != prev_key:
        tex_lines.append(r"\hline")

    tex_lines.append(
        rf"{param_label} & {val_str} & {mse_ld_str} \\"
    )
    prev_key = key

tex_lines.append(r"\hline")
tex_lines.append(r"\end{tabular}")
tex_lines.append(r"\end{table}")

tex_str = "\n".join(tex_lines)
tex_path = paths.output / "kernel_accuracy_benchmark.tex"
with open(tex_path, "w") as f:
    f.write(tex_str)
print(f"Saved LaTeX table to {tex_path}")
print("\n" + tex_str)

# ── Grid figure: ACF + residual for every parameter sweep ───────
# 2 columns x 3 rows of cells; each cell is an ACF panel over a residual
# panel sharing the lag axis.  Cells beyond the number of sweeps are
# blank, except the first of them, which carries the line-style legend.

from matplotlib.lines import Line2D

N_COLS, N_ROWS = 2, 3
fig = plt.figure(figsize=(14, 15))
outer = fig.add_gridspec(N_ROWS, N_COLS, hspace=0.32, wspace=0.18)

for i, (key, latex, sweep_res) in enumerate(results):
    row, col_idx = divmod(i, N_COLS)
    inner = outer[row, col_idx].subgridspec(2, 1, height_ratios=[3, 1],
                                            hspace=0.06)
    ax_acf = fig.add_subplot(inner[0])
    ax_acf_res = fig.add_subplot(inner[1], sharex=ax_acf)

    for j, sr in enumerate(sweep_res):
        col = colors[j]
        lbl = f"{latex} = {sr['label']}"
        idx = sr['tau_arr'] < sr['tmax_acf']

        # ACF: numerical (solid) and analytic (dashed)
        ax_acf.plot(sr['tau_arr'][idx], sr['acf_num'][idx],
                    color=col, lw=1.3, alpha=0.6)
        ax_acf.plot(sr['tau_arr'][idx], sr['acf_ana'][idx],
                    color=col, lw=2.0, ls='--', label=lbl)

        # Residual
        acf_resid = sr['acf_num'][idx] - sr['acf_ana'][idx]
        ax_acf_res.plot(sr['tau_arr'][idx], acf_resid,
                        color=col, lw=1.0, alpha=0.7)

    # Rotation-period markers
    for sr in sweep_res:
        for n in range(1, 3):
            for ax in (ax_acf, ax_acf_res):
                ax.axvline(n * sr['peq'], color='gray', ls=':', lw=0.5,
                           alpha=0.3)

    # ACF formatting
    ax_acf.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.3)
    ax_acf.set_title(f'Vary {latex}', fontsize=title_fs)
    ax_acf.legend(fontsize=legend_fs, loc='upper right')
    ax_acf.tick_params(labelsize=tick_fs)
    ax_acf.minorticks_on()
    ax_acf.set_xlim(0, TMAX_ACF)
    plt.setp(ax_acf.get_xticklabels(), visible=False)

    # Residual formatting (short panel: three y ticks keep the labels legible)
    ax_acf_res.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.3)
    ax_acf_res.tick_params(labelsize=tick_fs)
    ax_acf_res.minorticks_on()
    ax_acf_res.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    ax_acf_res.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    if col_idx == 0:
        ax_acf.set_ylabel('ACF', fontsize=label_fs)
        ax_acf_res.set_ylabel('Residual', fontsize=label_fs)
    if i + N_COLS >= len(results):      # lowest occupied cell in this column
        ax_acf_res.set_xlabel('Time lag [days]', fontsize=label_fs)

# Spare cells: line-style legend in the first, blank otherwise
for i in range(len(results), N_ROWS * N_COLS):
    row, col_idx = divmod(i, N_COLS)
    ax = fig.add_subplot(outer[row, col_idx])
    ax.axis('off')
    if i == len(results):
        handles = [
            Line2D([], [], color='0.3', lw=1.3, alpha=0.6,
                   label='Numerical (Monte Carlo simulations)'),
            Line2D([], [], color='0.3', lw=2.0, ls='--',
                   label='Analytic kernel'),
        ]
        ax.legend(handles=handles, loc='center', fontsize=label_fs,
                  frameon=False)

fname = "fig_kernel_comparison_grid.pdf"
fig.savefig(paths.figures / fname, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"Saved {fname}")
