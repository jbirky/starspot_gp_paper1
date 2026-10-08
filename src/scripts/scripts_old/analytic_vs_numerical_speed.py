"""
Benchmark kernel computation speed: analytic vs. numerical (Monte Carlo).

  1. Analytic kernel   (spotgp/src/analytic_kernel.py)
  2. Numerical kernel  (spotgp/src/numerical_kernel.py)

Tests multiple lag-array sizes for the analytic kernel and a fixed
configuration for the numerical MC kernel, then reports the speedup.

Output:
    output/kernel_speed_benchmark.csv
    output/kernel_speed_benchmark.tex
"""
import sys
import time
import numpy as np
import pandas as pd
import paths

# Compat shim for older numpy
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# ── Import implementations ────────────────────────────────────
sys.path.insert(0, str(paths.root.parent.parent / "packages" / "spotgp" / "src"))
from analytic_kernel import AnalyticKernel
from numerical_kernel import generate_sims, avg_covariance_tlag

# ── Parameters ──────────────────────────────────────────────────
hparam = dict(
    peq=5.0,
    kappa=0.0,
    inc=np.pi / 2,
    nspot=10,
    lspot=15.0,
    tau=3.0,
    alpha_max=0.1,
    fspot=0.0,
)

N_HARMONICS = 3
NSIM = 200        # number of MC simulations for numerical kernel
TSIM = 100        # simulation duration [days]
TSAMP = 0.05      # sampling cadence [days]
N_REPEATS = 5     # number of timing repeats (take median)

# Lag array sizes to test
lag_sizes = [50, 100, 500, 1000, 5000]


# ── Timing utilities ────────────────────────────────────────────
def time_func(func, n_repeats=N_REPEATS):
    """Time a function, return median wall-clock time in seconds."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times), result


# ── Benchmark: Analytic kernel ──────────────────────────────────
def bench_analytic(n_lags):
    tau = np.linspace(0, TSIM, n_lags)
    ak = AnalyticKernel(hparam, n_harmonics=N_HARMONICS)
    dt, _ = time_func(lambda: ak.kernel(tau))
    return dt


# ── Benchmark: Numerical kernel (MC) ────────────────────────────
NSIM_500 = 500

def bench_numerical(n_lags, nsim=NSIM):
    """Full MC: generate_sims + cov + avg_covariance_tlag.
    Adjusts tsamp so the simulation produces n_lags time points."""
    theta = [hparam["peq"], hparam["kappa"], hparam["inc"], hparam["nspot"]]
    tsamp = TSIM / n_lags

    def _run():
        fluxes = generate_sims(
            theta, nsim=nsim,
            tem=hparam["tau"], tdec=hparam["tau"],
            alpha_max=hparam["alpha_max"],
            lspot=hparam["lspot"],
            tsim=TSIM, tsamp=tsamp,
        )
        K = np.cov(fluxes.T)
        k_tau = avg_covariance_tlag(K)
        return k_tau

    dt, _ = time_func(_run, n_repeats=1)  # MC is expensive, run once
    return dt


# ── Run benchmarks ──────────────────────────────────────────────
print("=" * 60)
print("Kernel Speed Benchmark: Analytic vs. Numerical")
print("=" * 60)

rows = []
for n_lags in lag_sizes:
    dt_ana = bench_analytic(n_lags)
    dt_num200 = bench_numerical(n_lags, nsim=NSIM)
    dt_num500 = bench_numerical(n_lags, nsim=NSIM_500)
    sp200 = dt_num200 / dt_ana
    sp500 = dt_num500 / dt_ana
    print(f"  n_lags={n_lags:>5d}:  analytic={dt_ana:.4f}  "
          f"num200={dt_num200:.2f} ({sp200:.0f}x)  "
          f"num500={dt_num500:.2f} ({sp500:.0f}x)")
    rows.append(dict(n_lags=n_lags, analytic_s=dt_ana,
                     numerical_200_s=dt_num200, speedup_200=sp200,
                     numerical_500_s=dt_num500, speedup_500=sp500))

# ── Save CSV ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = paths.output / "kernel_speed_benchmark.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False, float_format="%.4g")
print(f"\nSaved CSV to {csv_path}")

# ── Generate LaTeX table ────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[ht!]")
lines.append(r"    \centering")
lines.append(r"    \caption{Kernel computation wall-clock time. "
             r"The analytic kernel is evaluated at varying lag-array sizes; "
             r"the numerical (Monte Carlo) kernel uses "
             + str(NSIM) + r" and " + str(NSIM_500) +
             r" simulations over " + str(TSIM) + r"~d. "
             r"All times in seconds.}")
lines.append(r"    \label{tab:kernel_speed}")
lines.append(r"    \begin{tabular}{rccccc}")
lines.append(r"    \toprule")
lines.append(r"    & & \multicolumn{2}{c}{$N_{\rm sim}=200$}"
             r" & \multicolumn{2}{c}{$N_{\rm sim}=500$} \\")
lines.append(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6}")
lines.append(r"    $N_{\rm lags}$ & Analytic [s] & "
             r"Numerical [s] & Speedup & "
             r"Numerical [s] & Speedup \\")
lines.append(r"    \midrule")

for _, row in df.iterrows():
    lines.append(f"    {int(row['n_lags'])} & {row['analytic_s']:.4f} & "
                 f"{row['numerical_200_s']:.2f} & "
                 f"{row['speedup_200']:.0f}$\\times$ & "
                 f"{row['numerical_500_s']:.2f} & "
                 f"{row['speedup_500']:.0f}$\\times$ \\\\")

lines.append(r"    \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"\end{table}")

tex_str = "\n".join(lines)
tex_path = paths.output / "kernel_speed_benchmark.tex"
with open(tex_path, "w") as f:
    f.write(tex_str)
print(f"Saved LaTeX to {tex_path}")

# Print for convenience
print("\n" + tex_str)
