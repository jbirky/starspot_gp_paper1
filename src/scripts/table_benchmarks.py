"""
Tables of the GP solver and kernel benchmarks, from the timings the compute_benchmark_*.py
scripts record by hand (timings depend on the machine, and the banded-solver run needs a
GPU, so the build does not run them):

    data/benchmark_gp_solver.json       ->  Table tab:gp_solver_speed
    data/benchmark_banded_vs_full.json  ->  Table tab:banded_vs_full
    data/benchmark_kernel_speed.json    ->  Table tab:kernel_speed

Each caption names the hardware its timings were measured on. A fourth output is the
log-likelihood time at N = 1000 that the text quotes, so the sentence can't go stale.

Usage:
    python table_benchmarks.py gp_solver_speed.tex banded_vs_full.tex kernel_speed.tex loglike_n1000.tex
"""
import json
import math
import sys
from pathlib import Path

import paths


def load(name):
    return json.loads((paths.data / f"benchmark_{name}.json").read_text())


def count(n):
    return f"{n:,}".replace(",", "{,}")


def duration(s):
    """A time to 3 significant figures, in s, ms or us."""
    for scale, unit in ((1, r"\text{s}"), (1e-3, r"\text{ms}"), (1e-6, r"\mu\text{s}")):
        value = float(f"{s / scale:.3g}")
        if value >= 1 or scale == 1e-6:
            decimals = max(0, 2 - int(math.floor(math.log10(value))))
            return rf"${value:.{decimals}f}\,{unit}$"


def memory(mb):
    if mb >= 1024:
        return rf"${mb / 1024:.2f}\,\text{{GB}}$"
    return rf"${mb:.0f}\,\text{{MB}}$" if mb >= 1 else rf"${mb:.1f}\,\text{{MB}}$"


def hardware(provenance):
    device = provenance["device"]
    if device.startswith("1 core of "):
        return f"one core of an {provenance['cpu']}"
    return f"an {device} GPU"


def table(caption, label, columns, header, rows):
    return "\n".join([
        r"\begin{table}[ht!]",
        r"    \centering",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        rf"    \begin{{tabular}}{{{columns}}}",
        r"    \toprule",
        *(f"    {line}" for line in header),
        r"    \midrule",
        *(f"    {' & '.join(row)} \\\\" for row in rows),
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])


def gp_solver_table():
    result = load("gp_solver")
    p = result["params"]
    caption = (
        r"GPSolver wall-clock time and peak memory as a function of the number of data "
        rf"points $N$, irregularly spaced over {p['t_span']:g}~d, with the full (dense) "
        r"Cholesky solver. \textit{Build}: covariance matrix + Cholesky factorization. "
        r"\textit{Log-likelihood}: JIT-compiled evaluation (kernel + Cholesky + solve) as "
        rf"called during MCMC. \textit{{Predict}}: predictive mean at {p['n_predict']} test "
        rf"points. Times are the median of {p['n_repeats']} runs after JIT warmup, on "
        rf"{hardware(result['provenance'])}. Memory is the peak RSS increase during each "
        r"operation, measured in an isolated subprocess.")
    header = [
        r"& \multicolumn{2}{c}{Build} & \multicolumn{2}{c}{Log-likelihood} "
        r"& \multicolumn{2}{c}{Predict} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"$N$ & Time & Memory & Time & Memory & Time & Memory \\",
    ]
    rows = [[count(r["N"]), *(f(r[f"{op}_{unit}"]) for op in ("build", "loglike", "predict")
                              for unit, f in (("s", duration), ("MB", memory)))]
            for r in result["rows"]]
    return table(caption, "tab:gp_solver_speed", "rcccccc", header, rows)


def banded_vs_full_table():
    result = load("banded_vs_full")
    p = result["params"]
    runs = {(r["xmax"], r["solver"]): r for r in result["rows"]}
    out_of_memory = any(r[f"{op}_s"] is None for r in result["rows"] for op in ("build", "predict"))
    largest = max((r for r in result["rows"] if r["solver"] == "banded"), key=lambda r: r["N"])
    caption = (
        r"GPSolver wall-clock time: \texttt{cholesky\_full} vs.\ \texttt{cholesky\_banded}, "
        rf"for data sampled uniformly at $\Delta t = {p['dt']:g}$~d over a baseline "
        r"$x_{\max}$. \textit{Build}: kernel evaluation + Cholesky factorization. "
        rf"\textit{{Predict}}: GP predictive mean at {p['n_predict']} test points. "
        r"\textit{Speedup}: full over banded build time. Times are the median of "
        rf"{p['n_repeats']} runs after JIT warmup, on {hardware(result['provenance'])}. The "
        r"full Cholesky solver scales as $\mathcal{O}(N^3)$ and was not run above "
        rf"$N = {count(p['n_full_max'])}$, where its covariance matrix no longer fits in GPU "
        r"memory; the banded solver scales as $\mathcal{O}(Nb^2)$ and builds the "
        rf"$N = {count(largest['N'])}$ solver in {duration(largest['build_s'])}."
        + (" OOM: the run exceeded the GPU's memory." if out_of_memory else ""))
    header = [
        r"& & \multicolumn{2}{c}{Full Cholesky} & \multicolumn{2}{c}{Banded Cholesky} & \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"$x_{\max}$ [d] & $N$ & Build & Predict & Build & Predict & Speedup \\",
    ]
    rows = []
    for xmax in sorted({r["xmax"] for r in result["rows"]}):
        full, banded = runs.get((xmax, "full")), runs[(xmax, "banded")]
        cell = lambda s: "OOM" if s is None else duration(s)
        times = lambda r: [cell(r["build_s"]), cell(r["predict_s"])] if r else ["---", "---"]
        speedup = (rf"${full['build_s'] / banded['build_s']:.1f}\times$"
                   if full and full["build_s"] is not None else "---")
        rows.append([count(xmax), count(banded["N"]), *times(full), *times(banded), speedup])
    return table(caption, "tab:banded_vs_full", "rr cc cc c", header, rows)


def kernel_speed_table():
    result = load("kernel_speed")
    p = result["params"]
    nsim = p["nsim"]
    caption = (
        r"A comparison showing the kernel computation time using the analytic and numerical "
        r"approaches. The analytic kernel is evaluated at varying lag-array sizes (with $N$ "
        r"simulated datapoints); the numerical (Monte Carlo) kernel uses "
        rf"{' and '.join(map(str, nsim))} simulations over {p['t_sim']:g}~d. Analytic times "
        rf"are the median of {p['n_repeats']} runs after JIT warmup and numerical times are "
        rf"single runs, on {hardware(result['provenance'])}.")
    header = [
        "& & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$N_{{\rm sim}}={n}$}}" for n in nsim) + r" \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"$N$ & Analytic & Numerical & Speedup & Numerical & Speedup \\",
    ]
    rows = [[count(r["N"]), duration(r["analytic_s"]),
             *(cell for n in nsim for cell in (
                 duration(r[f"numerical_{n}_s"]),
                 rf"{r[f'numerical_{n}_s'] / r['analytic_s']:.0f}$\times$"))]
            for r in result["rows"]]
    return table(caption, "tab:kernel_speed", "rccccc", header, rows)


def loglike_at_1000():
    """The quoted time: one log-likelihood evaluation at N = 1000."""
    (row,) = [r for r in load("gp_solver")["rows"] if r["N"] == 1000]
    return duration(row["loglike_s"])


if __name__ == "__main__":
    for path, make in zip(sys.argv[1:5], (gp_solver_table, banded_vs_full_table,
                                          kernel_speed_table, loglike_at_1000)):
        Path(path).write_text(make())
        print(f"Saved {path}")
