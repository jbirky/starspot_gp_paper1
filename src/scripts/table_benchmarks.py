"""
Tables of the GP solver and kernel benchmarks, from the timings the compute_benchmark_*.py
scripts record by hand (timings depend on the machine, and the banded-solver run needs a
GPU, so the build does not run them):

    data/benchmark_gp_solver_{cpu,gpu}.json       ->  Table tab:gp_solver_speed
    data/benchmark_banded_vs_full_{cpu,gpu}.json  ->  Table tab:banded_vs_full
    data/benchmark_kernel_speed.json              ->  Table tab:kernel_speed

The first two tables show the CPU and GPU runs one above the other. Each caption names
the hardware its timings were measured on. The last two outputs are the log-likelihood
times at N = 1000 on the CPU and on the GPU that the text quotes, so the sentence can't
go stale.

Usage:
    python table_benchmarks.py gp_solver_speed.tex banded_vs_full.tex kernel_speed.tex \
        loglike_n1000_cpu.tex loglike_n1000_gpu.tex
"""
import json
import math
import sys
from pathlib import Path

import paths


DEVICES = ("cpu", "gpu")


def load(name):
    return json.loads((paths.data / f"benchmark_{name}.json").read_text())


def load_devices(name):
    """The CPU and GPU runs of one benchmark, which must share their settings."""
    results = {device: load(f"{name}_{device}") for device in DEVICES}
    if results["cpu"]["params"] != results["gpu"]["params"]:
        raise ValueError(f"the CPU and GPU runs of benchmark {name} have different settings")
    return results


def count(n):
    return f"{n:,}".replace(",", "{,}")


def duration(s):
    """A time to 3 significant figures, in s, ms or us; OOM if the run ran out of memory."""
    if s is None:
        return "OOM"
    for scale, unit in ((1, r"\text{s}"), (1e-3, r"\text{ms}"), (1e-6, r"\mu\text{s}")):
        value = float(f"{s / scale:.3g}")
        if value >= 1 or scale == 1e-6:
            decimals = max(0, 2 - int(math.floor(math.log10(value))))
            return rf"${value:.{decimals}f}\,{unit}$"


def memory(mb):
    if mb is None:
        return "OOM"
    if mb >= 1024:
        return rf"${mb / 1024:.2f}\,\text{{GB}}$"
    return rf"${mb:.0f}\,\text{{MB}}$" if mb >= 1 else rf"${mb:.1f}\,\text{{MB}}$"


def hardware(provenance):
    device = provenance["device"]
    if device.startswith("1 core of "):
        return f"one core of an {provenance['cpu']}"
    return f"an {device} GPU"


def hardware_both(results):
    return (f"on {hardware(results['cpu']['provenance'])} (CPU) and on "
            f"{hardware(results['gpu']['provenance'])} (GPU)")


def group(results, device, n_columns):
    """The row that heads one device's block of a table."""
    label = results[device]["provenance"]["device"].replace("1 core of ", "one core of ")
    return rf"\multicolumn{{{n_columns}}}{{l}}{{\textit{{{device.upper()}: {label}}}}} \\"


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
        *(f"    {row}" if isinstance(row, str) else f"    {' & '.join(row)} \\\\"
          for row in rows),
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ])


def gp_solver_table():
    results = load_devices("gp_solver")
    p = results["cpu"]["params"]
    caption = (
        r"GPSolver wall-clock time and peak memory as a function of the number of data "
        rf"points $N$, irregularly spaced over {p['t_span']:g}~d, with the full (dense) "
        rf"Cholesky solver, {hardware_both(results)}. \textit{{Build}}: covariance matrix + "
        r"Cholesky factorization. \textit{Log-likelihood}: JIT-compiled evaluation (kernel "
        r"+ Cholesky + solve) as called during MCMC. \textit{Predict}: predictive mean and "
        rf"variance at {p['n_predict']} test points. Times are the median of "
        rf"{p['n_repeats']} runs after JIT compilation. Memory is the peak increase during "
        r"each operation of the resident memory (CPU) or of the GPU memory in use (GPU), "
        r"with each operation run in its own process.")
    header = [
        r"& \multicolumn{2}{c}{Build} & \multicolumn{2}{c}{Log-likelihood} "
        r"& \multicolumn{2}{c}{Predict} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"$N$ & Time & Memory & Time & Memory & Time & Memory \\",
    ]
    rows = []
    for device in DEVICES:
        if rows:
            rows.append(r"\midrule")
        rows.append(group(results, device, 7))
        rows += [[count(r["N"]), *(f(r[f"{op}_{unit}"]) for op in ("build", "loglike", "predict")
                                   for unit, f in (("s", duration), ("MB", memory)))]
                 for r in results[device]["rows"]]
    return table(caption, "tab:gp_solver_speed", "rcccccc", header, rows)


def banded_vs_full_table():
    results = load_devices("banded_vs_full")
    p = results["cpu"]["params"]
    all_rows = [r for result in results.values() for r in result["rows"]]
    out_of_memory = any(r[f"{op}_s"] is None for r in all_rows for op in ("build", "predict"))
    largest = {device: max((r for r in result["rows"] if r["solver"] == "banded"),
                           key=lambda r: r["N"]) for device, result in results.items()}
    caption = (
        r"GPSolver wall-clock time: \texttt{cholesky\_full} vs.\ \texttt{cholesky\_banded}, "
        rf"for data sampled uniformly at $\Delta t = {p['dt']:g}$~d over a baseline "
        rf"$x_{{\max}}$, {hardware_both(results)}. \textit{{Build}}: kernel evaluation + "
        r"Cholesky factorization. \textit{Predict}: GP predictive mean and variance at "
        rf"{p['n_predict']} test points. \textit{{Speedup}}: full over banded build time. "
        rf"Times are the median of {p['n_repeats']} runs after JIT compilation. The full "
        r"Cholesky solver scales as $\mathcal{O}(N^3)$ and was not run above "
        rf"$N = {count(p['n_full_max'])}$, where its covariance matrix no longer fits in GPU "
        r"memory; the banded solver scales as $\mathcal{O}(Nb^2)$, and its build at "
        rf"$N = {count(largest['cpu']['N'])}$ takes "
        + " and ".join("runs out of memory" if largest[d]["build_s"] is None else
                       f"{duration(largest[d]['build_s'])} on the {d.upper()}"
                       for d in DEVICES) + "."
        + (" OOM: the run exceeded the GPU's memory." if out_of_memory else ""))
    header = [
        r"& & \multicolumn{2}{c}{Full Cholesky} & \multicolumn{2}{c}{Banded Cholesky} & \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"$x_{\max}$ [d] & $N$ & Build & Predict & Build & Predict & Speedup \\",
    ]
    rows = []
    for device, result in results.items():
        if rows:
            rows.append(r"\midrule")
        rows.append(group(results, device, 7))
        runs = {(r["xmax"], r["solver"]): r for r in result["rows"]}
        for xmax in sorted({r["xmax"] for r in result["rows"]}):
            full, banded = runs.get((xmax, "full")), runs[(xmax, "banded")]
            times = lambda r: [duration(r["build_s"]), duration(r["predict_s"])] if r else ["---", "---"]
            speedup = (rf"${full['build_s'] / banded['build_s']:.1f}\times$"
                       if full and full["build_s"] is not None and banded["build_s"] is not None
                       else "---")
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


def loglike_at_1000(device):
    """The quoted time: one log-likelihood evaluation at N = 1000."""
    (row,) = [r for r in load(f"gp_solver_{device}")["rows"] if r["N"] == 1000]
    return duration(row["loglike_s"])


if __name__ == "__main__":
    for path, make in zip(sys.argv[1:6], (gp_solver_table, banded_vs_full_table,
                                          kernel_speed_table, lambda: loglike_at_1000("cpu"),
                                          lambda: loglike_at_1000("gpu"))):
        Path(path).write_text(make())
        print(f"Saved {path}")
