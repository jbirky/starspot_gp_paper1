"""
Benchmark GPSolver: cholesky_banded vs. cholesky_full matrix solver.

Compares computation speed and peak memory usage for three operations
(build, log-likelihood, predict) across different time series lengths N.

Each (N, solver, operation) triple runs in an isolated subprocess so that
peak RSS measurements are independent.

Usage:
    python benchmark_banded_vs_full.py              # default: jax backend
    python benchmark_banded_vs_full.py --backend jax
    python benchmark_banded_vs_full.py --backend numpy

Output:
    output/banded_vs_full_benchmark.csv
    output/banded_vs_full_benchmark.tex
"""
import argparse
import json
import subprocess
import sys
import textwrap
import numpy as np
import pandas as pd
import paths

# Compat shim for older numpy
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

# ── CLI ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Benchmark banded vs full Cholesky")
parser.add_argument("--backend", choices=["jax", "numpy"], default="jax",
                    help="Which spotgp implementation to use (default: jax)")
parser.add_argument("--method", choices=["full", "banded"], default=None,
                    help="Benchmark only this method (default: both)")
args = parser.parse_args()

BACKEND = args.backend
METHOD = args.method

# ── Parameters ──────────────────────────────────────────────────
DT = 0.1              # fixed time spacing [days]
XMAX_VALUES = [10, 50, 100, 500, 1000, 5000, 10000]  # max observation time [days]
N_HARMONICS = 3
N_LAT = 32
N_REPEATS = 1
YERR = 1e-4

spotgp_src = str(paths.root.parent.parent / "packages" / "spotgp"
                 / ("src_jax" if BACKEND == "jax" else "src_numpy"))

# ── Common preamble for all worker scripts ──────────────────────
_PREAMBLE_COMMON = textwrap.dedent(r"""
import gc, json, os, sys, time, threading
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

sys.path.insert(0, "{spotgp_src}")

BACKEND = "{BACKEND}"

from gp_solver import (GPSolver, _gp_log_likelihood,
                       _gp_log_likelihood_banded, KERNEL_HPARAM_KEYS)
from analytic_kernel import _gauss_legendre_grid

if BACKEND == "jax":
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platforms", "cuda")
    _ = jax.devices("cuda")
    # try:
    #     jax.config.update("jax_platforms", "cuda")
    #     _ = jax.devices("cuda")
    # except Exception:
    #     try:
    #         jax.config.update("jax_platforms", "gpu")
    #         _ = jax.devices("gpu")
    #     except Exception:
    #         jax.config.update("jax_platforms", "cpu")
    import jax.numpy as jnp

hparam = dict(peq=5.0, kappa=0.0, inc=np.pi/2, lspot=15.0, tau=3.0, sigma_k=0.01)

DT = {DT}
XMAX = {XMAX}
SOLVER = "{SOLVER}"
N_HARMONICS = {N_HARMONICS}
N_LAT = {N_LAT}
LAT_RANGE = (-np.pi / 2, np.pi / 2)
N_REPEATS = {N_REPEATS}
YERR = {YERR}

np.random.seed(42)
x = np.arange(0, XMAX, DT)
N = len(x)
y = np.random.normal(0, 1e-3, N)
yerr = np.full(N, YERR)


def _rss_mb():
    "Current process RSS in MB."
    try:
        with open(f"/proc/{{os.getpid()}}/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except (FileNotFoundError, OSError):
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class PeakRSSMonitor:
    "Poll RSS in a background thread to capture peak usage."

    def __init__(self, interval=0.005):
        self._interval = interval
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        gc.collect()
        self._baseline = _rss_mb()
        self._peak = self._baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            rss = _rss_mb()
            if rss > self._peak:
                self._peak = rss
            self._stop.wait(self._interval)

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()
        rss = _rss_mb()
        if rss > self._peak:
            self._peak = rss

    @property
    def peak_delta_mb(self):
        return max(0.0, self._peak - self._baseline)


def time_func(func, n_repeats=N_REPEATS):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)
""")

# ── Per-operation worker scripts ────────────────────────────────

_BUILD_BODY = textwrap.dedent(r"""
def build():
    return GPSolver(x, y, yerr, hparam, n_harmonics=N_HARMONICS,
                    matrix_solver=SOLVER)

# Warmup (JIT compilation for jax)
build()
gc.collect()

with PeakRSSMonitor() as mon:
    dt = time_func(build, n_repeats=N_REPEATS)
print(json.dumps(dict(dt=dt, mem=mon.peak_delta_mb)))
""")

_LOGL_BODY = textwrap.dedent(r"""
quad_nodes, quad_weights = _gauss_legendre_grid(N_LAT, LAT_RANGE[0], LAT_RANGE[1])

if BACKEND == "jax":
    quad_nodes = jnp.asarray(quad_nodes)
    quad_weights = jnp.asarray(quad_weights)
    theta_arr = jnp.array([hparam[k] for k in KERNEL_HPARAM_KEYS])
    x_jax, y_jax, yerr_jax = jnp.asarray(x), jnp.asarray(y), jnp.asarray(yerr)

    if SOLVER == "cholesky_banded":
        gp_tmp = GPSolver(x, y, yerr, hparam, n_harmonics=N_HARMONICS,
                          matrix_solver="cholesky_banded")
        BW = gp_tmp.bandwidth
        del gp_tmp

        logL_jit = jax.jit(
            _gp_log_likelihood_banded,
            static_argnames=("n_harmonics", "n_lat", "lat_range",
                             "fit_sigma_n", "bandwidth"),
        )

        def logL_call():
            val = logL_jit(
                theta_arr, x_jax, y_jax, yerr_jax, 0.0,
                n_harmonics=N_HARMONICS, n_lat=N_LAT, lat_range=LAT_RANGE,
                fit_sigma_n=False, bandwidth=BW,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
            )
            val.block_until_ready()
            return val
    else:
        logL_jit = jax.jit(
            _gp_log_likelihood,
            static_argnames=("n_harmonics", "n_lat", "lat_range", "fit_sigma_n"),
        )

        def logL_call():
            val = logL_jit(
                theta_arr, x_jax, y_jax, yerr_jax, 0.0,
                n_harmonics=N_HARMONICS, n_lat=N_LAT, lat_range=LAT_RANGE,
                fit_sigma_n=False,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
            )
            val.block_until_ready()
            return val

    # Warmup (JIT compilation)
    logL_call()
    logL_call()

else:
    # numpy backend
    theta_arr = np.array([hparam[k] for k in KERNEL_HPARAM_KEYS])

    if SOLVER == "cholesky_banded":
        gp_tmp = GPSolver(x, y, yerr, hparam, n_harmonics=N_HARMONICS,
                          matrix_solver="cholesky_banded")
        BW = gp_tmp.bandwidth
        del gp_tmp

        def logL_call():
            return _gp_log_likelihood_banded(
                theta_arr, x, y, yerr, 0.0,
                n_harmonics=N_HARMONICS, n_lat=N_LAT, lat_range=LAT_RANGE,
                fit_sigma_n=False, bandwidth=BW,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
            )
    else:
        def logL_call():
            return _gp_log_likelihood(
                theta_arr, x, y, yerr, 0.0,
                n_harmonics=N_HARMONICS, n_lat=N_LAT, lat_range=LAT_RANGE,
                fit_sigma_n=False,
                quad_nodes=quad_nodes, quad_weights=quad_weights,
            )

    # Warmup
    logL_call()

gc.collect()

with PeakRSSMonitor() as mon:
    dt = time_func(logL_call, n_repeats=N_REPEATS)
print(json.dumps(dict(dt=dt, mem=mon.peak_delta_mb)))
""")

_PREDICT_BODY = textwrap.dedent(r"""
gp = GPSolver(x, y, yerr, hparam, n_harmonics=N_HARMONICS,
              matrix_solver=SOLVER)
xpred = np.linspace(x.min(), x.max(), 200)

# Warmup
gp.predict(xpred)
gc.collect()

with PeakRSSMonitor() as mon:
    dt = time_func(lambda: gp.predict(xpred), n_repeats=N_REPEATS)
print(json.dumps(dict(dt=dt, mem=mon.peak_delta_mb)))
""")


# ── Helpers ─────────────────────────────────────────────────────

def _run_worker(xmax, solver, body):
    """Run a worker script in a subprocess, return parsed JSON result."""
    script = (_PREAMBLE_COMMON + body).format(
        spotgp_src=spotgp_src,
        BACKEND=BACKEND,
        DT=DT,
        XMAX=xmax,
        SOLVER=solver,
        N_HARMONICS=N_HARMONICS,
        N_LAT=N_LAT,
        N_REPEATS=N_REPEATS,
        YERR=YERR,
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print(f"  FAILED (xmax={xmax}, solver={solver}):\n{result.stderr}",
              file=sys.stderr)
        return None
    out_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    return json.loads(out_lines[-1])


def _fmt_time(s):
    """Format time for LaTeX."""
    if s < 0.001:
        return rf"{s*1e6:.0f}\,\mu\text{{s}}"
    if s < 1:
        return rf"{s*1e3:.1f}\,\text{{ms}}"
    return rf"{s:.2f}\,\text{{s}}"


def _fmt_mem(mb):
    """Format memory: use MB for < 1024, GB otherwise."""
    if mb >= 1024:
        return rf"{mb / 1024:.2f}\,\text{{GB}}"
    if mb >= 1:
        return rf"{mb:.0f}\,\text{{MB}}"
    return rf"{mb:.1f}\,\text{{MB}}"


# ── Run benchmarks ──────────────────────────────────────────────
print("=" * 70)
print(f"GPSolver Benchmark: cholesky_banded vs. cholesky_full  [{BACKEND}]")
print("=" * 70)

rows = []
for xmax in XMAX_VALUES:
    N = len(np.arange(0, xmax, DT))
    all_solvers = [("cholesky_full", "full"), ("cholesky_banded", "banded")]
    if METHOD is not None:
        all_solvers = [(s, t) for s, t in all_solvers if t == METHOD]
    for solver, tag in all_solvers:
        print(f"  xmax={xmax:>5g}d (N={N:>5d}), solver={tag:>6s} ... ",
              end="", flush=True)

        r_build = _run_worker(xmax, solver, _BUILD_BODY)
        r_logL = _run_worker(xmax, solver, _LOGL_BODY)
        r_pred = _run_worker(xmax, solver, _PREDICT_BODY)

        if not all([r_build, r_logL, r_pred]):
            print("FAILED")
            continue

        row = dict(
            xmax=xmax,
            N=N,
            solver=tag,
            build_s=r_build["dt"], build_MB=r_build["mem"],
            logL_s=r_logL["dt"], logL_MB=r_logL["mem"],
            predict_s=r_pred["dt"], predict_MB=r_pred["mem"],
        )
        rows.append(row)
        print(f"build={row['build_s']:.4f}s ({row['build_MB']:.1f} MB)  "
              f"logL={row['logL_s']:.4f}s ({row['logL_MB']:.1f} MB)  "
              f"predict={row['predict_s']:.4f}s ({row['predict_MB']:.1f} MB)")

# ── Save CSV ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = paths.output / f"banded_vs_full_benchmark_{BACKEND}.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False, float_format="%.4g")
print(f"\nSaved CSV to {csv_path}")

# ── Generate LaTeX table ────────────────────────────────────────
# Pivot so each row is one N, with full and banded columns side by side
lines = []
lines.append(r"\begin{table*}[ht!]")
lines.append(r"    \centering")
lines.append(r"    \caption{GPSolver wall-clock time and peak memory: "
             r"\texttt{cholesky\_full} vs.\ \texttt{cholesky\_banded}. "
             r"\textit{Build}: covariance matrix construction + factorization. "
             r"\textit{Log-likelihood}: JIT-compiled evaluation. "
             r"\textit{Predict}: predictive mean at 200 test points. "
             r"Times are the median of "
             + str(N_REPEATS) + r" runs after JIT warmup. "
             r"Memory is the peak RSS increase measured in an isolated subprocess.}")
lines.append(r"    \label{tab:banded_vs_full}")
lines.append(r"    \begin{tabular}{rr cc cc cc cc cc cc}")
lines.append(r"    \toprule")
lines.append(r"    & & \multicolumn{6}{c}{Full Cholesky}"
             r" & \multicolumn{6}{c}{Banded Cholesky} \\")
lines.append(r"    \cmidrule(lr){3-8} \cmidrule(lr){9-14}")
lines.append(r"    & & \multicolumn{2}{c}{Build}"
             r" & \multicolumn{2}{c}{Log-$\mathcal{L}$}"
             r" & \multicolumn{2}{c}{Predict}"
             r" & \multicolumn{2}{c}{Build}"
             r" & \multicolumn{2}{c}{Log-$\mathcal{L}$}"
             r" & \multicolumn{2}{c}{Predict} \\")
lines.append(r"    \cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}"
             r" \cmidrule(lr){9-10} \cmidrule(lr){11-12} \cmidrule(lr){13-14}")
lines.append(r"    $x_{\max}$ [d] & $N$ & Time & Mem & Time & Mem & Time & Mem"
             r" & Time & Mem & Time & Mem & Time & Mem \\")
lines.append(r"    \midrule")

for xmax in XMAX_VALUES:
    full = df[(df["xmax"] == xmax) & (df["solver"] == "full")]
    banded = df[(df["xmax"] == xmax) & (df["solver"] == "banded")]
    if full.empty or banded.empty:
        continue
    f = full.iloc[0]
    b = banded.iloc[0]
    lines.append(
        f"    {xmax:g} & {int(f['N'])}"
        f" & ${_fmt_time(f['build_s'])}$ & ${_fmt_mem(f['build_MB'])}$"
        f" & ${_fmt_time(f['logL_s'])}$ & ${_fmt_mem(f['logL_MB'])}$"
        f" & ${_fmt_time(f['predict_s'])}$ & ${_fmt_mem(f['predict_MB'])}$"
        f" & ${_fmt_time(b['build_s'])}$ & ${_fmt_mem(b['build_MB'])}$"
        f" & ${_fmt_time(b['logL_s'])}$ & ${_fmt_mem(b['logL_MB'])}$"
        f" & ${_fmt_time(b['predict_s'])}$ & ${_fmt_mem(b['predict_MB'])}$"
        r" \\"
    )

lines.append(r"    \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"\end{table*}")

tex_str = "\n".join(lines)
tex_path = paths.output / f"banded_vs_full_benchmark_{BACKEND}.tex"
with open(tex_path, "w") as f:
    f.write(tex_str)
print(f"Saved LaTeX to {tex_path}")
print("\n" + tex_str)
