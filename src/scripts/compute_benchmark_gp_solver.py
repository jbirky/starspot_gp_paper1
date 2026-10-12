"""
Benchmark GPSolver computation speed and peak memory usage as a function
of the number of data points.

Times three operations at each N:
  1. Build:  covariance matrix construction + Cholesky factorization
  2. Log-likelihood:  the pure-functional _gp_log_likelihood used during
             MCMC (kernel eval + Cholesky + solve, JIT-compiled)
  3. Predict: predictive mean at M=200 test points

Each (N, operation) pair runs in its own isolated subprocess so that
peak RSS measurements are independent.

Tests N = 10, 100, 1000, 5000, 10000 data points.

Output:
    output/gp_solver_speed_benchmark.csv
    output/gp_solver_speed_benchmark.tex
"""
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

# ── Parameters ──────────────────────────────────────────────────
N_VALUES = [10, 100, 1000, 5000, 10000]
N_HARMONICS = 3
N_LAT = 32
N_REPEATS = 5
YERR = 1e-4

# ── Common preamble for all worker scripts ──────────────────────
_PREAMBLE = textwrap.dedent(r"""
import gc, json, os, sys, time, threading
import numpy as np
if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

sys.path.insert(0, "{spotgp_src}")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gp_solver import GPSolver, _gp_log_likelihood, KERNEL_HPARAM_KEYS
from analytic_kernel import _gauss_legendre_grid

hparam = dict(peq=5.0, kappa=0.0, inc=np.pi/2, lspot=15.0, tau=3.0, sigma_k=0.01)

N = {N}
N_HARMONICS = {N_HARMONICS}
N_LAT = {N_LAT}
LAT_RANGE = (-np.pi / 2, np.pi / 2)
N_REPEATS = {N_REPEATS}
YERR = {YERR}

np.random.seed(42)
x = np.sort(np.random.uniform(0, 100, N))
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
    return GPSolver(x, y, yerr, hparam, n_harmonics=N_HARMONICS)

# Warmup (JIT compilation)
build()
gc.collect()

with PeakRSSMonitor() as mon:
    dt = time_func(build, n_repeats=N_REPEATS)
print(json.dumps(dict(dt=dt, mem=mon.peak_delta_mb)))
""")

_LOGL_BODY = textwrap.dedent(r"""
quad_nodes, quad_weights = _gauss_legendre_grid(N_LAT, LAT_RANGE[0], LAT_RANGE[1])
quad_nodes = jnp.asarray(quad_nodes)
quad_weights = jnp.asarray(quad_weights)
theta_arr = jnp.array([hparam[k] for k in KERNEL_HPARAM_KEYS])
x_jax, y_jax, yerr_jax = jnp.asarray(x), jnp.asarray(y), jnp.asarray(yerr)

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
gc.collect()

with PeakRSSMonitor() as mon:
    dt = time_func(logL_call, n_repeats=N_REPEATS)
print(json.dumps(dict(dt=dt, mem=mon.peak_delta_mb)))
""")

_PREDICT_BODY = textwrap.dedent(r"""
gp = GPSolver(x, y, yerr, hparam, n_harmonics=N_HARMONICS)
xpred = np.linspace(x.min(), x.max(), 200)

# Warmup
gp.predict(xpred)
gc.collect()

with PeakRSSMonitor() as mon:
    dt = time_func(lambda: gp.predict(xpred), n_repeats=N_REPEATS)
print(json.dumps(dict(dt=dt, mem=mon.peak_delta_mb)))
""")


# ── Helpers ─────────────────────────────────────────────────────
spotgp_src = str(paths.root.parent.parent / "packages" / "spotgp" / "src")


def _run_worker(N, body):
    """Run a worker script in a subprocess, return parsed JSON result."""
    script = (_PREAMBLE + body).format(
        spotgp_src=spotgp_src,
        N=N,
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
        print(f"  FAILED:\n{result.stderr}", file=sys.stderr)
        return None
    out_lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
    return json.loads(out_lines[-1])


def _fmt_mem(mb):
    """Format memory: use MB for < 1024, GB otherwise."""
    if mb >= 1024:
        return rf"{mb / 1024:.2f}\,\text{{GB}}"
    if mb >= 1:
        return rf"{mb:.0f}\,\text{{MB}}"
    return rf"{mb:.1f}\,\text{{MB}}"


# ── Run benchmarks ──────────────────────────────────────────────
print("=" * 70)
print("GPSolver Speed & Memory Benchmark vs N data points")
print("=" * 70)

rows = []
for N in N_VALUES:
    r_build = _run_worker(N, _BUILD_BODY)
    r_logL = _run_worker(N, _LOGL_BODY)
    r_pred = _run_worker(N, _PREDICT_BODY)

    if not all([r_build, r_logL, r_pred]):
        continue

    row = dict(
        N=N,
        build_s=r_build["dt"], build_MB=r_build["mem"],
        logL_s=r_logL["dt"], logL_MB=r_logL["mem"],
        predict_s=r_pred["dt"], predict_MB=r_pred["mem"],
    )
    rows.append(row)

    print(f"  N={N:>5d}:  "
          f"build={row['build_s']:.4f}s ({row['build_MB']:.1f} MB)  "
          f"logL={row['logL_s']:.4f}s ({row['logL_MB']:.1f} MB)  "
          f"predict={row['predict_s']:.4f}s ({row['predict_MB']:.1f} MB)")

# ── Save CSV ────────────────────────────────────────────────────
df = pd.DataFrame(rows)
csv_path = paths.output / "gp_solver_speed_benchmark.csv"
csv_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(csv_path, index=False, float_format="%.4g")
print(f"\nSaved CSV to {csv_path}")

# ── Generate LaTeX table ────────────────────────────────────────
lines = []
lines.append(r"\begin{table}[ht!]")
lines.append(r"    \centering")
lines.append(r"    \caption{GPSolver wall-clock time and peak memory as a "
             r"function of the number of data points $N$. "
             r"\textit{Build}: covariance matrix + Cholesky factorization. "
             r"\textit{Log-likelihood}: JIT-compiled evaluation (kernel + "
             r"Cholesky + solve) as called during MCMC. "
             r"\textit{Predict}: predictive mean at 200 test points. "
             r"Times are the median of "
             + str(N_REPEATS) + r" runs after JIT warmup. "
             r"Memory is the peak RSS increase during each operation, "
             r"measured in an isolated subprocess.}")
lines.append(r"    \label{tab:gp_solver_speed}")
lines.append(r"    \begin{tabular}{rcccccc}")
lines.append(r"    \toprule")
lines.append(r"    & \multicolumn{2}{c}{Build}"
             r" & \multicolumn{2}{c}{Log-likelihood}"
             r" & \multicolumn{2}{c}{Predict} \\")
lines.append(r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
lines.append(r"    $N$ & Time [s] & Memory"
             r" & Time [s] & Memory"
             r" & Time [s] & Memory \\")
lines.append(r"    \midrule")

for _, row in df.iterrows():
    lines.append(
        f"    {int(row['N'])} "
        f"& {row['build_s']:.4f} & ${_fmt_mem(row['build_MB'])}$ "
        f"& {row['logL_s']:.4f} & ${_fmt_mem(row['logL_MB'])}$ "
        f"& {row['predict_s']:.4f} & ${_fmt_mem(row['predict_MB'])}$ \\\\"
    )

lines.append(r"    \bottomrule")
lines.append(r"    \end{tabular}")
lines.append(r"\end{table}")

tex_str = "\n".join(lines)
tex_path = paths.output / "gp_solver_speed_benchmark.tex"
with open(tex_path, "w") as f:
    f.write(tex_str)
print(f"Saved LaTeX to {tex_path}")
print("\n" + tex_str)
