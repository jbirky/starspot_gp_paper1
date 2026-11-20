"""
Benchmark of the banded against the full (dense) Cholesky GP solver (Table
tab:banded_vs_full): wall-clock time of three operations,

  build           GPSolver construction (kernel evaluation + Cholesky factorization)
  log-likelihood  the JIT-compiled log-likelihood that samplers call,
                  GPSolver.log_likelihood_fn
  predict         GPSolver.predict at N_PREDICT test points (mean and variance)

for data sampled uniformly at cadence DT over a baseline x_max, so N = x_max / DT, on
one CPU core (--device cpu) or on the GPU (--device gpu). The full solver is skipped
above N_FULL_MAX points, where its N x N covariance no longer fits in GPU memory, and a
build on one CPU core would take hours. Each (solver, operation, N) runs in its own
process. Times are the median of N_REPEATS calls after a warm-up call that compiles.

Run by hand with the pinned spotgp, since the timings depend on the machine. On the CPU,
e.g. in the showyourwork env:
    python compute_benchmark_banded_vs_full.py --device cpu
and on the GPU, in an env with CUDA-enabled JAX:
    PYTHONPATH=<checkout of the pinned spotgp> python compute_benchmark_banded_vs_full.py --device gpu
table_benchmarks.py makes the table from the results.

Output:
    data/benchmark_banded_vs_full_cpu.json, data/benchmark_banded_vs_full_gpu.json
"""
import argparse
import json

import benchmarking

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
parser.add_argument("--worker", nargs=3, metavar=("SOLVER", "OPERATION", "X_MAX"),
                    help="time one operation of one solver at one baseline (used internally)")
ARGS = parser.parse_args()
benchmarking.use_device(ARGS.device)  # before importing jax

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import paths
from spotgp.gp_solver import GPSolver

HPARAM = dict(peq=5.0, kappa=0.0, inc=np.pi / 2, lspot=15.0, tau_spot=3.0, sigma_k=0.01)
N_HARMONICS = 3
DT = 0.1                                            # cadence [d]
XMAX_VALUES = (10, 50, 100, 500, 1000, 5000, 10000)  # baselines [d]
N_FULL_MAX = 10_000
YERR = 1e-4
N_PREDICT = 200
N_REPEATS = 3
SOLVERS = ("full", "banded")
OPERATIONS = ("build", "loglike", "predict")
OUTPUT = paths.data / f"benchmark_banded_vs_full_{ARGS.device}.json"


def solver(xmax, kind):
    rng = np.random.default_rng(42)
    x = np.arange(0, xmax, DT)
    y = rng.normal(0, 1e-3, len(x))
    return GPSolver(x, y, np.full(len(x), YERR), HPARAM, n_harmonics=N_HARMONICS,
                    matrix_solver=f"cholesky_{kind}")


def ready(gp):
    """Wait for the arrays a GPSolver computes asynchronously."""
    jax.block_until_ready([v for v in vars(gp).values() if isinstance(v, jax.Array)])


def worker(kind, op, xmax):
    """Time one operation of one solver at one baseline, and print {"s": time}."""
    if op == "build":
        call = lambda: ready(solver(xmax, kind))
    elif op == "loglike":
        gp = solver(xmax, kind)
        theta = jnp.asarray(gp.theta0)
        call = lambda: gp.log_likelihood_fn(theta).block_until_ready()
    elif op == "predict":
        gp = solver(xmax, kind)
        xpred = np.linspace(0, xmax, N_PREDICT)
        call = lambda: jax.block_until_ready(gp.predict(xpred))
    call()  # warm-up: JIT compilation
    print(json.dumps(dict(s=benchmarking.median_time(call, N_REPEATS))))


def main():
    benchmarking.check_spotgp_pin()
    rows = []
    for xmax in XMAX_VALUES:
        n = len(np.arange(0, xmax, DT))
        for kind in SOLVERS:
            if kind == "full" and n > N_FULL_MAX:
                continue
            row = dict(xmax=xmax, N=n, solver=kind)
            for op in OPERATIONS:
                try:
                    row[f"{op}_s"] = benchmarking.run_worker(
                        __file__, "--device", ARGS.device, "--worker", kind, op, xmax)["s"]
                except RuntimeError as error:  # running out of GPU memory is a result; anything else is not
                    if "RESOURCE_EXHAUSTED" not in str(error):
                        raise
                    row[f"{op}_s"] = None
            rows.append(row)
            print(f"x_max={xmax:>5g} d  N={n:>6d}  {kind:>6s}  " + "  ".join(
                f"{op} {'out of memory' if row[f'{op}_s'] is None else format(row[f'{op}_s'], '.4g') + ' s'}"
                for op in OPERATIONS))
    gp = solver(XMAX_VALUES[0], "banded")
    params = dict(hparam=HPARAM, n_harmonics=N_HARMONICS, n_lat=gp.n_lat, dt=DT,
                  n_full_max=N_FULL_MAX, yerr=YERR, n_predict=N_PREDICT,
                  n_repeats=N_REPEATS)
    benchmarking.save(OUTPUT, params, rows, device=benchmarking.device_name(ARGS.device))


if __name__ == "__main__":
    if ARGS.worker:
        worker(ARGS.worker[0], ARGS.worker[1], float(ARGS.worker[2]))
    else:
        main()
