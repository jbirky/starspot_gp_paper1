"""
Benchmark of the dense GP solver (Table tab:gp_solver_speed): wall-clock time and peak
memory of three operations as a function of the number of data points N,

  build           GPSolver construction with matrix_solver="cholesky_full"
                  (kernel evaluation + Cholesky factorization)
  log-likelihood  the JIT-compiled log-likelihood that samplers call,
                  GPSolver.log_likelihood_fn (kernel + Cholesky + solve)
  predict         predictive mean at N_PREDICT test points

for N irregularly spaced times over T_SPAN days. Each (operation, N) pair runs in its
own process, on one CPU core, so that the peak-memory measurements are independent.
Times are the median of N_REPEATS calls after a warm-up call that JIT-compiles; memory
is the peak increase of the resident memory during those calls.

Run by hand with the pinned spotgp, since the timings depend on the machine; e.g. in
the showyourwork env:
    python compute_benchmark_gp_solver.py
table_benchmarks.py makes the table from the result.

Output:
    data/benchmark_gp_solver.json
"""
import argparse
import json

import benchmarking
benchmarking.use_cpu()

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import paths
from spotgp.gp_solver import GPSolver

HPARAM = dict(peq=5.0, kappa=0.0, inc=np.pi / 2, lspot=15.0, tau_spot=3.0, sigma_k=0.01)
N_HARMONICS = 3
N_VALUES = (10, 100, 1000, 5000, 10000)
T_SPAN = 100.0     # baseline of the data [d]
YERR = 1e-4
N_PREDICT = 200
N_REPEATS = 5
OPERATIONS = ("build", "loglike", "predict")
OUTPUT = paths.data / "benchmark_gp_solver.json"


def solver(n):
    rng = np.random.default_rng(42)
    x = np.sort(rng.uniform(0, T_SPAN, n))
    y = rng.normal(0, 1e-3, n)
    return GPSolver(x, y, np.full(n, YERR), HPARAM, n_harmonics=N_HARMONICS,
                    matrix_solver="cholesky_full")


def ready(gp):
    """Wait for the arrays a GPSolver computes asynchronously."""
    jax.block_until_ready([v for v in vars(gp).values() if isinstance(v, jax.Array)])


def worker(op, n):
    """Time one operation at one N, and print {"s": time, "MB": peak memory}."""
    if op == "build":
        call = lambda: ready(solver(n))
    elif op == "loglike":
        gp = solver(n)
        theta = jnp.asarray(gp.theta0)
        call = lambda: gp.log_likelihood_fn(theta).block_until_ready()
    elif op == "predict":
        gp = solver(n)
        xpred = np.linspace(0, T_SPAN, N_PREDICT)
        call = lambda: jax.block_until_ready(gp.predict(xpred))
    call()  # warm-up: JIT compilation
    with benchmarking.PeakRSS() as mem:
        s = benchmarking.median_time(call, N_REPEATS)
    print(json.dumps(dict(s=s, MB=mem.mb)))


def main():
    benchmarking.check_spotgp_pin()
    rows = []
    for n in N_VALUES:
        row = dict(N=n)
        for op in OPERATIONS:
            result = benchmarking.run_worker(__file__, op, n)
            row[f"{op}_s"], row[f"{op}_MB"] = result["s"], result["MB"]
        rows.append(row)
        print(f"N={n:>5d}  " + "  ".join(
            f"{op} {row[f'{op}_s']:.4g} s ({row[f'{op}_MB']:.1f} MB)" for op in OPERATIONS))
    params = dict(hparam=HPARAM, n_harmonics=N_HARMONICS, n_lat=solver(10).n_lat,
                  matrix_solver="cholesky_full", t_span=T_SPAN, yerr=YERR,
                  n_predict=N_PREDICT, n_repeats=N_REPEATS)
    benchmarking.save(OUTPUT, params, rows, device=f"1 core of {benchmarking.cpu_name()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--worker", nargs=2, metavar=("OPERATION", "N"),
                        help="time one operation at one N (used internally)")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker[0], int(args.worker[1]))
    else:
        main()
