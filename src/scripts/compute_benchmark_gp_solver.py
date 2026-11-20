"""
Benchmark of the dense GP solver (Table tab:gp_solver_speed): wall-clock time and peak
memory of three operations as a function of the number of data points N,

  build           GPSolver construction with matrix_solver="cholesky_full"
                  (kernel evaluation + Cholesky factorization)
  log-likelihood  the JIT-compiled log-likelihood that samplers call,
                  GPSolver.log_likelihood_fn (kernel + Cholesky + solve)
  predict         GPSolver.predict at N_PREDICT test points (mean and variance)

for N irregularly spaced times over T_SPAN days, on one CPU core (--device cpu) or on
the GPU (--device gpu). Each (operation, N) pair runs in its own process, so that the
peak-memory measurements are independent. Times are the median of N_REPEATS calls after
compilation. Memory is the peak increase during those calls of the resident memory
(CPU) or of the GPU memory in use (GPU). build and predict run op by op, so a warm-up
call compiles them. The log-likelihood is compiled ahead of time instead, so that its
first call, which allocates the buffers it keeps, is inside the measurement.

Run by hand with the pinned spotgp, since the timings depend on the machine. On the CPU,
e.g. in the showyourwork env:
    python compute_benchmark_gp_solver.py --device cpu
and on the GPU, in an env with CUDA-enabled JAX:
    PYTHONPATH=<checkout of the pinned spotgp> python compute_benchmark_gp_solver.py --device gpu
table_benchmarks.py makes the table from the results.

Output:
    data/benchmark_gp_solver_cpu.json, data/benchmark_gp_solver_gpu.json
"""
import argparse
import json

import benchmarking

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
parser.add_argument("--worker", nargs=2, metavar=("OPERATION", "N"),
                    help="time one operation at one N (used internally)")
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
N_VALUES = (10, 100, 1000, 5000, 10000)
T_SPAN = 100.0     # baseline of the data [d]
YERR = 1e-4
N_PREDICT = 200
N_REPEATS = 5
OPERATIONS = ("build", "loglike", "predict")
OUTPUT = paths.data / f"benchmark_gp_solver_{ARGS.device}.json"


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
        call()  # warm-up: compiles the operations the build runs one at a time
    elif op == "loglike":
        gp = solver(n)
        theta = jnp.asarray(gp.theta0)
        # Compile without calling: on the CPU the memory of the first call stays
        # resident, so after a warm-up call the peak would leave out the largest
        # buffers. The first call is timed with the others.
        loglike = gp.log_likelihood_fn.lower(theta).compile()
        call = lambda: loglike(theta).block_until_ready()
    elif op == "predict":
        gp = solver(n)
        xpred = np.linspace(0, T_SPAN, N_PREDICT)
        call = lambda: jax.block_until_ready(gp.predict(xpred))
        call()  # warm-up
    with benchmarking.peak_memory(ARGS.device) as mem:
        s = benchmarking.median_time(call, N_REPEATS)
    print(json.dumps(dict(s=s, MB=mem.mb)))


def main():
    benchmarking.check_spotgp_pin()
    rows = []
    for n in N_VALUES:
        row = dict(N=n)
        for op in OPERATIONS:
            try:
                result = benchmarking.run_worker(__file__, "--device", ARGS.device,
                                                 "--worker", op, n)
                row[f"{op}_s"], row[f"{op}_MB"] = result["s"], result["MB"]
            except RuntimeError as error:  # running out of GPU memory is a result; anything else is not
                if "RESOURCE_EXHAUSTED" not in str(error):
                    raise
                row[f"{op}_s"] = row[f"{op}_MB"] = None
        rows.append(row)
        print(f"N={n:>5d}  " + "  ".join(
            f"{op} out of memory" if row[f"{op}_s"] is None else
            f"{op} {row[f'{op}_s']:.4g} s ({row[f'{op}_MB']:.1f} MB)" for op in OPERATIONS))
    params = dict(hparam=HPARAM, n_harmonics=N_HARMONICS, n_lat=solver(10).n_lat,
                  matrix_solver="cholesky_full", t_span=T_SPAN, yerr=YERR,
                  n_predict=N_PREDICT, n_repeats=N_REPEATS)
    benchmarking.save(OUTPUT, params, rows, device=benchmarking.device_name(ARGS.device))


if __name__ == "__main__":
    if ARGS.worker:
        worker(ARGS.worker[0], int(ARGS.worker[1]))
    else:
        main()
