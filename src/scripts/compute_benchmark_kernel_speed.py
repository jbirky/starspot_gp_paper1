"""
Benchmark of the kernel computation time (Table tab:kernel_speed): the analytic kernel
(AnalyticKernel.kernel on N lags) against the numerical Monte Carlo kernel
(NumericalKernel: N_sim simulated lightcurves of N points over T_SIM days, and the
average of their autocovariance), for each N_sim in NSIM.

Everything runs on one CPU core. An analytic time is the median of N_REPEATS calls
after a warm-up call that JIT-compiles; a numerical kernel, which takes seconds, is
timed once.

Run by hand with the pinned spotgp, since the timings depend on the machine; e.g. in
the showyourwork env:
    python compute_benchmark_kernel_speed.py
table_benchmarks.py makes the table from the result.

Output:
    data/benchmark_kernel_speed.json
"""
import time

import benchmarking
benchmarking.use_cpu()

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import paths
from spotgp.analytic_kernel import AnalyticKernel
from spotgp.numerical_kernel import NumericalKernel

HPARAM = dict(peq=5.0, kappa=0.0, inc=np.pi / 2, nspot=10, lspot=15.0, tau_spot=3.0,
              alpha_max=0.1, fspot=0.0)
N_HARMONICS = 3
N_VALUES = (50, 100, 500, 1000, 5000)  # lags (analytic) = time points per simulation (numerical)
NSIM = (200, 500)                      # simulations per numerical kernel
T_SIM = 100.0                          # simulated baseline [d]
N_REPEATS = 5
OUTPUT = paths.data / "benchmark_kernel_speed.json"


def time_analytic(n):
    kernel = AnalyticKernel(HPARAM, n_harmonics=N_HARMONICS)
    lag = np.linspace(0, T_SIM, n)
    call = lambda: jax.block_until_ready(kernel.kernel(lag))
    call()  # warm-up: JIT compilation
    return benchmarking.median_time(call, N_REPEATS)


def time_numerical(n, nsim):
    np.random.seed(42)
    t0 = time.perf_counter()
    kernel = NumericalKernel(HPARAM, tsim=T_SIM, tsamp=T_SIM / n, nsim=nsim, verbose=False)
    dt = time.perf_counter() - t0
    assert len(kernel.tarr) == n, f"{len(kernel.tarr)} simulated points, expected {n}"
    return dt


def main():
    benchmarking.check_spotgp_pin()
    rows = []
    for n in N_VALUES:
        row = dict(N=n, analytic_s=time_analytic(n))
        for nsim in NSIM:
            row[f"numerical_{nsim}_s"] = time_numerical(n, nsim)
        rows.append(row)
        print(f"N={n:>5d}  analytic {row['analytic_s']:.4f} s  "
              + "  ".join(f"numerical({nsim}) {row[f'numerical_{nsim}_s']:.2f} s" for nsim in NSIM))
    params = dict(hparam=HPARAM, n_harmonics=N_HARMONICS, nsim=NSIM, t_sim=T_SIM,
                  n_repeats=N_REPEATS)
    benchmarking.save(OUTPUT, params, rows, device=f"1 core of {benchmarking.cpu_name()}")


if __name__ == "__main__":
    main()
