"""Quick benchmark of full Cholesky on JAX for small N values."""
import os
os.environ["JAX_ENABLE_X64"] = "1"

import time
import numpy as np
import paths

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import jax
import jax.numpy as jnp
from spotgp.gp_solver import GPSolver
from spotgp.observations import TimeSeriesData

DT = 0.1
XMAX_VALUES = [10, 50, 100, 500, 1000]
hparam = dict(peq=5.0, kappa=0.0, inc=np.pi/2, lspot=15.0, tau_spot=3.0, sigma_k=0.01)

rows = []
for xmax in XMAX_VALUES:
    x = np.arange(0, xmax, DT)
    N = len(x)
    y = np.random.default_rng(42).normal(0, 1e-3, N)
    yerr = np.full(N, 1e-4)

    data = TimeSeriesData(x, y, yerr, normalize=False)

    # Warmup
    gp = GPSolver(data, model_or_hparam=hparam, kernel_type="analytic",
                  matrix_solver="cholesky_full")

    # Build
    times_build = []
    for _ in range(3):
        t0 = time.perf_counter()
        gp2 = GPSolver(data, model_or_hparam=hparam, kernel_type="analytic",
                       matrix_solver="cholesky_full")
        t1 = time.perf_counter()
        times_build.append(t1 - t0)
    build_s = np.median(times_build)

    # Log-likelihood
    _ = gp.log_likelihood()
    times_logl = []
    for _ in range(3):
        t0 = time.perf_counter()
        val = gp.log_likelihood()
        t1 = time.perf_counter()
        times_logl.append(t1 - t0)
    logl_s = np.median(times_logl)

    # Predict
    xpred = np.linspace(x[0], x[-1], 200)
    _ = gp.predict(xpred)
    times_pred = []
    for _ in range(3):
        t0 = time.perf_counter()
        gp.predict(xpred)
        t1 = time.perf_counter()
        times_pred.append(t1 - t0)
    pred_s = np.median(times_pred)

    print(f"N={N:>6d}  build={build_s:.4f}s  logL={logl_s:.4f}s  predict={pred_s:.4f}s")
    rows.append(dict(xmax=xmax, N=N, solver="full",
                     build_s=build_s, logL_s=logl_s, predict_s=pred_s))

import pandas as pd
df = pd.DataFrame(rows)
out = paths.output / "banded_vs_full_benchmark_jax_full.csv"
df.to_csv(out, index=False, float_format="%.4g")
print(f"Saved {out}")
