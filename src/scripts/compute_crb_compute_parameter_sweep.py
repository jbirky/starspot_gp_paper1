import os
import sys

# Prefer CUDA, fall back to CPU
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
# os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

try:
    devices = jax.devices()
except RuntimeError as err:
    # No CUDA plugin or no usable driver — retry on CPU alone.
    print(f"CUDA backend unavailable ({err}); falling back to CPU.")
    os.environ["JAX_PLATFORMS"] = "cpu"
    jax.config.update("jax_platforms", "cpu")
    devices = jax.devices()

print(f"JAX backend: {devices[0].platform} ({len(devices)} device(s))")

import numpy as np
import pandas as pd
import itertools
import tqdm
from scipy.stats.qmc import Sobol
import paths

from spotgp import (
    TrapezoidSymmetricEnvelope,
    VisibilityFunction,
    SpotEvolutionModel,
    TimeSeriesData,
    LightcurveModel,
    GPSolver,
)   


def pearson_from_cov(cov):                                                                  
    sigma = np.sqrt(np.diag(cov))                                                           
    return cov / np.outer(sigma, sigma) 

def build_crb_solver(ref_params, bounds, tsim=200, tsamp=0.25, sigma_n=1e-4, nspot_rate=0.25):
    """Build (and JAX-precompile) a single GPSolver reused for the whole sweep.

    With ``matrix_solver="cholesky_full"`` the Fisher information depends only on
    the lag grid, the noise level, the model's functional forms (envelope type,
    latitude distribution, n_harmonics/n_lat) and theta — never on the simulated
    flux.  Since ``tsim``/``tsamp``/``sigma_n`` are fixed across the sweep, one
    solver serves every Sobol sample; ``ref_params`` only sets the point at which
    the covariance is first factorized and JAX is warmed up.
    """

    envelope   = TrapezoidSymmetricEnvelope(lspot=ref_params["lspot"], tau_spot=ref_params["tau_spot"], numerical=True)
    visibility = VisibilityFunction(peq=ref_params["peq"], kappa=ref_params["kappa"], inc=ref_params["inc"])
    model      = SpotEvolutionModel(envelope=envelope, visibility=visibility, sigma_k=ref_params["sigma_k"])
    lc         = LightcurveModel.from_spot_model(model, nspot_rate=nspot_rate, tsim=tsim, tsamp=tsamp)
    obs        = TimeSeriesData(x=lc.t, y=lc.flux, yerr=sigma_n, normalize=False)
    gp         = GPSolver(obs, model, bounds=bounds, matrix_solver="cholesky_full").build_jax(recompute=False)

    return gp


def compute_crb_matrix(gp, true_params, eigval_floor=1e-6):
    """CRB covariance at ``true_params``, evaluated with the prebuilt solver."""

    theta   = jnp.array([float(true_params[k]) for k in gp.param_keys], dtype=jnp.float64)
    cov_crb = np.array(gp.mass_matrix_fisher(theta))

    # Check whether any Fisher eigenvalue was clipped by the regularization floor.
    # If so, the CRB is unreliable — return NaNs.
    # fisher = np.array(gp._fisher_matrix)
    # eigvals = np.linalg.eigvalsh(fisher)
    # if np.any(eigvals < eigval_floor * 1.01):
    #     n = cov_crb.shape[0]
    #     return np.full((n, n), np.nan)

    return cov_crb

# ===========================================================================================
# Configuration for parameter sweep

param_labels = {"peq": r"$P_{\rm eq}$ [d]", "kappa": r"$\kappa$","inc": r"$i$", "lspot": r"$\ell_{\rm spot}$ [d]", "tau_spot": r"$\tau_{\rm spot}$ [d]", "sigma_k":  r"$\sigma_k$",}                                                                  

TSIM = 365
TSAMP = 0.5
SIGMA_N = 1e-4
N_SPOT_RATE = 0.25

bounds = dict(
    peq      = (0.1, 50.),
    kappa    = (-1.0, 1.0),
    inc      = (np.deg2rad(0.0), np.deg2rad(90.0)),
    lspot    = (1.0, 100.),
    tau_spot = (1.0, 100.),
    sigma_k  = (1e-3,  1),
)

N_SAMPLES = 4096   # must be a power of 2 for Sobol sequences

param_names = list(param_labels.keys())
lo = np.array([bounds[k][0] for k in param_names])
hi = np.array([bounds[k][1] for k in param_names])

sampler   = Sobol(d=len(param_names), scramble=True)
unit_cube = sampler.random(N_SAMPLES)          # shape (N_SAMPLES, n_params) in [0, 1)
param_grid = lo + (hi - lo) * unit_cube        # scale to parameter bounds  

# ===========================================================================================
# Run parameter sweep and save results

# Build and warm up the solver once — the sweep only varies theta, so the lag
# grid, the Cholesky structure and the JAX compilation are all shared.
gp = build_crb_solver(dict(zip(param_names, param_grid[0])), bounds,
                      tsim=TSIM, tsamp=TSAMP, sigma_n=SIGMA_N, nspot_rate=N_SPOT_RATE)

assert tuple(gp.param_keys) == tuple(param_names), (
    f"solver theta layout {gp.param_keys} does not match sweep order {tuple(param_names)}")

results = []
for pg in tqdm.tqdm(param_grid):
    true_params = dict(zip(param_labels.keys(), pg))
    param_keys = list(true_params.keys())
    param_values = list(true_params.values())

    cov_crb = compute_crb_matrix(gp, true_params)
    sigma_crb = np.sqrt(np.diag(cov_crb))

    rdict = {}
    
    for key in param_keys:
        rdict[f"true_{key}"] = true_params[key]

    for key in param_keys:
        rdict[f"sigma_{key}"] = sigma_crb[param_keys.index(key)]
        
    pairs_idx = list(itertools.combinations(range(len(true_params)), 2))
    for i, j in pairs_idx:
        rdict[f"r_{param_keys[i]}_{param_keys[j]}"] = pearson_from_cov(cov_crb)[i, j]
        
    results.append(rdict)
    
results_final = pd.DataFrame(data={k: np.array([d[k] for d in results]) for k in results[0]})
results_final.to_csv(paths.data / f"crb_sobol_sweep_{TSIM}_{N_SAMPLES}.csv", index=False)
