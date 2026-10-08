import os
import numpy as np
import time
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import paths

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cuda")
_ = jax.devices("cuda")
jax.block_until_ready(jax.jit(lambda x: x + 1)(jnp.zeros(1, dtype=jnp.float64)))

from spotgp import (TimeSeriesData, TrapezoidSymmetricEnvelope, VisibilityFunction,
                    SpotEvolutionModel, GPSolver, LightcurveModel)
from spotgp.mcmc import DynestySampler

# ===================================================================
# CLI arguments
# python compute_mcmc_simulated_dynesty.py --nlive=5000 --dlogz=0.5 --tsim=365.0 --inc_deg=90
# python compute_mcmc_simulated_dynesty.py --nlive=5000 --dlogz=0.5 --tsim=365.0 --inc_deg=60
# python compute_mcmc_simulated_dynesty.py --nlive=5000 --dlogz=0.5 --tsim=365.0 --inc_deg=30
# python compute_mcmc_simulated_dynesty.py --nlive=5000 --dlogz=0.5 --tsim=1460.0 --inc_deg=90
# python compute_mcmc_simulated_dynesty.py --nlive=5000 --dlogz=0.5 --tsim=1460.0 --inc_deg=60
# python compute_mcmc_simulated_dynesty.py --nlive=5000 --dlogz=0.5 --tsim=1460.0 --inc_deg=30
# ===================================================================

import argparse
parser = argparse.ArgumentParser(description="Run dynesty nested sampling on simulated starspot light curve")
# data generation
parser.add_argument("--tsim", type=float, default=365.0, help="Simulation duration in days (default: 365)")
parser.add_argument("--tsamp", type=float, default=1.0, help="Sampling cadence in days (default: 0.5)")
parser.add_argument("--sigma_n", type=float, default=1e-4, help="Observational noise standard deviation (default: 1e-4)")
# parser.add_argument("--nspot_rate", type=float, default=0.25, help="Spot emergence rate in spots/day (default: 0.25)")
parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation (default: 42)")
parser.add_argument("--inc_deg", type=float, default=90, help="Inclination in degrees (default: 90)")
parser.add_argument("--peq", type=float, default=5.0, help="Equatorial rotation period in days (default: 5.0)")
# dynesty options
parser.add_argument("--nlive", type=int, default=1000, help="Number of live points (default: 500)")
parser.add_argument("--dlogz", type=float, default=1.0, help="Stopping criterion for evidence tolerance (default: 1.0)")
parser.add_argument("--bound", type=str, default="multi", help="Bounding method (default: multi)")
parser.add_argument("--sample", type=str, default="rwalk", help="Sampling method (default: rwalk)")
parser.add_argument("--bootstrap", type=int, default=None, help="Number of bootstrap iterations for enlargement (0 to disable, default: dynesty auto)")
parser.add_argument("--dynamic", action="store_true", help="Use dynamic nested sampling")
args = parser.parse_args()


if __name__ == '__main__':

    # ===================================================================
    # True parameters and simulated data
    # ===================================================================

    true_params = {
        "peq":      args.peq,
        "kappa":    0.3,
        "inc":      np.deg2rad(args.inc_deg),
        "lspot":    20.0,
        "tau_spot": 5.0,
        "nspot_rate": 0.25,
        "fspot":      0,
        "alpha_max":  0.08,
    }

    results_dir = f"{paths.data}/results_simulated_dynesty_mean/tsim_{args.tsim}_tsamp_{args.tsamp}_inc_{args.inc_deg}_peq_{args.peq}"
    os.makedirs(results_dir, exist_ok=True)

    # Generate synthetic light curve
    rng = np.random.default_rng(args.seed)
    envelope = TrapezoidSymmetricEnvelope(lspot=true_params["lspot"], tau_spot=true_params["tau_spot"])
    visibility = VisibilityFunction(peq=true_params["peq"], kappa=true_params["kappa"],
                                    inc=true_params["inc"])
    model = SpotEvolutionModel(envelope=envelope,
                                    visibility=visibility,
                                    nspot_rate=true_params["nspot_rate"],
                                    fspot=true_params["fspot"],
                                    alpha_max=true_params["alpha_max"])
    true_params["sigma_k"] = model.sigma_k

    lc = LightcurveModel.from_spot_model(model, nspot_rate=true_params["nspot_rate"],
                                         tsim=args.tsim, tsamp=args.tsamp)
    flux_true = lc.flux.copy()
    noise = rng.normal(0, args.sigma_n, size=len(flux_true))
    flux_obs = flux_true + noise
    flux_obs_zero = flux_obs - np.mean(flux_obs)
    yerr = np.full_like(flux_obs, args.sigma_n)

    ts = TimeSeriesData(x=lc.t, y=flux_obs_zero, yerr=yerr, normalize=False)

    print(f"Simulated data: N={ts.N}, baseline={ts.baseline:.1f} days, "
          f"cadence={ts.median_dt:.3f} days")
    print(f"True params: {true_params}")

    # Save simulated data
    np.savez(os.path.join(results_dir, "simulated_data.npz"),
             time=lc.t, flux_true=flux_true, flux_obs=flux_obs, yerr=yerr,
             true_params=true_params, seed=args.seed, sigma_n=args.sigma_n)
    
    plt.figure(figsize=[18,6])
    plt.errorbar(lc.t, flux_obs, yerr=yerr)
    plt.savefig(os.path.join(results_dir, "lc_data.png"))
    plt.close()

    # ===================================================================
    # Set up spot model and GP solver
    # ===================================================================

    bounds = {
        "peq":      (0.9 * true_params["peq"], 1.1 * true_params["peq"]),
        "kappa":    (-1.0, 1.0),
        "inc":      (true_params["inc"] - np.pi/6, true_params["inc"] + np.pi/6),
        "lspot":    (10.0, 30.0),
        "tau_spot": (0.0, 10.0),
        "sigma_k":  (1e-3, 1e-2),
    }

    gp = GPSolver(ts, model, bounds).build_jax()

    # Pre-compile the log-likelihood on the main process
    _dummy = gp.log_likelihood_fn(jnp.array(gp.theta0))
    jax.block_until_ready(_dummy)
    print(f"JAX log-likelihood compiled on {jax.devices()[0]}.")

    # ===================================================================
    # Set up DynestySampler
    # ===================================================================

    sampler = DynestySampler(gp, save_dir=results_dir)
    ndim = gp.n_params

    # ===================================================================
    # Run dynesty via DynestySampler
    # ===================================================================

    t0 = time.time()
    print(f"\nStarting dynesty nested sampling (nlive={args.nlive})...")
    print(f"  bound={args.bound}, sample={args.sample}, dlogz={args.dlogz}")
    print(f"  dynamic={args.dynamic}, ndim={ndim}")
    print(f"  params: {gp.param_keys}")

    if args.dynamic:
        samples, info = sampler.run_dynamic_sampling(
            nlive_init=args.nlive,
            dlogz_init=args.dlogz,
            wt_kwargs={'pfrac': 1.0},
        )
    else:
        sampling_kwargs = dict(
            nlive=args.nlive,
            dlogz=args.dlogz,
            bound=args.bound,
            sample=args.sample,
            print_progress=True,
        )
        if args.bootstrap is not None:
            sampling_kwargs["bootstrap"] = args.bootstrap
        samples, info = sampler.run_sampling(**sampling_kwargs)

    elapsed = time.time() - t0
    print(f"\nDynesty completed in {elapsed:.1f} seconds")

    # ===================================================================
    # Print summary and save results
    # ===================================================================

    sampler.summary()

    # Save full dynesty info alongside the checkpoint
    np.savez(os.path.join(results_dir, "dynesty_results.npz"),
             samples=samples,
             logz=info["logz"],
             logzerr=info["logzerr"],
             param_keys=list(gp.param_keys),
             true_params=true_params,
             bounds=bounds,
             elapsed=elapsed)

    np.save(os.path.join(results_dir, "posterior_samples.npy"), samples)

    # ===================================================================
    # Corner plot
    # ===================================================================

    try:
        import corner
        import matplotlib.pyplot as plt

        labels = list(gp.param_keys)
        truths = [true_params[k] for k in gp.param_keys]

        fig = corner.corner(samples, labels=labels, truths=truths,
                            quantiles=[0.16, 0.5, 0.84], show_titles=True)
        fig.savefig(os.path.join(results_dir, "corner_plot.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nCorner plot saved to {results_dir}/corner_plot.png")
    except ImportError:
        print("\ncorner not installed, skipping corner plot.")

    print(f"\nAll results saved to {results_dir}/")
