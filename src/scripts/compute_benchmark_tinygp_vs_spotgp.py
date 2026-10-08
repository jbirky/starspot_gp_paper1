"""
Benchmark: tinygp (celerite O(N) solver) vs george (full O(N^3) solver)
vs spotgp (banded Cholesky) for a one-sided exponential envelope starspot
kernel.

The one-sided exponential envelope Gamma(t) = exp(-a*t)*H(t) gives
R_Gamma(tau) = (1/(2a)) * exp(-a|tau|), which is celerite-representable.
The full kernel with N_harm harmonics is:

  K(tau) = sigma_k^2 * R_Gamma(tau) * [|c0|^2 + 2*sum_n |cn|^2 cos(n*w0*tau)]

This decomposes into J = N_harm + 1 celerite terms (1 real + N_harm complex).

We compare wall-clock time for log-likelihood evaluation as a function of
the number of datapoints N = T_obs / dt.

Times are measured on the GPU for the JAX solvers (tinygp, spotgp) and on the CPU
for george, so this script needs CUDA-enabled JAX and is not run in the
showyourwork build (whose environment has CPU-only JAX). Run it by hand on a
machine with an NVIDIA GPU; plot_benchmark_tinygp_vs_spotgp.py plots the result.

Output:
    data/benchmark_tinygp_vs_spotgp.npz
"""
import os
import platform
import subprocess
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cuda,cpu")
_ = jax.devices("cuda")

import paths
import spotgp
from spotgp.gp_solver import GPSolver
from spotgp.analytic_kernel import AnalyticKernel
from spotgp.spot_model import SpotEvolutionModel
from spotgp.visibility import VisibilityFunction, EdgeOnVisibilityFunction
from spotgp.envelope import ExponentialEnvelope

import tinygp
from tinygp import GaussianProcess, kernels

import george
from george import kernels as george_kernels

# ── Parameters ────────────────────────────────────────────────────
# Kernel hyperparameters
peq = 5.0           # rotation period [days]
inc = np.pi / 4     # stellar inclination [rad]
kappa = 0.0         # differential rotation
tau_spot = 3.0      # one-sided exponential decay timescale [days]
sigma_k = 0.01      # kernel amplitude
n_harmonics = 2     # number of visibility harmonics

# Observation settings
TSAMP = 1.0         # sampling cadence [days]
SIGMA_N = 1e-3      # white noise level

# Observation durations to benchmark
TSIM_VALUES = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])
N_REPEAT = 5        # number of timing repetitions

# ── One-sided exponential envelope ───────────────────────────────
class OneSidedExponentialEnvelope(ExponentialEnvelope):
    """
    Gamma(t) = exp(-t / tau_spot) H(t), so R_Gamma(lag) = (tau_spot / 2) exp(-|lag| / tau_spot).

    spotgp's ExponentialEnvelope is two-sided, exp(-|t| / tau_spot), whose
    R_Gamma = (tau_spot + |lag|) exp(-|lag| / tau_spot) is not a single celerite term.
    """

    def Gamma(self, t):
        t = jnp.asarray(t, dtype=float)
        return jnp.where(t >= 0, jnp.exp(-t / self._tau_spot), 0.0)

    def Gamma_hat(self, omega):
        """|FT[Gamma]| = tau_spot / sqrt(1 + (omega * tau_spot)^2)."""
        omega = jnp.asarray(omega, dtype=float)
        return self._tau_spot / jnp.sqrt(1.0 + (omega * self._tau_spot) ** 2)

    def R_Gamma(self, lag):
        return self.r_gamma_jax(jnp.array([self._tau_spot]), jnp.asarray(lag, dtype=float))

    def r_gamma_jax(self, theta_env, lag):
        tau_spot = theta_env[0]
        return 0.5 * tau_spot * jnp.exp(-jnp.abs(lag) / tau_spot)

    # the parent's closed forms are for the two-sided envelope
    def sympy_Gamma(self):
        return None

    def sympy_Gamma_hat(self):
        return None

    def sympy_R_Gamma(self):
        return None


model_spotgp = SpotEvolutionModel(
    envelope=OneSidedExponentialEnvelope(tau_spot),
    visibility=VisibilityFunction(peq=peq, kappa=kappa, inc=inc),
    sigma_k=sigma_k,
)

# ── Compute visibility coefficients at a reference latitude ──────
# For a simple benchmark we use latitude-averaged |cn|^2 from spotgp.
# These are computed once and reused for the tinygp kernel construction.
ak = AnalyticKernel(model_spotgp, n_harmonics=n_harmonics)

# Extract latitude-averaged |cn|^2 by evaluating the kernel and fitting
# We compute the kernel at specific lags to extract the coefficients.
omega0 = 2 * np.pi / peq
test_lags = jnp.linspace(0, 4 * peq, 5000)
k_test = np.array(ak.kernel(test_lags))

# R_Gamma for one-sided exponential: (1/(2a)) * exp(-a|tau|), a = 1/tau_spot
a = 1.0 / tau_spot
R_Gamma_test = (1.0 / (2 * a)) * np.exp(-a * np.array(test_lags))

# The kernel = sigma_k^2 * R_Gamma * [c0^2 + 2*c1^2*cos(w0*t) + 2*c2^2*cos(2*w0*t)]
# Fit the cn^2 coefficients by least squares
A_matrix = np.column_stack([
    R_Gamma_test,
    2 * R_Gamma_test * np.cos(omega0 * np.array(test_lags)),
    2 * R_Gamma_test * np.cos(2 * omega0 * np.array(test_lags)),
]) * sigma_k**2

cn_sq, _, _, _ = np.linalg.lstsq(A_matrix, k_test, rcond=None)
c0_sq, c1_sq, c2_sq = cn_sq
print(f"Fitted |cn|^2: c0^2={c0_sq:.6f}, c1^2={c1_sq:.6f}, c2^2={c2_sq:.6f}")
# tinygp and george time the same kernel as spotgp only if the fit is exact
fit_err = np.max(np.abs(A_matrix @ cn_sq - k_test)) / np.max(np.abs(k_test))
assert fit_err < 1e-8 and np.all(cn_sq >= 0), \
    f"spotgp kernel is not the celerite form assumed here (fit error {fit_err:.1e}, |cn|^2 {cn_sq})"


# ── Build tinygp celerite kernel ─────────────────────────────────
# K(tau) = sigma_k^2 * (1/(2a)) * exp(-a|tau|) * [c0^2 + 2*c1^2*cos(w0*tau) + ...]
#
# Expanding:
#   = sigma_k^2 * c0^2/(2a) * exp(-a|tau|)                           [Real term]
#   + sigma_k^2 * c1^2/a    * exp(-a|tau|) * cos(w0*tau)             [Celerite term]
#   + sigma_k^2 * c2^2/a    * exp(-a|tau|) * cos(2*w0*tau)           [Celerite term]
#
# tinygp.kernels.quasisep.Exp(scale=1/a) gives exp(-a|tau|)
# tinygp.kernels.quasisep.Celerite(a, b, c, d) gives:
#   (a+b)*exp(-c*t)*cos(d*t) + (a-b)*exp(-c*t)*sin(d*t)
# For pure cosine (no sine): a = amplitude/2, b = amplitude/2, c = a_decay, d = omega

def build_tinygp_kernel():
    """Construct the celerite kernel for tinygp.

    Celerite requires each term to be positive semi-definite, so negative
    fitted |cn|^2 coefficients are clamped to zero.
    """
    amp_dc = float(sigma_k**2 * max(c0_sq, 0) / (2 * a))
    amp_n1 = float(sigma_k**2 * max(c1_sq, 0) / a)
    amp_n2 = float(sigma_k**2 * max(c2_sq, 0) / a)

    # DC term: Real exponential decay
    # Exp(sigma, scale) gives sigma^2 * exp(-|tau|/scale)
    kernel = amp_dc * kernels.quasisep.Exp(scale=tau_spot)

    # n=1 harmonic: exp(-a|tau|) * cos(omega0 * tau)
    # Celerite(a, b, c, d): (a+b)*exp(-c*t)*cos(d*t) + (a-b)*exp(-c*t)*sin(d*t)
    # For pure cosine: set a = b = amplitude/2
    if amp_n1 > 0:
        kernel += kernels.quasisep.Celerite(
            a=amp_n1 / 2, b=amp_n1 / 2, c=a, d=omega0
        )

    # n=2 harmonic: exp(-a|tau|) * cos(2*omega0 * tau)
    if amp_n2 > 0:
        kernel += kernels.quasisep.Celerite(
            a=amp_n2 / 2, b=amp_n2 / 2, c=a, d=2 * omega0
        )

    return kernel


def build_tinygp_gp(x, yerr):
    """Build a tinygp GaussianProcess."""
    kernel = build_tinygp_kernel()
    return GaussianProcess(kernel, x, diag=yerr**2)


# JIT-compile the tinygp log-probability
@jax.jit
def tinygp_log_prob(x, y, yerr):
    gp = build_tinygp_gp(x, yerr)
    return gp.log_probability(y)


# ── Build george kernel ───────────────────────────────────────────
# george.kernels.ExpKernel(metric) gives exp(-|tau|/sqrt(metric)),
# so metric = tau_spot^2 for exp(-|tau|/tau_spot).
# george.kernels.CosineKernel(log_period) gives cos(2*pi*|tau|/period).
# We need cos(n*omega0*tau) = cos(2*pi*n*tau/peq), so period = peq/n.

def build_george_gp(x, yerr):
    """Build a george GP with the same kernel decomposition."""
    amp_dc = float(sigma_k**2 * max(c0_sq, 0) / (2 * a))
    amp_n1 = float(sigma_k**2 * max(c1_sq, 0) / a)
    amp_n2 = float(sigma_k**2 * max(c2_sq, 0) / a)

    metric = tau_spot**2  # ExpKernel: exp(-|tau|/sqrt(metric))

    # DC term: amplitude * exp(-|tau|/tau_spot)
    kernel = amp_dc * george_kernels.ExpKernel(metric=metric)

    # n=1 harmonic: exp(-|tau|/tau_spot) * cos(2*pi*tau/peq)
    if amp_n1 > 0:
        kernel += amp_n1 * george_kernels.ExpKernel(metric=metric) \
                  * george_kernels.CosineKernel(log_period=np.log(peq))

    # n=2 harmonic: exp(-|tau|/tau_spot) * cos(2*pi*tau/(peq/2))
    if amp_n2 > 0:
        kernel += amp_n2 * george_kernels.ExpKernel(metric=metric) \
                  * george_kernels.CosineKernel(log_period=np.log(peq / 2))

    gp = george.GP(kernel)
    gp.compute(x, yerr=yerr)
    return gp


# ── Build edge-on SpotEvolutionModel (no latitude quadrature) ────
edgeon_model = SpotEvolutionModel(
    envelope=ExponentialEnvelope(tau_spot),
    visibility=EdgeOnVisibilityFunction(peq),
    sigma_k=sigma_k,
)

# ── Benchmark loop ───────────────────────────────────────────────
ndata_values = (TSIM_VALUES / TSAMP).astype(int)

times_tinygp = []
times_george = []
times_spotgp_banded = []
times_spotgp_edgeon_banded = []
times_spotgp_full = []
bandwidths_banded = []
bandwidths_edgeon = []

rng = np.random.default_rng(42)

for tsim in TSIM_VALUES:
    ndata = int(tsim / TSAMP)
    x_obs = jnp.arange(0, tsim, TSAMP, dtype=jnp.float64)
    ndata = len(x_obs)
    y_obs = jnp.array(rng.normal(0, SIGMA_N, ndata), dtype=jnp.float64)
    yerr_obs = jnp.full(ndata, SIGMA_N, dtype=jnp.float64)

    print(f"\ntsim={tsim} d, N={ndata}")

    # ── tinygp ──
    # Warmup: two calls to ensure JIT compilation is fully cached for this shape
    for _ in range(2):
        _ = tinygp_log_prob(x_obs, y_obs, yerr_obs).block_until_ready()

    t_list = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        _ = tinygp_log_prob(x_obs, y_obs, yerr_obs).block_until_ready()
        t_list.append(time.perf_counter() - t0)
    t_tinygp = np.median(t_list)
    times_tinygp.append(t_tinygp)
    print(f"  tinygp:        {t_tinygp:.4f} s")

    # ── george ──
    x_np = np.array(x_obs)
    y_np = np.array(y_obs)
    yerr_np = np.array(yerr_obs)
    if ndata <= 10000:
        try:
            # Warmup
            gp_george = build_george_gp(x_np, yerr_np)
            _ = gp_george.log_likelihood(y_np)

            # Time the full pipeline: kernel eval + Cholesky + solve
            t_list = []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                gp_george = build_george_gp(x_np, yerr_np)
                _ = gp_george.log_likelihood(y_np)
                t_list.append(time.perf_counter() - t0)
            t_george = np.median(t_list)
            times_george.append(t_george)
            print(f"  george:        {t_george:.4f} s")
        except Exception as e:
            print(f"  george:        FAILED ({e})")
            times_george.append(np.nan)
    else:
        times_george.append(np.nan)
        print(f"  george:        skipped (N={ndata} too large)")

    # ── spotgp banded ──
    try:
        gp_banded = GPSolver(
            np.array(x_obs), np.array(y_obs), np.array(yerr_obs),
            model_spotgp,
            matrix_solver="cholesky_banded",
            n_harmonics=n_harmonics,
        )
        bandwidths_banded.append(gp_banded.bandwidth)
        theta0 = gp_banded.theta0
        log_post_fn = gp_banded.log_posterior
        # Warmup: two calls to ensure JIT compilation is fully cached
        for _ in range(2):
            _ = log_post_fn(theta0).block_until_ready()

        t_list = []
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            _ = log_post_fn(theta0).block_until_ready()
            t_list.append(time.perf_counter() - t0)
        t_banded = np.median(t_list)
        times_spotgp_banded.append(t_banded)
        print(f"  spotgp banded: {t_banded:.4f} s (b={gp_banded.bandwidth})")
    except Exception as e:
        print(f"  spotgp banded: FAILED ({e})")
        times_spotgp_banded.append(np.nan)
        bandwidths_banded.append(np.nan)

    # ── spotgp edge-on banded ──
    # try:
    #     gp_edgeon = GPSolver(
    #         np.array(x_obs), np.array(y_obs), np.array(yerr_obs),
    #         edgeon_model,
    #         matrix_solver="cholesky_banded",
    #         n_harmonics=n_harmonics,
    #     )
    #     bandwidths_edgeon.append(gp_edgeon.bandwidth)
    #     theta0_eo = gp_edgeon.theta0
    #     log_post_fn_eo = gp_edgeon.log_posterior
    #     # Warmup: two calls to ensure JIT compilation is fully cached
    #     for _ in range(2):
    #         _ = log_post_fn_eo(theta0_eo).block_until_ready()

    #     t_list = []
    #     for _ in range(N_REPEAT):
    #         t0 = time.perf_counter()
    #         _ = log_post_fn_eo(theta0_eo).block_until_ready()
    #         t_list.append(time.perf_counter() - t0)
    #     t_eo = np.median(t_list)
    #     times_spotgp_edgeon_banded.append(t_eo)
    #     print(f"  spotgp edge-on banded: {t_eo:.4f} s (b={gp_edgeon.bandwidth})")
    # except Exception as e:
    #     print(f"  spotgp edge-on banded: FAILED ({e})")
    #     times_spotgp_edgeon_banded.append(np.nan)
    #     bandwidths_edgeon.append(np.nan)

    # ── spotgp full (only for small N) ──
    if ndata <= 2000:
        try:
            gp_full = GPSolver(
                np.array(x_obs), np.array(y_obs), np.array(yerr_obs),
                model_spotgp,
                matrix_solver="cholesky_full",
                n_harmonics=n_harmonics,
            )
            theta0_full = gp_full.theta0
            log_post_fn_full = gp_full.log_posterior
            # Warmup: two calls to ensure JIT compilation is fully cached
            for _ in range(2):
                _ = log_post_fn_full(theta0_full).block_until_ready()

            t_list = []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                _ = log_post_fn_full(theta0_full).block_until_ready()
                t_list.append(time.perf_counter() - t0)
            t_full = np.median(t_list)
            times_spotgp_full.append(t_full)
            print(f"  spotgp full:   {t_full:.4f} s")
        except Exception as e:
            print(f"  spotgp full:   FAILED ({e})")
            times_spotgp_full.append(np.nan)
    else:
        times_spotgp_full.append(np.nan)
        print(f"  spotgp full:   skipped (N={ndata} too large)")

# ── Save ──────────────────────────────────────────────────────────
def _spotgp_commit():
    """Commit of the spotgp source in use, if it is a git checkout."""
    try:
        return subprocess.run(["git", "-C", os.path.dirname(spotgp.__file__), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown commit"


def _cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


outpath = paths.data / "benchmark_tinygp_vs_spotgp.npz"
np.savez(
    outpath,
    tsim=TSIM_VALUES, tsamp=TSAMP, n_harmonics=n_harmonics, ndata=ndata_values,
    times_tinygp=np.array(times_tinygp, dtype=float),
    times_george=np.array(times_george, dtype=float),
    times_spotgp_banded=np.array(times_spotgp_banded, dtype=float),
    times_spotgp_full=np.array(times_spotgp_full, dtype=float),
    bandwidths_banded=np.array(bandwidths_banded, dtype=float),
    # provenance of the timings
    gpu=jax.devices()[0].device_kind, cpu=_cpu_name(),
    versions=(f"spotgp {spotgp.__version__} ({_spotgp_commit()}), jax {jax.__version__}, "
              f"tinygp {tinygp.__version__}, george {george.__version__}"),
    date=time.strftime("%Y-%m-%d"),
)
print(f"\nSaved timings to {outpath}")
