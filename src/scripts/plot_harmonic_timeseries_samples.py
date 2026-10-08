"""
Draw and plot GP time series samples of the individual n = 0, 1, 2, ...
rotation harmonics, using spotgp ``SpotTerm`` objects and ``Term.sample``.

The companion to `plot_kernel_harmonic_components.py`: that script decomposes
the kernel k(tau), this one decomposes the lightcurves the kernel generates.

The decomposition is exact and the components are independently samplable.
Writing the kernel (Eq. 66) as a sum over harmonics,

    K(tau) = sum_n K_n(tau),
    K_n(tau) = sigma_k^2 int R_Gamma(tau) w_n |c_n(I,Phi)|^2
                            cos(n omega_0(Phi) tau) p(Phi) dPhi

with w_0 = 1 and w_n = 2 for n >= 1, each K_n is separately a valid
(positive semi-definite) kernel: the n = 0 term is R_Gamma times a positive
constant, and its transform is |Gamma_hat|^2 >= 0; the n >= 1 terms have
transform [S_Gamma(w - n w_0) + S_Gamma(w + n w_0)] / 2 >= 0, a shifted copy
of the same non-negative spectrum.  A positive mixture over latitude preserves
this, so the statement survives differential rotation.

Because covariances add for independent processes, drawing y_n ~ GP(0, K_n)
independently and summing gives a draw from the full GP:

    sum_n y_n  ~  GP(0, sum_n K_n)  =  GP(0, K).

So the panels below are not merely illustrative -- the components really do
add up to a valid lightcurve, and the script asserts sum_n K_n == K to
machine precision before plotting.

Isolating a harmonic uses the ``harmonics`` argument of `VisibilityFunction`:
``harmonics=[n]`` keeps exactly that order's term in the latitude integral,
and the `SpotTerm` built on it exposes the resulting kernel as
``Term.k_of_lag`` and draws realizations with ``Term.sample``.  Kernel terms
compose additively, so no modification to the kernel, quadrature, or solver
is needed.

Output:
    figures/fig_harmonic_timeseries_samples.pdf
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib

import paths  # noqa: F401  (registers spotgp submodules)
from spotgp.envelope import TrapezoidSymmetricEnvelope
from spotgp.spot_model import SpotEvolutionModel
from spotgp.terms import SpotTerm
from spotgp.visibility import VisibilityFunction

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14


# ── Parameters ───────────────────────────────────────────────────
P_eq = 5.0            # rotation period [days]
kappa = 0.0           # solid body
inc_deg = 60.0        # stellar inclination
ell = 15.0            # spot plateau duration [days]
tau_s = 3.0           # rise/decay timescale [days]
sigma_k = 1.0
N_HARM = 2            # keep n = 0, 1, 2
T_OBS, N_PTS = 60.0, 1200
SEED = 42

# Gauss-Legendre, not the default trapezoid: the latitude integrand has kinks
# at Phi = +/- I, where the trapezoid rule biases K(0) high by several percent
# at this inclination.  The nodes span the latitude distribution's full
# range (the default uniform [-pi/2, pi/2]).
QUAD = dict(quadrature="gauss-legendre", n_lat=64)


def make_term(orders):
    """
    Build a SpotTerm whose kernel keeps only the harmonic orders ``orders``.

    Parameters
    ----------
    orders : sequence of int
        Harmonic orders retained by the visibility function.  A single
        order isolates that harmonic; ``range(N_HARM + 1)`` gives the
        full kernel.

    Returns
    -------
    SpotTerm
    """
    inc = np.radians(inc_deg)
    model = SpotEvolutionModel(
        envelope=TrapezoidSymmetricEnvelope(lspot=ell, tau_spot=tau_s),
        visibility=VisibilityFunction(peq=P_eq, kappa=kappa, inc=inc,
                                      harmonics=list(orders)),
        sigma_k=sigma_k)
    # The kernel takes its harmonic orders from the visibility function.
    return SpotTerm(model, **QUAD)


def kernel_of(term, lag):
    """Evaluate a term's kernel k(lag) at its default parameters."""
    theta = jnp.asarray(term.theta0)
    return np.asarray(term.k_of_lag(theta, jnp.asarray(lag, dtype=float)))


if __name__ == "__main__":
    t = np.linspace(0, T_OBS, N_PTS)

    term_full = make_term(range(N_HARM + 1))
    terms = {n: make_term([n]) for n in range(N_HARM + 1)}

    # ── Check the decomposition is exact before trusting the figure ──
    lags = np.linspace(0, T_OBS, 501)
    k_full = kernel_of(term_full, lags)
    k_parts = {n: kernel_of(term, lags) for n, term in terms.items()}
    resid = np.abs(sum(k_parts.values()) - k_full).max()
    print(f"max |sum_n K_n - K_full| = {resid:.3e}  "
          f"(K(0) = {k_full[0]:.4f})")
    assert resid < 1e-8 * max(k_full[0], 1.0), \
        "harmonic components do not sum to the full kernel"

    # ── Draw one independent realization per harmonic ────────────────
    # Independent seeds: the components must be uncorrelated for their sum
    # to be a draw from K_full.  Sharing a seed would correlate them.
    parts = {n: term.sample(term.theta0, t, n_samples=1, seed=SEED + n)
             for n, term in terms.items()}
    total = sum(parts.values())

    # An independent draw from the full kernel, for visual comparison: the
    # sum and this draw are different realizations of the *same* process.
    direct = term_full.sample(term_full.theta0, t, n_samples=1,
                              seed=SEED + 100)

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(N_HARM + 2, 1, figsize=(9, 10), sharex=True,
                             gridspec_kw=dict(hspace=0.12))
    colors = ['C0', 'C1', 'C2']
    labels = {0: r'$n=0$ (DC, aperiodic)',
              1: r'$n=1$ (fundamental)',
              2: r'$n=2$ (second harmonic)'}

    for n in range(N_HARM + 1):
        ax = axes[n]
        ax.plot(t, parts[n], color=colors[n], lw=0.9)
        ax.axhline(0, color='0.8', lw=0.5, zorder=0)
        frac = k_parts[n][0] / k_full[0]
        ax.set_ylabel(labels.get(n, rf'$n={n}$'))
        ax.text(0.995, 0.06,
                rf'$\mathcal{{K}}_{n}(0)/\mathcal{{K}}(0) = {frac:.3f}$',
                transform=ax.transAxes, ha='right', fontsize=10)

    axes[N_HARM + 1].plot(t, total, color='k', lw=0.9)
    axes[N_HARM + 1].axhline(0, color='0.8', lw=0.5, zorder=0)
    axes[N_HARM + 1].set_ylabel(r'$\sum_n$ (sum of above)')

    # axes[N_HARM + 2].plot(t, direct, color='0.35', lw=0.9)
    # axes[N_HARM + 2].axhline(0, color='0.8', lw=0.5, zorder=0)
    # axes[N_HARM + 2].set_ylabel('direct draw\nfrom full $\\mathcal{K}$',
    #                             fontsize=9)
    axes[N_HARM + 1].set_xlabel('Time [days]')

    for m in range(1, int(T_OBS / P_eq) + 1):
        for ax in axes:
            ax.axvline(m * P_eq, color='0.85', lw=0.5, ls=':', zorder=0)

    # Panels are short, so let matplotlib pick 3 ticks rather than packing in
    # a label per unit -- the default density is unreadable at this height.
    for ax in axes:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.tick_params(labelsize=10)

    axes[0].set_xlim(0, T_OBS)
    axes[0].set_title(
        rf'Harmonic decomposition of GP samples '
        rf'($I={inc_deg:.0f}^\circ$, $P_{{\rm eq}}={P_eq:.0f}$ d, '
        rf'$\kappa={kappa:.0f}$)', fontsize=14)

    fig.savefig(paths.figures / "fig_harmonic_timeseries_samples.pdf",
                bbox_inches='tight')
    print("Saved fig_harmonic_timeseries_samples.pdf")

    # ── Variance budget ──────────────────────────────────────────────
    print(f"\nvariance budget at I = {inc_deg:.0f} deg:")
    for n in range(N_HARM + 1):
        print(f"  n={n}: K_n(0)/K(0) = {k_parts[n][0]/k_full[0]:7.4f}")
    print(f"  sum  : {sum(k_parts[n][0] for n in k_parts)/k_full[0]:7.4f}")
