"""
Decomposition of the single-spot projected area signal A(t)
into the normalized squared spot-size envelope Gamma(t) = alpha^2(t) / alpha_max^2
and the visibility function Pi(t) = max{cos(beta(t)), 0}, for the trapezoidal
and exponential envelope models (2 columns).
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'


# ── Envelope functions ──────────────────────────────────────────

def trapezoidal_alpha(t, lspot, tau_em, tau_dec, alpha_max):
    """Trapezoidal spot-size evolution centered at t=0."""
    dt1 = t + lspot / 2 + tau_em
    dt2 = t + lspot / 2
    dt3 = t - lspot / 2
    dt4 = t - lspot / 2 - tau_dec

    alpha = (dt1 * np.heaviside(dt1, 1) - dt2 * np.heaviside(dt2, 1)) / tau_em
    alpha += -(dt3 * np.heaviside(dt3, 1) - dt4 * np.heaviside(dt4, 1)) / tau_dec
    alpha *= alpha_max
    return alpha


def exponential_alpha(t, tau_max, tau_em, tau_dec, alpha_max):
    """
    Piecewise exponential spot-size evolution (C^0), centered at t=0.
    Rise: exp(t/tau_em) for t < -tau_max/2
    Plateau: alpha_max for -tau_max/2 <= t <= tau_max/2
    Decay: exp(-(t)/tau_dec) for t > tau_max/2
    """
    t1 = -tau_max / 2
    t2 = tau_max / 2
    alpha = np.where(
        t < t1,
        alpha_max * np.exp((t - t1) / tau_em),
        np.where(
            t <= t2,
            alpha_max,
            alpha_max * np.exp(-(t - t2) / tau_dec)
        )
    )
    return alpha


# ── Visibility function ─────────────────────────────────────────

def visibility(t, inc, Phi, Lambda0, P):
    """Pi(t) = max{cos(beta(t)), 0}."""
    Lambda = Lambda0 + 2 * np.pi * t / P
    cos_beta = np.cos(inc) * np.sin(Phi) + np.sin(inc) * np.cos(Phi) * np.cos(Lambda)
    return np.maximum(cos_beta, 0)


# ── Parameters ──────────────────────────────────────────────────

P = 3.0           # rotation period [days]
inc = np.pi / 2   # inclination (edge-on)
Phi = np.radians(30)  # spot latitude
Lambda0 = 0.0     # initial longitude
alpha_max = 0.1   # max angular radius [rad]

# Trapezoidal params
lspot_trap = 20.0     # plateau duration [days]
tau_em_trap = 5.0     # emergence timescale
tau_dec_trap = 5.0    # decay timescale

# Exponential params
tau_max_exp = 20.0
tau_em_exp = 2.5
tau_dec_exp = 2.5

# Time array
total_dur = lspot_trap + tau_em_trap + tau_dec_trap + 10
t = np.linspace(-total_dur / 2, total_dur / 2, 8000)

# ── Compute signals ─────────────────────────────────────────────

alpha_trap = trapezoidal_alpha(t, lspot_trap, tau_em_trap, tau_dec_trap, alpha_max)
alpha_exp = exponential_alpha(t, tau_max_exp, tau_em_exp, tau_dec_exp, alpha_max)
Gamma_trap = (alpha_trap / alpha_max)**2
Gamma_exp = (alpha_exp / alpha_max)**2
Pi = visibility(t, inc, Phi, Lambda0, P)
A_trap = Gamma_trap * Pi
A_exp = Gamma_exp * Pi

# ── Figure ──────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True,
                         gridspec_kw={'hspace': 0.08, 'wspace': 0.25})

label_fs = 16
title_fs = 18
tick_fs = 12

titles = ['Trapezoidal', 'Exponential']
Gammas = [Gamma_trap, Gamma_exp]
As = [A_trap, A_exp]

for col in range(2):
    # Top: Gamma(t)
    axes[0, col].plot(t, Gammas[col], color='black', lw=1.5)
    axes[0, col].set_title(titles[col], fontsize=title_fs, pad=8)
    axes[0, col].set_ylim(-0.05, 1.15)
    axes[0, col].tick_params(labelsize=tick_fs)
    if col == 0:
        axes[0, col].set_ylabel(r'$\Gamma(t)$', fontsize=label_fs)

    # Middle: Pi(t)
    axes[1, col].plot(t, Pi, color='black', lw=1.0)
    axes[1, col].axhline(0, color='gray', ls='--', lw=0.6, alpha=0.5)
    axes[1, col].set_ylim(-0.05, 1.1)
    axes[1, col].tick_params(labelsize=tick_fs)
    if col == 0:
        axes[1, col].set_ylabel(r'$\mathcal{V}(t) = \max\{\cos\beta,\, 0\}$', fontsize=label_fs)

    # Bottom: A(t) = Gamma(t) * Pi(t)
    axes[2, col].plot(t, As[col], color='black', lw=1.2)
    axes[2, col].set_xlabel('Time [days]', fontsize=label_fs)
    axes[2, col].set_ylim(-0.02, max(A.max() for A in As) * 1.15)
    axes[2, col].tick_params(labelsize=tick_fs)
    if col == 0:
        axes[2, col].set_ylabel(r'$A(t) \propto \Gamma(t) \cdot \mathcal{V}(t)$', fontsize=label_fs)

    for row in range(3):
        axes[row, col].set_xlim(t[0], t[-1])

fig.savefig(paths.figures / "spot_decomposition_panel.pdf", bbox_inches="tight", dpi=300)
