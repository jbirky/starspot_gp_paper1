"""
Compare three spot envelope models: trapezoidal, exponential, and skew-normal.

Three panels:
  1. Gamma(t) vs time
  2. |FT{Gamma}(omega)|^2 vs frequency
  3. R_Gamma(tau) vs time lag

Output:
    figures/fig_envelope_kernel_comparison.pdf
"""
import numpy as np
from scipy.special import erf as scipy_erf
import matplotlib.pyplot as plt
import matplotlib
import paths

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

# ── Parameters ───────────────────────────────────────────────────
alpha_max = 0.1       # peak angular radius [rad]
ell = 15.0            # plateau duration [days]  (trapezoidal / exponential)
tau_s = 3.0           # rise/decay timescale [days]
sigma_sn = 5.0        # skew-normal scale [days]
n_sn = -3.0           # skew-normal skewness (< 0: rapid rise / slow decay)

# ── Envelope definitions ────────────────────────────────────────

def Gamma_trapezoid(t, ell, tau_s):
    """Normalized squared trapezoidal envelope, centered at t=0."""
    abs_t = np.abs(t)
    L = ell / 2
    S = L + tau_s
    result = np.zeros_like(abs_t)
    plateau = abs_t <= L
    ramp = (abs_t > L) & (abs_t <= S)
    result[plateau] = 1.0
    result[ramp] = ((S - abs_t[ramp]) / tau_s) ** 2
    return result


def Gamma_exponential(t, ell, tau_s):
    """Normalized squared exponential envelope (C^0), centered at t=0."""
    L = ell / 2
    return np.where(
        t < -L,
        np.exp(2 * (t + L) / tau_s),
        np.where(t <= L, 1.0, np.exp(-2 * (t - L) / tau_s))
    )


def Gamma_skew_normal(t, sigma_sn, n_sn):
    """Normalized skew-normal envelope (Baranyi et al. 2021, Eq. 1)."""
    z = t / sigma_sn
    env = np.exp(-z**2 / 2.0) * (1.0 + scipy_erf(n_sn * z / np.sqrt(2.0)))
    env = np.maximum(env, 0.0)
    peak = env.max()
    return env / peak if peak > 0.0 else env


# ── R_Gamma via FFT-based autocorrelation ────────────────────────

def R_Gamma_fft(t_grid, Gamma_grid):
    """Compute R_Gamma(tau) = int Gamma(s) Gamma(s+tau) ds via FFT."""
    dt = t_grid[1] - t_grid[0]
    n = len(Gamma_grid)
    n_fft = 2 * n
    G_fft = np.fft.rfft(Gamma_grid, n=n_fft)
    R_full = np.fft.irfft(np.abs(G_fft)**2, n=n_fft)[:n] * dt
    lag_grid = np.arange(n) * dt
    return lag_grid, R_full


def Gamma_hat_sq_fft(t_grid, Gamma_grid):
    """Compute |FT{Gamma}(omega)|^2 via FFT."""
    dt = t_grid[1] - t_grid[0]
    n = len(Gamma_grid)
    n_fft = 2 * n
    G_fft = np.fft.rfft(Gamma_grid, n=n_fft) * dt
    Gh_sq = np.abs(G_fft)**2
    omega_grid = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=dt)
    return omega_grid, Gh_sq


# ── Compute everything ──────────────────────────────────────────

# Wide time grid
t_margin = max(8 * tau_s, 12 * sigma_sn)
t_grid = np.linspace(-ell / 2 - t_margin, ell / 2 + t_margin, 2**14)

# Gamma(t)
G_trap = Gamma_trapezoid(t_grid, ell, tau_s)
G_exp = Gamma_exponential(t_grid, ell, tau_s)
G_sn = Gamma_skew_normal(t_grid, sigma_sn, n_sn)

# |FT{Gamma}|^2
omega_trap, Ghsq_trap = Gamma_hat_sq_fft(t_grid, G_trap)
omega_exp, Ghsq_exp = Gamma_hat_sq_fft(t_grid, G_exp)
omega_sn, Ghsq_sn = Gamma_hat_sq_fft(t_grid, G_sn)

# R_Gamma(tau)
lag_trap, R_trap = R_Gamma_fft(t_grid, G_trap)
lag_exp, R_exp = R_Gamma_fft(t_grid, G_exp)
lag_sn, R_sn = R_Gamma_fft(t_grid, G_sn)

# Normalize R_Gamma
R_trap_norm = R_trap / R_trap[0]
R_exp_norm = R_exp / R_exp[0]
R_sn_norm = R_sn / R_sn[0]

# Normalize |FT{Gamma}|^2
Ghsq_trap_norm = Ghsq_trap / Ghsq_trap[0]
Ghsq_exp_norm = Ghsq_exp / Ghsq_exp[0]
Ghsq_sn_norm = Ghsq_sn / Ghsq_sn[0]

# ── Figure ──────────────────────────────────────────────────────

label_fs = 20
tick_fs = 18
legend_fs = 16
title_fs = 20
param_fs = 18
lw = 2.5

colors = {'trap': 'C0', 'exp': 'C2', 'sn': 'C3'}
labels = {'trap': 'Trapezoidal', 'exp': 'Exponential', 'sn': 'Skew-normal'}

fig, axes = plt.subplots(1, 3, figsize=(21, 6))

# ── Panel 1: Gamma(t) ──
ax = axes[0]
ax.plot(t_grid, G_trap, color=colors['trap'], lw=lw, label=labels['trap'])
ax.plot(t_grid, G_exp, color=colors['exp'], lw=lw, label=labels['exp'])
ax.plot(t_grid, G_sn, color=colors['sn'], lw=lw, label=labels['sn'])
ax.set_xlabel('Time [days]', fontsize=label_fs)
ax.set_ylabel(r'$\Gamma(t)$', fontsize=label_fs)
ax.set_title(r'Squared envelope $\Gamma(t)$', fontsize=title_fs)
# ax.legend(fontsize=legend_fs)
ax.set_xlim(-ell / 2 - 4 * tau_s, ell / 2 + 4 * tau_s)
ax.set_ylim(-0.05, 1.15)
ax.tick_params(labelsize=tick_fs)
ax.minorticks_on()

# ── Panel 2: |FT{Gamma}|^2 ──
ax = axes[1]
omega_max = 5.0  # rad/day
mask_trap = omega_trap <= omega_max
mask_exp = omega_exp <= omega_max
mask_sn = omega_sn <= omega_max
ax.semilogy(omega_trap[mask_trap], Ghsq_trap_norm[mask_trap],
            color=colors['trap'], lw=lw, label=labels['trap'])
ax.semilogy(omega_exp[mask_exp], Ghsq_exp_norm[mask_exp],
            color=colors['exp'], lw=lw, label=labels['exp'])
ax.semilogy(omega_sn[mask_sn], Ghsq_sn_norm[mask_sn],
            color=colors['sn'], lw=lw, label=labels['sn'])
ax.set_xlabel(r'$\omega$ [rad/day]', fontsize=label_fs)
ax.set_ylabel(r'$S_\Gamma(\omega)$ (normalized)', fontsize=label_fs)
ax.set_title(r'Power spectrum $S_\Gamma(\omega)$', fontsize=title_fs)
# ax.legend(fontsize=legend_fs)
ax.set_xlim(0, omega_max)
ax.tick_params(labelsize=tick_fs)
ax.minorticks_on()

# ── Panel 3: R_Gamma(tau) ──
ax = axes[2]
tau_max_plot = ell + 4 * tau_s
mask_trap = lag_trap <= tau_max_plot
mask_exp = lag_exp <= tau_max_plot
mask_sn = lag_sn <= tau_max_plot
ax.plot(lag_trap[mask_trap], R_trap_norm[mask_trap],
        color=colors['trap'], lw=lw, label=labels['trap'])
ax.plot(lag_exp[mask_exp], R_exp_norm[mask_exp],
        color=colors['exp'], lw=lw, label=labels['exp'])
ax.plot(lag_sn[mask_sn], R_sn_norm[mask_sn],
        color=colors['sn'], lw=lw, label=labels['sn'])
ax.axhline(0, color='gray', ls='--', lw=0.5)
ax.set_xlabel(r'$\tau$ [days]', fontsize=label_fs)
ax.set_ylabel(r'$R_\Gamma(\tau)\, /\, R_\Gamma(0)$', fontsize=label_fs)
ax.set_title(r'Envelope autocorrelation $R_\Gamma(\tau)$', fontsize=title_fs)
ax.legend(fontsize=legend_fs)
ax.set_xlim(0, tau_max_plot)
ax.tick_params(labelsize=tick_fs)
ax.minorticks_on()

# Parameter annotation
param_text = (rf'$\ell = {ell:.0f}$\,d, '
              rf'$\tau_s = {tau_s:.0f}$\,d, '
              rf'$\sigma_{{\rm sn}} = {sigma_sn:.0f}$\,d, '
              rf'$n_{{\rm sn}} = {n_sn:.0f}$')
fig.text(0.5, -0.02, param_text, ha='center', fontsize=param_fs,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.tight_layout()
fig.savefig(paths.figures / "fig_envelope_kernel_comparison.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_envelope_kernel_comparison.pdf")
