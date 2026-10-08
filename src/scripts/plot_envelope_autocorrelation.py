"""
Plot the piecewise components of Gamma_hat(omega) and R_Gamma(tau).

Left panel:  Fourier transform Gamma_hat(omega), showing the plateau integral I1,
             the parabolic ramp integral I2, and their sum.
Right panel: Autocorrelation R_Gamma(tau), showing the four piecewise polynomial
             intervals with different colors.

Output: figures/fig_envelope_autocorrelation.pdf
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

label_fs = 20
tick_fs = 18
legend_fs = 14
title_fs = 20
lw = 2.0

# ── Parameters ──
ell = 15.0       # spot plateau duration [days]
tau_s = 3.0      # rise/decay timescale [days]
L = ell / 2
S = L + tau_s


# ══════════════════════════════════════════════════════════════════════
# Gamma_hat(omega): Fourier transform of the squared envelope
# ══════════════════════════════════════════════════════════════════════

def Gamma_hat_full(w, ell, tau_s):
    """Gamma_hat(omega) of the normalized Gamma = alpha^2 / alpha_max^2, Eq. Gamma_hat."""
    result = np.zeros_like(w)
    nz = np.abs(w) > 1e-14
    ww = w[nz]
    result[nz] = (4 / (tau_s**2 * ww**3)) * (
        tau_s * ww * np.cos(ww * ell / 2)
        + np.sin(ww * ell / 2)
        - np.sin(ww * ell / 2 + ww * tau_s)
    )
    result[~nz] = ell + 2 * tau_s / 3
    return result


def I1_plateau(w, ell):
    """Plateau contribution: 2 I_1 = 2 sin(w*ell/2) / w."""
    result = np.zeros_like(w)
    nz = np.abs(w) > 1e-14
    result[nz] = 2 * np.sin(w[nz] * ell / 2) / w[nz]
    result[~nz] = ell
    return result


def I2_ramp(w, ell, tau_s):
    """Parabolic ramp contribution: 2 I_2 = Gamma_hat - 2 I_1."""
    return Gamma_hat_full(w, ell, tau_s) - I1_plateau(w, ell)


# ══════════════════════════════════════════════════════════════════════
# R_Gamma(tau): closed-form piecewise autocorrelation
# ══════════════════════════════════════════════════════════════════════

def R_Gamma_interval1(tau, ell, tau_s):
    """0 <= tau <= tau_s."""
    return (ell + 2*tau_s/5
                 - 4*tau**2 / (3*tau_s)
                 + 2*tau**3 / (3*tau_s**2)
                 - tau**5 / (15*tau_s**4))


def R_Gamma_interval2(tau, ell, tau_s):
    """tau_s <= tau <= ell."""
    return ell + 2*tau_s/3 - tau


def R_Gamma_interval3(tau, ell, tau_s):
    """ell <= tau <= ell + tau_s."""
    ts = tau_s
    t = tau
    return (t**5 / (30*ts**4)
                 - (ell + 2*ts) * t**4 / (6*ts**4)
                 + (ell**2 + 4*ell*ts + 2*ts**2) * t**3 / (3*ts**4)
                 - ell*(ell**2 + 6*ell*ts + 6*ts**2) * t**2 / (3*ts**4)
                 + (ell**4 + 8*ell**3*ts + 12*ell**2*ts**2 - 6*ts**4) * t / (6*ts**4)
                 + (-ell**5 - 10*ell**4*ts - 20*ell**3*ts**2
                    + 30*ell*ts**4 + 20*ts**5) / (30*ts**4))


def R_Gamma_interval4(tau, ell, tau_s):
    """ell + tau_s <= tau <= ell + 2*tau_s."""
    return (ell + 2*tau_s - tau)**5 / (30 * tau_s**4)


def R_Gamma_numerical(tau_arr, ell, tau_s, ns=20000):
    """Compute R_Gamma via direct numerical convolution (for verification)."""
    S = ell / 2 + tau_s
    s_grid = np.linspace(-S - 0.01, S + 0.01, ns)

    def Gamma(s):
        abs_s = np.abs(s)
        result = np.zeros_like(abs_s)
        plateau = abs_s <= ell / 2
        ramp = (abs_s > ell / 2) & (abs_s <= S)
        result[plateau] = 1.0
        result[ramp] = ((S - abs_s[ramp]) / tau_s)**2
        return result

    R = np.empty_like(tau_arr)
    G_s = Gamma(s_grid)
    for i, t in enumerate(tau_arr):
        G_st = Gamma(s_grid + t)
        R[i] = np.trapz(G_s * G_st, s_grid)
    return R


# ══════════════════════════════════════════════════════════════════════
# Figure
# ══════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ── Left panel: Gamma_hat(omega) ──
ax = axes[0]
omega = np.linspace(0.001, 4.0, 2000)

Gh_full = Gamma_hat_full(omega, ell, tau_s)
Gh_I1 = I1_plateau(omega, ell)
Gh_I2 = I2_ramp(omega, ell, tau_s)

ax.plot(omega, Gh_full, 'k-', lw=lw + 0.5, label=r'$\hat{\Gamma}(\omega) = 2(I_1 + I_2)$')
ax.plot(omega, Gh_I1, 'C0--', lw=lw, alpha=0.8,
        label=r'$2 I_1$ (plateau)')
ax.plot(omega, Gh_I2, 'C1--', lw=lw, alpha=0.8,
        label=r'$2 I_2$ (parabolic ramp)')
ax.axhline(0, color='gray', lw=0.5, alpha=0.4)

ax.set_xlabel(r'$\omega$ [rad/day]', fontsize=label_fs)
ax.set_ylabel(r'$\hat{\Gamma}(\omega)$', fontsize=label_fs)
ax.set_title(r'Fourier transform of $\Gamma(t) = \alpha^2(t)/\alpha_{\max}^2$', fontsize=title_fs)
ax.legend(fontsize=legend_fs, loc='upper right')
ax.tick_params(labelsize=tick_fs)
ax.set_xlim(0, 4.0)
ax.minorticks_on()

# ── Right panel: R_Gamma(tau) ──
ax = axes[1]

# Interval boundaries
b1, b2, b3, b4 = tau_s, ell, ell + tau_s, ell + 2 * tau_s

# Grids for each interval (with slight overlap for continuity)
eps = 0.001
t1 = np.linspace(0, b1, 200)
t2 = np.linspace(b1, b2, 200)
t3 = np.linspace(b2, b3, 200)
t4 = np.linspace(b3, b4, 200)

R1 = R_Gamma_interval1(t1, ell, tau_s)
R2 = R_Gamma_interval2(t2, ell, tau_s)
R3 = R_Gamma_interval3(t3, ell, tau_s)
R4 = R_Gamma_interval4(t4, ell, tau_s)

# Numerical verification
t_all = np.linspace(0, b4 + 2, 400)
R_num = R_Gamma_numerical(t_all, ell, tau_s)

ax.plot(t_all, R_num, 'k-', lw=1.0, alpha=0.3, label='Numerical (convolution)')
ax.plot(t1, R1, 'C0-', lw=lw + 0.5, label=rf'Interval 1: $0 \leq \tau \leq \tau_s$')
ax.plot(t2, R2, 'C1-', lw=lw + 0.5, label=rf'Interval 2: $\tau_s \leq \tau \leq \ell$ (linear)')
ax.plot(t3, R3, 'C2-', lw=lw + 0.5, label=rf'Interval 3: $\ell \leq \tau \leq \ell + \tau_s$')
ax.plot(t4, R4, 'C3-', lw=lw + 0.5, label=rf'Interval 4: $\ell + \tau_s \leq \tau \leq \ell + 2\tau_s$')

# Mark interval boundaries
for b, lab in [(b1, r'$\tau_s$'), (b2, r'$\ell$'),
               (b3, r'$\ell\!+\!\tau_s$'), (b4, r'$\ell\!+\!2\tau_s$')]:
    ax.axvline(b, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.text(b, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -0.2e-8,
            lab, fontsize=legend_fs - 1, ha='center', va='top', color='gray')

ax.axhline(0, color='gray', lw=0.5, alpha=0.4)

ax.set_xlabel(r'$\tau$ [days]', fontsize=label_fs)
ax.set_ylabel(r'$R_\Gamma(\tau)$', fontsize=label_fs)
ax.set_title(r'Autocorrelation of $\Gamma(t) = \alpha^2(t)/\alpha_{\max}^2$', fontsize=title_fs)
ax.legend(fontsize=legend_fs - 1, loc='upper right')
ax.tick_params(labelsize=tick_fs)
ax.set_xlim(0, b4 + 1)
ax.minorticks_on()

# Add parameter annotation
param_text = (rf'$\ell = {ell:.0f}$\,d, '
              rf'$\tau_s = {tau_s:.0f}$\,d')
ax.text(0.97, 0.55, param_text, transform=ax.transAxes,
        fontsize=legend_fs - 1, ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.tight_layout()
fig.savefig(paths.figures / "fig_envelope_autocorrelation.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_envelope_autocorrelation.pdf")
