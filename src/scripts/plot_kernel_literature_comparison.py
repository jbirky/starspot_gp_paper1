"""
Compare the starspot kernel to quasi-periodic (QP) and SHO kernels.

Top-left:    Kernel functions k(tau) for all three.
Top-right:   PSD (Fourier transform) showing harmonic structure.
Bottom-left: Kernel functions for a second parameter regime.
Bottom-right: PSD for the second regime.

Output: figures/fig_kernel_literature_comparison.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import paths

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

from spotgp.analytic_kernel import AnalyticKernel

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['cmss10', 'Computer Modern Sans Serif']
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

label_fs = 20
title_fs = 20
tick_fs = 18
legend_fs = 16
lw = 1.5

plot_colors = ["k", "r", "royalblue", "forestgreen"]
alphas = [1.0, 0.8, 0.8, 0.8]


# ── Kernel definitions ──────────────────────────────────────────
def kernel_qp(tau, A, P, lam, omega):
    """
    Quasi-periodic kernel:
        k(tau) = A^2 exp(-tau^2 / (2 lambda^2)) exp(-2 sin^2(pi tau / P) / omega^2)
    """
    return A**2 * np.exp(-tau**2 / (2 * lam**2)) * np.exp(
        -2 * np.sin(np.pi * tau / P)**2 / omega**2)


def kernel_sho(tau, A, P, lam):
    """
    Simple harmonic oscillator kernel:
        k(tau) = A^2 exp(-tau / lambda) cos(2 pi tau / P)
    """
    return A**2 * np.exp(-np.abs(tau) / lam) * np.cos(2 * np.pi * tau / P)


def kernel_gordon2020(tau, S0, omega0):
    """
    Gordon, Agol & Foreman-Mackey (2020) SHO kernel at Q = 1/sqrt(2) (Eq. 6):
        k(tau) = S0 * omega0 * exp(-omega0 * tau / sqrt(2))
                 * cos(omega0 * tau / sqrt(2) - pi/4)
    """
    x = omega0 * np.abs(tau) / np.sqrt(2)
    return S0 * omega0 * np.exp(-x) * np.cos(x - np.pi / 4)


def psd_from_acf(tau, k_vals, freq):
    """
    Compute PSD from ACF via numerical cosine transform:
        S(f) = 2 * integral_0^inf k(tau) cos(2 pi f tau) dtau
    """
    dtau = tau[1] - tau[0]
    psd = np.zeros_like(freq)
    for i, f in enumerate(freq):
        psd[i] = 2 * np.sum(k_vals * np.cos(2 * np.pi * f * tau)) * dtau
    return np.maximum(psd, 0)


# ── Two parameter regimes ──────────────────────────────────────
# Regime 1: Edge-on, moderate spot lifetime (shows harmonics clearly)
# Regime 2: Inclined, longer lifetime (more decay, different harmonic weights)
# Regime 3: Differential rotation (shows effect of kappa)

regimes = [
    dict(
        label="Edge-on, short-lived spots",
        hparam=dict(peq=5.0, kappa=0.0, inc=np.pi / 2, nspot=10,
                    lspot=10.0, tau_spot=3.0, alpha_max=0.1, fspot=0.0),
        P=10.0, lam_qp=12.0, omega_qp=0.7, A_qp=1.0,
        lam_sho=10.0, A_sho=1.0,
        omega0_gordon=2 * np.pi / 5.0, S0_gordon=1.0,
    ),
    dict(
        label=r"Inclined ($I=60^\circ$), long-lived spots",
        hparam=dict(peq=5.0, kappa=0.0, inc=np.radians(60), nspot=10,
                    lspot=25.0, tau_spot=5.0, alpha_max=0.1, fspot=0.0),
        P=10.0, lam_qp=25.0, omega_qp=0.7, A_qp=1.0,
        lam_sho=20.0, A_sho=1.0,
        omega0_gordon=2 * np.pi / 5.0, S0_gordon=1.0,
    ),
    dict(
        label=r"Differential rotation ($\kappa=0.5$)",
        hparam=dict(peq=5.0, kappa=1.0, inc=np.pi / 2, nspot=10,
                    lspot=15.0, tau_spot=3.0, alpha_max=0.1, fspot=0.0),
        P=10.0, lam_qp=12.0, omega_qp=0.7, A_qp=1.0,
        lam_sho=10.0, A_sho=1.0,
        omega0_gordon=2 * np.pi / 5.0, S0_gordon=1.0,
    ),
]


# ── Figure ──────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

tau_max_plot = 40.0
tau_max_fft = 200.0  # use longer range for PSD accuracy
tau = np.linspace(0, tau_max_fft, 20000)
freq = np.linspace(1e-3, 0.8, 2000)

for row, regime in enumerate(regimes):
    ax_k = axes[row, 0]
    ax_psd = axes[row, 1]

    P = regime["P"]
    hp = regime["hparam"]

    # ── Starspot kernel ──
    ak = AnalyticKernel(hp, n_harmonics=5)
    k_spot = ak.kernel(tau)
    # Normalize to ACF
    k_spot_acf = k_spot / k_spot[0] if k_spot[0] != 0 else k_spot

    # ── QP kernel ──
    k_qp = kernel_qp(tau, regime["A_qp"], P, regime["lam_qp"], regime["omega_qp"])
    k_qp_acf = k_qp / k_qp[0]

    # ── SHO kernel ──
    k_sho = kernel_sho(tau, regime["A_sho"], P, regime["lam_sho"])
    k_sho_acf = k_sho / k_sho[0]

    # ── Gordon et al. (2020) SHO kernel (Q = 1/sqrt(2)) ──
    k_gordon = kernel_gordon2020(tau, regime["S0_gordon"], regime["omega0_gordon"])
    k_gordon_acf = k_gordon / k_gordon[0] if k_gordon[0] != 0 else k_gordon

    # ── Plot kernel (ACF) — only show up to tau_max_plot ──
    idx_plot = tau <= tau_max_plot
    ax_k.plot(tau[idx_plot], k_spot_acf[idx_plot], plot_colors[0], lw=lw, alpha=alphas[0],
              label='Starspot (this work)')
    ax_k.plot(tau[idx_plot], k_qp_acf[idx_plot], plot_colors[1], lw=lw, alpha=alphas[1],
              label='Quasi-periodic')
    ax_k.plot(tau[idx_plot], k_sho_acf[idx_plot], plot_colors[2], lw=lw, alpha=alphas[2],
              label='SHO')
    # ax_k.plot(tau[idx_plot], k_gordon_acf[idx_plot], plot_colors[3], lw=lw, alpha=alphas[3],
    #           label='Gordon+2020')

    # Period markers
    for n in range(1, int(tau_max_plot / P) + 1):
        ax_k.axvline(n * P, color='gray', lw=0.5, alpha=0.3)
    ax_k.axhline(0, color='gray', lw=0.5, alpha=0.3)

    ax_k.set_ylabel('Normalized ACF', fontsize=label_fs)
    ax_k.set_title(regime["label"], fontsize=title_fs)
    ax_k.legend(fontsize=legend_fs, loc='upper right')
    ax_k.tick_params(labelsize=tick_fs)
    ax_k.set_xlim(0, tau_max_plot)
    ax_k.set_ylim(-0.4, 1.05)
    ax_k.minorticks_on()

    # ── PSD via cosine transform ──
    psd_spot = psd_from_acf(tau, k_spot_acf, freq)
    psd_qp = psd_from_acf(tau, k_qp_acf, freq)
    psd_sho = psd_from_acf(tau, k_sho_acf, freq)
    psd_gordon = psd_from_acf(tau, k_gordon_acf, freq)

    # Avoid log(0)
    psd_spot = np.maximum(psd_spot, 1e-15)
    psd_qp = np.maximum(psd_qp, 1e-15)
    psd_sho = np.maximum(psd_sho, 1e-15)
    psd_gordon = np.maximum(psd_gordon, 1e-15)

    ax_psd.semilogy(freq, psd_spot, plot_colors[0], lw=lw, alpha=alphas[0], label='Starspot (this work)')
    ax_psd.semilogy(freq, psd_qp, plot_colors[1], lw=lw, alpha=alphas[1], label='Quasi-periodic')
    ax_psd.semilogy(freq, psd_sho, plot_colors[2], lw=lw, alpha=alphas[2], label='SHO')
    # ax_psd.semilogy(freq, psd_gordon, plot_colors[3], lw=lw, alpha=alphas[3], label='Gordon+2020')

    # Clip y-axis to avoid numerical floor artifacts
    all_psd = np.concatenate([psd_spot, psd_qp, psd_sho, psd_gordon])
    psd_max = all_psd.max()
    ax_psd.set_ylim(psd_max * 1e-5, psd_max * 5)

    # Harmonic markers
    for n in range(1, 4):
        ax_psd.axvline(n / P, color='gray', lw=0.5, alpha=0.4)
        ax_psd.text(n / P + 0.005, psd_max * 2,
                    rf'$n={n}$', fontsize=legend_fs, color='gray', va='top')

    ax_psd.set_ylabel('PSD', fontsize=label_fs)
    ax_psd.set_title(regime["label"], fontsize=title_fs)
    # ax_psd.legend(fontsize=legend_fs, loc='lower right')
    ax_psd.tick_params(labelsize=tick_fs)
    ax_psd.set_xlim(0, 0.8)
    ax_psd.minorticks_on()

# Bottom row x-labels
axes[-1, 0].set_xlabel(r'Time lag $\tau$ [days]', fontsize=label_fs)
axes[-1, 1].set_xlabel('Frequency [cycles/day]', fontsize=label_fs)

fig.tight_layout()
fig.savefig(paths.figures / "fig_kernel_literature_comparison.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_kernel_literature_comparison.pdf")
