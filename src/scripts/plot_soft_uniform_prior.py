"""
Plot the soft uniform prior from Eq. 73 of the paper.

Shows the prior density for a single parameter theta_i with bounds [a, b]
for different steepness values k, compared to a hard uniform prior.

Output: figures/fig_soft_uniform_prior.pdf
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
legend_fs = 16
lw = 2.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def log_soft_uniform(theta, a, b, k):
    """Log soft uniform prior for a single parameter (Eq. 73)."""
    return (np.log(sigmoid(k * (theta - a)) + 1e-300)
            + np.log(sigmoid(k * (b - theta)) + 1e-300)
            - np.log(b - a))


# Parameter bounds
a, b = 0.0, 1.0
theta = np.linspace(-0.3, 1.3, 2000)

# Steepness values to show
k_values = [5, 20, 100]
colors = ['C0', 'C1', 'C2']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left panel: log prior ──
ax = axes[0]
for k_val, color in zip(k_values, colors):
    log_p = log_soft_uniform(theta, a, b, k_val)
    ax.plot(theta, log_p, color=color, lw=lw, label=rf'$k={k_val}$')

# Hard uniform for reference
log_p_hard = np.where((theta >= a) & (theta <= b), 0.0, -30)
ax.plot(theta, log_p_hard, 'k--', lw=1.5, alpha=0.5, label='Hard uniform')

ax.set_xlabel(r'$\theta_i$', fontsize=label_fs)
ax.set_ylabel(r'$\ln\, p(\theta_i)$', fontsize=label_fs)
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-15, 2)
ax.axvline(a, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axvline(b, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.legend(fontsize=legend_fs)
ax.tick_params(labelsize=tick_fs)
ax.minorticks_on()
ax.set_title('Log prior density', fontsize=label_fs)

# ── Right panel: prior density (linear scale) ──
ax = axes[1]
for k_val, color in zip(k_values, colors):
    log_p = log_soft_uniform(theta, a, b, k_val)
    p = np.exp(log_p)
    ax.plot(theta, p, color=color, lw=lw, label=rf'$k={k_val}$')

# Hard uniform
p_hard = np.where((theta >= a) & (theta <= b), 1.0 / (b - a), 0.0)
ax.plot(theta, p_hard, 'k--', lw=1.5, alpha=0.5, label='Hard uniform')

ax.set_xlabel(r'$\theta_i$', fontsize=label_fs)
ax.set_ylabel(r'$p(\theta_i)$', fontsize=label_fs)
ax.set_xlim(-0.3, 1.3)
ax.set_ylim(-0.05, 1.5)
ax.axvline(a, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.axvline(b, color='gray', ls=':', lw=0.8, alpha=0.5)
ax.legend(fontsize=legend_fs)
ax.tick_params(labelsize=tick_fs)
ax.minorticks_on()
ax.set_title('Prior density', fontsize=label_fs)

fig.tight_layout()
fig.savefig(paths.figures / "fig_soft_uniform_prior.pdf",
            bbox_inches="tight", dpi=200)
print("Saved fig_soft_uniform_prior.pdf")
