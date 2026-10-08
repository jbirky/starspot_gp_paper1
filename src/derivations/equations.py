r"""
Registry of manuscript equations, keyed by their \label in ms.tex.

Each equation is transcribed from ms.tex into SymPy once, here. Everything else is
checked against that expression (the spec):

    derivation  the notebook named in the entry imports the spec with get(label) and
                asserts that its derived result equals it. The `derivation` rule in
                the Snakefile re-runs the notebook whenever this file changes.
    manuscript  check_equations.py asserts that the equation's LaTeX in ms.tex is the
                text the spec was last reviewed against (reviewed_equations.txt).
    spotgp      check_equations.py asserts that the function in `impl` matches the
                spec in value and gradient over `domain`, evaluating the spec with
                50-digit arithmetic.

Where an equation has several members (c_n = integral = closed form), the spec is the
last one; the earlier members are steps of the derivation. Symbols that the text
around an equation defines (the Delta t_i of eqn:trapezoid, b = ell/2 + tau of
eq:J_ibp) are substituted into its spec.

To add an equation: transcribe it below, add an Equation to EQUATIONS, take it from
the registry in its derivation notebook, then record the LaTeX you checked it
against with

    python src/scripts/check_equations.py --accept <label>
"""
from dataclasses import dataclass, field

import sympy as sp


@dataclass(frozen=True)
class Equation:
    label: str                # \label in ms.tex
    expr: sp.Expr             # the equation as typeset, in SymPy
    derivation: str           # notebook in src/derivations that derives it
    impl: str | None = None   # spotgp function that implements it, "module:function" or "module:Class.method"
    args: tuple = ()          # symbols passed to impl, in order
    # Where impl is checked: symbol -> (lo, hi). Bounds may depend on earlier symbols.
    # The last symbol is the variable; the others are parameters, drawn at random
    # and from extra_params, and for each draw the variable is swept across its
    # range and both sides of every piecewise breakpoint. Integer symbols are drawn
    # as integers in [lo, hi] and are not differentiated.
    domain: dict = field(default_factory=dict)
    extra_params: tuple = ()  # parameter draws to always include, {symbol: value}
    rtol: float = 1e-10       # tolerance, relative to the largest |value| or |gradient| in a sweep
    # Symbols of expr that another registry equation defines, {symbol: that spec},
    # substituted before the spotgp check
    given: dict = field(default_factory=dict)


# Shared with the notebooks (a Symbol with the same name and assumptions compares equal)
tau = sp.Symbol("tau", nonnegative=True)   # lag
ell = sp.Symbol("ell", positive=True)      # plateau duration, ell_spot
tau_s = sp.Symbol("tau_s", positive=True)  # ramp duration, tau_spot
t = sp.Symbol("t", real=True)              # time
t1, t2 = sp.symbols("t_1 t_2", real=True)  # start and end of the plateau
t_ref = sp.Symbol("t_ref", real=True)      # centre of the trapezoid's plateau
omega = sp.Symbol("omega", positive=True)  # angular frequency
tau_em = sp.Symbol("tau_e", positive=True)     # emergence timescale
tau_dec = sp.Symbol("tau_d", positive=True)    # decay timescale
tau_plat = sp.Symbol("tau_plat", positive=True)  # plateau duration of the exponential envelope, tau_plat
alpha_max = sp.Symbol("alpha_max", positive=True)

inc = sp.Symbol("I_inc", positive=True)          # stellar inclination I (sympy's I is the imaginary unit)
phi = sp.Symbol("Phi", real=True)                # spot latitude
b0 = sp.Symbol("b_0", real=True)                 # cos I sin Phi
b1 = sp.Symbol("b_1", positive=True)             # sin I cos Phi
n = sp.Symbol("n", integer=True)                 # harmonic order
theta = sp.Symbol("theta", real=True)            # rotation phase omega_0 t + Lambda_0
theta_vis = sp.Symbol("theta_v", positive=True)  # half-width of the visibility window, theta_vis
omega0 = sp.Symbol("omega_0", positive=True)     # rotation frequency
sigma_k = sp.Symbol("sigma_k", positive=True)    # kernel amplitude
R_Gamma = sp.Function("R_Gamma")                 # envelope autocorrelation, of the lag
C0, C1, C2 = sp.symbols("C_0 C_1 C_2", nonnegative=True)  # latitude averages <|c_n|^2>_Phi


# ── Section sec:spot_profile: trapezoidal and exponential envelopes ───────────────

# alpha_trap / alpha_max, with the Delta t_i and the normalizations I = tau_em,
# E = tau_dec that the text after the equation defines
_dt = (t - (t_ref - ell / 2 - tau_em), t - (t_ref - ell / 2),
       t - (t_ref + ell / 2), t - (t_ref + ell / 2 + tau_dec))
ALPHA_TRAP = ((_dt[0] * sp.Heaviside(_dt[0]) - _dt[1] * sp.Heaviside(_dt[1])) / tau_em
              - (_dt[2] * sp.Heaviside(_dt[2]) - _dt[3] * sp.Heaviside(_dt[3])) / tau_dec)

ALPHA_EXP = alpha_max * sp.Piecewise(
    (sp.exp((t - t1) / tau_em), t < t1),
    (1, (t >= t1) & (t <= t2)),
    (sp.exp(-(t - t2) / tau_dec), t > t2))


# ── Section subsec:ft_cosine: Fourier coefficients of the visibility function ─────

CN_EDGEON = sp.cos(phi) * sp.Piecewise(
    (1 / sp.pi, sp.Eq(n, 0)),
    (sp.Rational(1, 4), sp.Eq(sp.Abs(n), 1)),
    ((-1)**(sp.Abs(n) / 2 + 1) / (sp.pi * (n**2 - 1)), sp.Eq(n % 2, 0)),
    (0, True))  # n = +-3, +-5, ...

# The closed-form member; the integral member is the right-hand side of eq:cn_integral
CN_GENERAL = (b0 * sp.sin(n * theta_vis) / n
              + b1 / 2 * (sp.sin((n - 1) * theta_vis) / (n - 1)
                          + sp.sin((n + 1) * theta_vis) / (n + 1))) / sp.pi


# ── Section subsec:ft_envelope: Fourier transforms of the squared envelopes ───────

# Normalized Gamma = alpha^2 / alpha_max^2 of the symmetric trapezoid
GAMMA_HAT = 4 / (tau_s**2 * omega**3) * (
    tau_s * omega * sp.cos(omega * ell / 2)
    + sp.sin(omega * ell / 2)
    - sp.sin(omega * ell / 2 + omega * tau_s))

# Normalized Gamma = alpha_exp^2 / alpha_max^2, with t_1 = 0 and t_2 = tau_plat
GAMMA_EXP = sp.Piecewise(
    (sp.exp(2 * t / tau_em), t < 0),
    (1, (t >= 0) & (t <= tau_plat)),
    (sp.exp(-2 * (t - tau_plat) / tau_dec), t > tau_plat))

# Gamma_hat = Gamma_hat_R - i Gamma_hat_I, term by term as typeset, grouped by envelope piece
GAMMA_HAT_EXP_R_TERMS = {
    "rise": 2 * tau_em / (4 + omega**2 * tau_em**2),
    "plateau": sp.sin(omega * tau_plat) / omega,
    "decay": (2 * tau_dec * sp.cos(omega * tau_plat) / (4 + omega**2 * tau_dec**2)
              - omega * tau_dec**2 * sp.sin(omega * tau_plat) / (4 + omega**2 * tau_dec**2)),
}
GAMMA_HAT_EXP_I_TERMS = {
    "rise": -omega * tau_em**2 / (4 + omega**2 * tau_em**2),
    "plateau": (1 - sp.cos(omega * tau_plat)) / omega,
    "decay": (omega * tau_dec**2 * sp.cos(omega * tau_plat) / (4 + omega**2 * tau_dec**2)
              + 2 * tau_dec * sp.sin(omega * tau_plat) / (4 + omega**2 * tau_dec**2)),
}


# ── Sections subsec:kernel_full and subsec:kernel_anal_case: the kernel ───────────

THETA_VIS_INC = sp.acos(-sp.cot(inc) * sp.tan(phi))

CN_INC = (sp.cos(inc) * sp.sin(phi) / n * sp.sin(n * theta_vis)
          + sp.sin(inc) * sp.cos(phi) / 2 * (sp.sin((n - 1) * theta_vis) / (n - 1)
                                             + sp.sin((n + 1) * theta_vis) / (n + 1))) / sp.pi

KERNEL_N2 = sigma_k**2 * R_Gamma(tau) * (
    C0 + 2 * C1 * sp.cos(omega0 * tau) + 2 * C2 * sp.cos(2 * omega0 * tau))

# For I = pi/2 and kappa = 0, the conditions typeset after the bracket
KERNEL_EDGEON_CLOSED = sigma_k**2 * R_Gamma(tau) * (
    1 / (2 * sp.pi**2) + sp.cos(omega0 * tau) / 16 + sp.cos(2 * omega0 * tau) / (9 * sp.pi**2))


# ── Appendix app:fourier_visibility: Fourier coefficients of the truncated cosine ─

THETA_VIS = sp.acos(-b0 / b1)

CN_INTEGRAL = sp.Integral((b0 + b1 * sp.cos(theta)) * sp.exp(-sp.I * n * theta),
                          (theta, -theta_vis, theta_vis)) / (2 * sp.pi)

CN_SPLIT = (b0 * sp.Integral(sp.exp(-sp.I * n * theta), (theta, -theta_vis, theta_vis))
            + b1 * sp.Integral(sp.cos(theta) * sp.exp(-sp.I * n * theta),
                               (theta, -theta_vis, theta_vis))) / (2 * sp.pi)

CN_INT_CONST = 2 * sp.sin(n * theta_vis) / n  # for n != 0

CN_INT_COS = (sp.sin((1 - n) * theta_vis) / (1 - n)
              + sp.sin((1 + n) * theta_vis) / (1 + n))  # for n != +-1

CN_RESULT = (b0 * sp.sin(n * theta_vis) / n
             + b1 / 2 * (sp.sin((n - 1) * theta_vis) / (n - 1)
                         + sp.sin((n + 1) * theta_vis) / (n + 1))) / sp.pi  # for n != 0, +-1


# ── Appendix app:fourier_trapezoid: Fourier transform of the squared trapezoid ────
# (tau there is tau_spot, not the lag)

ALPHA_PIECEWISE = alpha_max * sp.Piecewise(
    (0, t < -ell / 2 - tau_s),
    ((t + ell / 2 + tau_s) / tau_s, (-ell / 2 - tau_s <= t) & (t < -ell / 2)),
    (1, (-ell / 2 <= t) & (t <= ell / 2)),
    ((-t + ell / 2 + tau_s) / tau_s, (ell / 2 < t) & (t <= ell / 2 + tau_s)),
    (0, t > ell / 2 + tau_s))

GAMMA_PIECEWISE = sp.Piecewise(
    (0, sp.Abs(t) > ell / 2 + tau_s),
    (((sp.Abs(t) - ell / 2 - tau_s) / (-tau_s))**2,
     (ell / 2 < sp.Abs(t)) & (sp.Abs(t) <= ell / 2 + tau_s)),
    (1, sp.Abs(t) <= ell / 2))

I1_PLATEAU = sp.sin(omega * ell / 2) / omega

# with b = ell/2 + tau
J_IBP = (tau_s * sp.cos(omega * ell / 2) / omega
         - (sp.sin(omega * (ell / 2 + tau_s)) - sp.sin(omega * ell / 2)) / omega**2)

I2_RAMP = (-sp.sin(omega * ell / 2) / omega
           + 2 * sp.cos(omega * ell / 2) / (tau_s * omega**2)
           - 2 * (sp.sin(omega * ell / 2 + omega * tau_s) - sp.sin(omega * ell / 2)) / (tau_s**2 * omega**3))

I1_PLUS_I2 = (2 * sp.cos(omega * ell / 2) / (tau_s * omega**2)
              + 2 * sp.sin(omega * ell / 2) / (tau_s**2 * omega**3)
              - 2 * sp.sin(omega * ell / 2 + omega * tau_s) / (tau_s**2 * omega**3))

GAMMA_HAT_DIRECT = 4 / (tau_s**2 * omega**3) * (
    tau_s * omega * sp.cos(omega * ell / 2)
    + sp.sin(omega * ell / 2)
    - sp.sin(omega * ell / 2 + omega * tau_s))


# ── Appendix app:acf_trapezoid: autocorrelation of the squared envelope ──────────

P3 = (tau**5 / (30 * tau_s**4)
      - (ell + 2 * tau_s) * tau**4 / (6 * tau_s**4)
      + (ell**2 + 4 * ell * tau_s + 2 * tau_s**2) * tau**3 / (3 * tau_s**4)
      - ell * (ell**2 + 6 * ell * tau_s + 6 * tau_s**2) * tau**2 / (3 * tau_s**4)
      + (ell**4 + 8 * ell**3 * tau_s + 12 * ell**2 * tau_s**2 - 6 * tau_s**4) * tau / (6 * tau_s**4)
      + (-ell**5 - 10 * ell**4 * tau_s - 20 * ell**3 * tau_s**2 + 30 * ell * tau_s**4 + 20 * tau_s**5)
      / (30 * tau_s**4))

# Normalized envelope Gamma = alpha^2 / alpha_max^2, so no alpha_max^4 factor
R_GAMMA_CLOSED = sp.Piecewise(
    (ell + 2 * tau_s / 5 - 4 * tau**2 / (3 * tau_s) + 2 * tau**3 / (3 * tau_s**2)
     - tau**5 / (15 * tau_s**4), tau <= tau_s),
    (ell + 2 * tau_s / 3 - tau, tau <= ell),
    (P3, tau <= ell + tau_s),
    ((ell + 2 * tau_s - tau)**5 / (30 * tau_s**4), tau <= ell + 2 * tau_s),
    (0, True))


# In the order of ms.tex
EQUATIONS = (
    Equation("eqn:trapezoid", ALPHA_TRAP, derivation="derive_fourier_trapezoid.ipynb"),
    # spotgp has no implementation of this envelope: its ExponentialAsymmetricEnvelope
    # has no plateau and decays as exp(-|t| / tau) rather than exp(-2 |t| / tau)
    Equation("eq:alpha_exp", ALPHA_EXP, derivation="derive_kernel_exponential.ipynb"),
    Equation("eq:cn_edgeon", CN_EDGEON, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:cn_general", CN_GENERAL, derivation="derive_fourier_cosine.ipynb"),
    Equation(
        "eq:Gamma_hat", GAMMA_HAT, derivation="derive_fourier_gamma.ipynb",
        impl="spotgp.envelope:_Gamma_hat", args=(omega, ell, tau_s),
        # The bracket is O(omega^3) and spotgp evaluates it as typeset, so at small
        # omega tau_s cancellation costs it digits: at omega tau_s = 1e-2 the gradient is
        # good to ~1e-8, and at 1e-3 the value to ~1e-10. The sweep starts at
        # omega tau_s = 0.1. spotgp's PSD evaluates it at omega - n omega_0, which does
        # reach that range next to each harmonic peak
        domain={ell: (10, 30), tau_s: (0.01, 10), omega: (sp.Rational(1, 10) / tau_s, 50)}),
    Equation("eq:Gamma_exp", GAMMA_EXP, derivation="derive_kernel_exponential.ipynb"),
    Equation("eq:Gamma_hat_exp_R", sum(GAMMA_HAT_EXP_R_TERMS.values()),
             derivation="derive_kernel_exponential.ipynb"),
    Equation("eq:Gamma_hat_exp_I", sum(GAMMA_HAT_EXP_I_TERMS.values()),
             derivation="derive_kernel_exponential.ipynb"),
    Equation("eq:theta_vis_inc", THETA_VIS_INC, derivation="derive_fourier_cosine.ipynb"),
    Equation(
        "eq:cn_inc", CN_INC, derivation="derive_fourier_cosine.ipynb",
        impl="spotgp.visibility:_cn_general_jax", args=(n, inc, phi),
        given={theta_vis: THETA_VIS_INC},
        # The equation holds for n != 0, +-1 (spotgp's n = 0, 1 are the limits printed
        # after eq:cn_general) and in the band |Phi| < I. d theta_vis / d Phi is infinite
        # at the edges of the band, where spotgp clips the arccos argument, so the
        # sweep stops just inside them. Exactly edge-on, d c_n / d I vanishes and the
        # gradient error would be measured against rounding, so I stays below pi/2
        domain={n: (2, 6), inc: (0.05, sp.pi / 2), phi: (-0.9999 * inc, 0.9999 * inc)}),
    Equation("eq:kernel_n2", KERNEL_N2, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:kernel_edgeon_closed", KERNEL_EDGEON_CLOSED, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:theta_vis", THETA_VIS, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:cn_integral", CN_INTEGRAL, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:cn_split", CN_SPLIT, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:cn_int_const", CN_INT_CONST, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:cn_int_cos", CN_INT_COS, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:cn_result", CN_RESULT, derivation="derive_fourier_cosine.ipynb"),
    Equation("eq:alpha_piecewise", ALPHA_PIECEWISE, derivation="derive_fourier_trapezoid.ipynb"),
    Equation("eq:Gamma_piecewise", GAMMA_PIECEWISE, derivation="derive_fourier_gamma.ipynb"),
    Equation("eq:I1_plateau", I1_PLATEAU, derivation="derive_fourier_gamma.ipynb"),
    Equation("eq:J_ibp", J_IBP, derivation="derive_fourier_gamma.ipynb"),
    Equation("eq:I2_ramp", I2_RAMP, derivation="derive_fourier_gamma.ipynb"),
    Equation("eq:I1_plus_I2", I1_PLUS_I2, derivation="derive_fourier_gamma.ipynb"),
    Equation("eq:Gamma_hat_direct", GAMMA_HAT_DIRECT, derivation="derive_fourier_gamma.ipynb"),
    Equation(
        "eq:R_Gamma_closed", R_GAMMA_CLOSED, derivation="derive_R_Gamma.ipynb",
        impl="spotgp.envelope:_R_Gamma_symmetric", args=(tau, ell, tau_s),
        # the priors of the simulated-data fits (compute_mcmc_simulated_dynesty.py);
        # the lags run past the end of the support
        domain={ell: (10, 30), tau_s: (0.01, 10), tau: (0, 1.2 * (ell + 2 * tau_s))},
        extra_params=(
            {ell: 10, tau_s: 10},     # ell = tau_s: edge of the closed form's validity
            {ell: 205, tau_s: 0.33},  # long plateau, where P3 expanded in tau loses
            {ell: 1000, tau_s: 0.1},  # all precision (spotgp evaluates it in tau - ell)
        )),
    Equation("eq:P3", P3, derivation="derive_R_Gamma.ipynb"),
)


def get(label):
    """The registry entry for an ms.tex \\label."""
    for eq in EQUATIONS:
        if eq.label == label:
            return eq
    raise KeyError(f"{label} is not in the equation registry (src/derivations/equations.py)")
