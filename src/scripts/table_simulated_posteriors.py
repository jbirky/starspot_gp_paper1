"""
Posterior summary table for the dynesty fits to simulated lightcurves (Table
tab:simulated_posterior): the posterior mean and standard deviation of each
parameter, for each true inclination and observing baseline, next to the injected
value and the prior bounds.

Input:
    data/results_simulated_dynesty_mean/tsim_{365.0,1460.0}_tsamp_1.0_inc_{30.0,60.0,90.0}_peq_5.0/dynesty_results.npz
    (from compute_mcmc_simulated_dynesty.py, which stores equally weighted posterior samples)

Output:
    tex/output/simulated_posterior_table.tex

Usage:
    python table_simulated_posteriors.py [output.tex]
"""
import sys
from pathlib import Path

import numpy as np

import paths

TSIM = (365.0, 1460.0)       # observing baselines [d], one column pair each
INC_DEG = (30.0, 60.0, 90.0)  # true inclinations, one block of rows each
TSAMP, PEQ = 1.0, 5.0
RESULTS = paths.data / "results_simulated_dynesty_mean"

# parameter key -> (row label, factor from the sampled value to the tabulated one)
PARAMS = {
    "peq":      (r"$P_{\rm eq}$ [d]", 1.0),
    "kappa":    (r"$\kappa$", 1.0),
    "inc":      (r"$I$ [deg]", 180.0 / np.pi),
    "lspot":    (r"$\ell_{\rm spot}$ [d]", 1.0),
    "tau_spot": (r"$\tau_{\rm spot}$ [d]", 1.0),
    "sigma_k":  (r"$\sigma_k$", 1.0),
}


def prior_bounds(result):
    """Uniform prior bounds of the fit. Runs made before the bounds were stored in the
    results file get the ones compute_mcmc_simulated_dynesty.py sets."""
    if "bounds" in result.files:
        return result["bounds"].item()
    true = result["true_params"].item()
    return {
        "peq":      (0.9 * true["peq"], 1.1 * true["peq"]),
        "kappa":    (-1.0, 1.0),
        "inc":      (true["inc"] - np.pi / 6, true["inc"] + np.pi / 6),
        "lspot":    (10.0, 30.0),
        "tau_spot": (0.0, 10.0),
        "sigma_k":  (1e-3, 1e-2),
    }


def number(x):
    """A true value or prior bound: shortest form, with a typeset minus sign."""
    x = float(np.round(x, 10)) + 0.0  # drop rounding residue and negative zero
    return f"{x:g}".replace("-", "$-$")


def mean_std(x):
    """Posterior mean and standard deviation, both rounded to two significant figures
    of the standard deviation."""
    mean, std = np.mean(x), np.std(x, ddof=1)
    decimals = max(0, 1 - int(np.floor(np.log10(std))))
    return f"{mean:.{decimals}f}", f"{std:.{decimals}f}"


def load(tsim, inc_deg):
    path = RESULTS / f"tsim_{tsim}_tsamp_{TSAMP}_inc_{inc_deg}_peq_{PEQ}" / "dynesty_results.npz"
    result = np.load(path, allow_pickle=True)
    keys = list(result["param_keys"])
    assert keys == list(PARAMS), f"{path}: parameters {keys}, expected {list(PARAMS)}"
    samples = result["samples"]
    assert np.all(np.isfinite(samples)), f"{path}: non-finite samples"
    bounds = prior_bounds(result)
    for k, key in enumerate(keys):
        lo, hi = bounds[key]
        assert lo <= samples[:, k].min() and samples[:, k].max() <= hi, (
            f"{path}: {key} samples fall outside the prior bounds {bounds[key]}")
    return result["true_params"].item(), bounds, samples


def table():
    rows = []
    for b, inc_deg in enumerate(INC_DEG):
        fits = [load(tsim, inc_deg) for tsim in TSIM]
        true, bounds = fits[0][0], fits[0][1]
        for other_true, other_bounds, _ in fits[1:]:  # both baselines share truths and priors
            assert other_true == true and other_bounds == bounds
        rows.append(rf"    \textbf{{{inc_deg:g}$^\circ$}}")
        for k, (key, (label, scale)) in enumerate(PARAMS.items()):
            lo, hi = bounds[key]
            cells = [label, number(true[key] * scale), f"({number(lo * scale)}, {number(hi * scale)})"]
            for _, _, samples in fits:
                cells += mean_std(samples[:, k] * scale)
            rows.append("     & " + " & ".join(cells) + r" \\")
        if b < len(INC_DEG) - 1:
            rows.append(r"    \hline")
    rows[-1] = rows[-1].removesuffix(r" \\")

    return "\n".join([
        "% Written by src/scripts/table_simulated_posteriors.py from the dynesty results in",
        "% src/data/results_simulated_dynesty_mean; don't edit by hand.",
        r"\begin{deluxetable*}{llcc|cc|cc}",
        r"\tablecaption{Posterior summary statistics from nested sampling (\code{dynesty}) on simulated "
        r"lightcurves. For each combination of observation duration ($T_{\rm sim}$) and true inclination "
        r"($I_{\rm true}$), we report the posterior mean and standard deviation alongside the true (injected) "
        r"parameter value and the bounds of its uniform prior. The inclination is converted from radians to "
        rf"degrees for readability. All simulations use a sampling cadence of $\Delta t = {TSAMP:g}$\,d."
        r"\label{tab:simulated_posterior}}",
        r"\tablehead{",
        r"    \colhead{$I_{\rm true}$} & \colhead{Parameter} & \colhead{True} & \colhead{Prior} & "
        + " & ".join(rf"\multicolumn{{2}}{{c}}{{$T_{{\rm sim}} = {t:g}$\,d}}" for t in TSIM) + r" \\",
        r"    \colhead{} & \colhead{} & \colhead{Value} & \colhead{(min, max)} & "
        + " & ".join([r"\colhead{Mean} & \colhead{Std}"] * len(TSIM)),
        "}",
        r"\startdata",
        *rows,
        r"\enddata",
        r"\end{deluxetable*}",
        ""])


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else paths.output / "simulated_posterior_table.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(table())
    print(f"wrote {out}")
