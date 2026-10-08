# Custom rules for figures whose script writes more than one figure environment's
# worth of output, or writes it in more than one invocation. showyourwork gives
# these rules priority over the one-rule-per-\script figure rules it generates.

from glob import glob

# One script makes all four CRB corner plots (four separate figure environments).
rule crb_corner_plots:
    input:
        "src/scripts/plot_crb_param_sweep.py",
        "src/data/crb_sobol_sweep_365_4096.csv",
    output:
        "src/tex/figures/crb_sweep_corner_contour_kappa.png",
        "src/tex/figures/crb_sweep_corner_contour_inc.png",
        "src/tex/figures/crb_sweep_corner_contour_peq.png",
        "src/tex/figures/crb_sweep_corner_contour_sigma_k.png",
    shell:
        "python {input[0]}"

# The script takes the sampling cadence [d] as its argument; the figure shows two.
rule kernel_mse_vs_ndata:
    input:
        "src/scripts/plot_kernel_mse_vs_ndata.py",
    output:
        "src/tex/figures/fig_kernel_mse_vs_ndata_0p5d.pdf",
        "src/tex/figures/fig_kernel_mse_vs_ndata_5p0d.pdf",
    shell:
        "python {input} 0.5 && python {input} 5.0"

# The script writes the accuracy table that ms.tex \inputs alongside its figure.
rule kernel_accuracy_benchmark:
    input:
        "src/scripts/plot_analytic_vs_numerical_kernel_single_panel.py",
    output:
        "src/tex/figures/fig_kernel_comparison_grid.pdf",
        "src/tex/output/kernel_accuracy_benchmark.tex",
    shell:
        "python {input}"

# The posterior summary table of the dynesty fits to simulated lightcurves. The fits
# are run by hand (compute_mcmc_simulated_dynesty.py) and their results kept in src/data.
rule simulated_posterior_table:
    input:
        "src/scripts/table_simulated_posteriors.py",
        expand("src/data/results_simulated_dynesty_mean/tsim_{tsim}_tsamp_1.0_inc_{inc}_peq_5.0/dynesty_results.npz",
               tsim=["365.0", "1460.0"], inc=["30.0", "60.0", "90.0"]),
    output:
        "src/tex/output/simulated_posterior_table.tex",
    shell:
        "python {input[0]} {output}"

# Each derivation notebook asserts its results against the equations in ms.tex, so
# this rule fails if a derivation no longer matches the paper. The output is a link
# to the notebook that ms.tex includes with \variable. A notebook that takes its
# equations from the registry (src/derivations/equations.py) also re-runs when the
# registry changes.
def derivation_inputs(wildcards):
    notebook = f"src/derivations/{wildcards.name}.ipynb"
    inputs = [notebook, "src/scripts/run_derivation.py"]
    with open(notebook) as f:
        if "import equations" in f.read():
            inputs.append("src/derivations/equations.py")
    return inputs

rule derivation:
    input:
        derivation_inputs,
    output:
        "src/tex/output/derivations/{name}.tex",
    shell:
        "python {input[1]} {input[0]} {output}"

# Checks every equation in the registry against ms.tex, its derivation notebook, and
# spotgp, and fails if any of them disagree. See src/scripts/check_equations.py.
rule equation_checks:
    input:
        "src/scripts/check_equations.py",
        "src/derivations/equations.py",
        "src/derivations/reviewed_equations.txt",
        "src/tex/ms.tex",
        glob("src/derivations/*.ipynb"),
    output:
        "src/tex/output/equation_checks.tex",
        "src/tex/output/equation_icons.tex",
    shell:
        "python {input[0]} {output}"
