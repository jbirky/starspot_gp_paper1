"""
Check every equation in the registry (src/derivations/equations.py) against the
manuscript, its derivation notebook, and spotgp.

    manuscript  The equation's LaTeX in ms.tex must be the text recorded in
                src/derivations/reviewed_equations.txt when its spec was last checked
                against it. After editing an equation in ms.tex, update its spec in
                equations.py to match, then record the new text with --accept.
    derivation  The notebook named in the entry must take its spec from the registry
                with get(label). The notebook itself asserts that its derivation
                equals the spec, when the `derivation` rule runs it.
    spotgp      The function in `impl` must match the spec in value and in gradient,
                within rtol of the largest |value| (|gradient|) in each sweep. The
                parameters are drawn from the entry's domain and extra_params; for
                each draw the variable is swept across its range and both sides of
                every piecewise breakpoint, and every branch must be reached. Integer
                parameters (a harmonic order) are drawn as integers, and the gradient
                is taken with respect to the other arguments. The spec is evaluated
                with mpmath at 50 digits, so the check measures spotgp and not the
                rounding error of the spec as typeset.

Every check runs and all failures are reported together. Only if all pass are the
optional outputs written: a LaTeX table of the results, and a file of margin icons
that link each registry equation in the PDF to its derivation notebook.

Usage:
    python check_equations.py [src/tex/output/equation_checks.tex [src/tex/output/equation_icons.tex]]
    python check_equations.py --accept eq:R_Gamma_closed [eq:P3 ...]
"""
import argparse
import difflib
import functools
import importlib
import json
import re
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import sympy as sp

import paths

sys.path.insert(0, str(paths.src / "derivations"))
import equations  # noqa: E402

MS = paths.tex / "ms.tex"
REVIEWED = paths.src / "derivations" / "reviewed_equations.txt"
DISPLAY_ENVS = "equation|align|gather|multline|eqnarray"
N_DRAWS = 40   # random parameter draws per equation
N_SWEEP = 30   # random values of the variable per draw
EPS = 1e-7     # relative offset either side of a breakpoint
mpmath.mp.dps = 50


# ── manuscript ────────────────────────────────────────────────────────────────

def equation_sources(tex):
    """{label: LaTeX} for every labelled display-math environment in tex."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    sources = {}
    for m in re.finditer(rf"\\begin\{{({DISPLAY_ENVS})(\*?)\}}(.*?)\\end\{{\1\2\}}", tex, re.S):
        labels = re.findall(r"\\label\{([^}]*)\}", m.group(3))
        body = re.sub(r"\\label\{[^}]*\}", "", m.group(3))
        body = textwrap.dedent("\n".join(l.rstrip() for l in body.splitlines() if l.strip()))
        for label in labels:
            # an environment holding several numbered equations is not split into rows
            sources[label] = body if len(labels) == 1 else None
    return sources


def same_latex(a, b):
    return a.split() == b.split()


def read_reviewed():
    reviewed, label = {}, None
    if REVIEWED.exists():
        for line in REVIEWED.read_text().splitlines():
            if line.startswith("%% "):
                label = line[3:].strip()
                reviewed[label] = []
            elif label is not None:
                reviewed[label].append(line)
    return {k: "\n".join(v) for k, v in reviewed.items()}


def accept(labels, sources):
    reviewed = read_reviewed()
    for label in labels:
        equations.get(label)  # only registry equations
        if sources.get(label) is None:
            sys.exit(f"{label}: no single labelled equation in ms.tex to accept")
        reviewed[label] = sources[label]
        print(f"{label}: recorded as reviewed")
    order = [eq.label for eq in equations.EQUATIONS]
    labels = sorted(reviewed, key=lambda k: order.index(k) if k in order else len(order))
    REVIEWED.write_text(
        "% The LaTeX of each registry equation (src/derivations/equations.py) as it read in\n"
        "% ms.tex when its spec was last checked against it. Written by\n"
        "%     python src/scripts/check_equations.py --accept <label>\n"
        "% Don't edit by hand.\n"
        + "".join(f"%% {k}\n{reviewed[k]}\n" for k in labels))


def check_manuscript(eq, sources, reviewed):
    if eq.label not in sources:
        return [f"no display equation with \\label{{{eq.label}}} in ms.tex"]
    if sources[eq.label] is None:
        return [f"\\label{{{eq.label}}} shares its environment with other labels, "
                "which check_equations.py does not split"]
    if eq.label not in reviewed:
        return ["never reviewed: check the spec against ms.tex, then run "
                f"check_equations.py --accept {eq.label}"]
    if not same_latex(sources[eq.label], reviewed[eq.label]):
        diff = difflib.unified_diff(reviewed[eq.label].splitlines(), sources[eq.label].splitlines(),
                                    "reviewed", "ms.tex", lineterm="")
        return ["the LaTeX in ms.tex changed since the spec was checked against it. Update the "
                f"spec in equations.py, then run check_equations.py --accept {eq.label}\n"
                + textwrap.indent("\n".join(diff), "    ")]
    return []


# ── derivation ────────────────────────────────────────────────────────────────

def check_derivation(eq):
    nb_path = paths.src / "derivations" / eq.derivation
    if not nb_path.exists():
        return [f"derivation notebook {eq.derivation} not found"]
    code = "\n".join("".join(c["source"]) for c in json.loads(nb_path.read_text())["cells"]
                     if c["cell_type"] == "code")
    if not re.search(rf"""get\(\s*["']{re.escape(eq.label)}["']\s*\)""", code):
        return [f"{eq.derivation} does not take its spec from the registry "
                f"(no get(\"{eq.label}\") in a code cell)"]
    return []


# ── spotgp ────────────────────────────────────────────────────────────────────

def breakpoints(expr, var):
    """Values of var at which a piecewise condition in expr changes."""
    points = []
    for pw in expr.atoms(sp.Piecewise):
        for _, cond in pw.args:
            for rel in cond.atoms(sp.core.relational.Relational):
                if var in rel.free_symbols:
                    points += sp.solve(sp.Eq(rel.lhs, rel.rhs), var)
    return points


def bounds(eq, symbol, params):
    return [float(sp.sympify(b).subs(params)) for b in eq.domain[symbol]]


def parameter_draws(eq, rng):
    draws = []
    for _ in range(N_DRAWS):
        params = {}
        for s in list(eq.domain)[:-1]:
            lo, hi = bounds(eq, s, params)
            params[s] = int(rng.integers(lo, hi + 1)) if s.is_integer else rng.uniform(lo, hi)
        draws.append(params)
    return draws + [{s: int(v) if s.is_integer else float(v) for s, v in p.items()} for p in eq.extra_params]


def sweep(eq, var, params, rng):
    lo, hi = bounds(eq, var, params)
    xs = [*rng.uniform(lo, hi, N_SWEEP), lo, hi]
    for b in breakpoints(eq.expr.subs(eq.given), var):
        b = float(b.subs(params))
        if lo <= b <= hi:
            d = EPS * max(abs(b), 1.0)
            xs += [b - d, b, b + d]
    return sorted(xs)


def check_impl(eq, rng):
    """Failures, and the largest value and gradient errors relative to their sweep's scale."""
    expr = eq.expr.subs(eq.given)
    if not expr.free_symbols <= set(eq.args):
        return [f"spec has symbols {expr.free_symbols - set(eq.args)} not passed to {eq.impl}"], None
    module, name = eq.impl.split(":")
    fn = functools.reduce(getattr, name.split("."), importlib.import_module(module))
    # differentiate with respect to the continuous arguments only
    diff_args = [i for i, a in enumerate(eq.args) if not a.is_integer]
    grad_fn = jax.jit(jax.grad(lambda *a: jnp.sum(fn(*a)), argnums=tuple(diff_args)))

    spec = sp.lambdify(eq.args, expr, "mpmath")
    spec_grad = [sp.lambdify(eq.args, sp.diff(expr, eq.args[i]), "mpmath") for i in diff_args]
    piecewise = list(expr.atoms(sp.Piecewise))
    pieces = [[sp.lambdify(eq.args, cond, "mpmath") for _, cond in pw.args] for pw in piecewise]
    reached = [set() for _ in pieces]

    var = list(eq.domain)[-1]
    assert not var.is_integer, f"{eq.label}: the swept variable {var} must be continuous"
    failures, worst_val, worst_grad = [], 0.0, 0.0
    for params in parameter_draws(eq, rng):
        rows = []
        for x in sweep(eq, var, params, rng):
            point = {**params, var: x}
            vals = [point[a] for a in eq.args]
            mp_vals = [mpmath.mpf(v) for v in vals]
            for k, conds in enumerate(pieces):
                reached[k].add(next(i for i, c in enumerate(conds) if c(*mp_vals)))
            rows.append((point,
                         float(spec(*mp_vals)), float(np.ravel(fn(*vals))[0]),
                         np.array([float(g(*mp_vals)) for g in spec_grad]),
                         np.array([float(g) for g in grad_fn(*vals)])))

        val_scale = max(abs(r[1]) for r in rows)
        grad_scale = np.max([np.abs(r[3]) for r in rows], axis=0)
        for point, v_spec, v_impl, g_spec, g_impl in rows:
            e_val = abs(v_impl - v_spec) / val_scale if val_scale else abs(v_impl - v_spec)
            e_grad = np.max(np.abs(g_impl - g_spec) / np.where(grad_scale > 0, grad_scale, 1.0))
            worst_val, worst_grad = max(worst_val, e_val), max(worst_grad, e_grad)
            if not (e_val <= eq.rtol and e_grad <= eq.rtol):  # also catches nan
                where = ", ".join(f"{s}={v:.12g}" for s, v in point.items())
                failures.append(f"at {where}: spec {v_spec:.12g}, spotgp {v_impl:.12g} "
                                f"(error {e_val:.1e}); gradient error {e_grad:.1e}")

    for pw, hit in zip(piecewise, reached):
        for i in sorted(set(range(len(pw.args))) - hit):
            failures.append(f"the domain never reaches branch {i} of the spec ({pw.args[i][1]})")
    if len(failures) > 5:
        failures = failures[:5] + [f"... and {len(failures) - 5} more"]
    return failures, (worst_val, worst_grad)


# ── report ────────────────────────────────────────────────────────────────────

def sci(x):
    if x == 0:
        return "0"
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def table(results, n_labels):
    tt = lambda s: r"\texttt{" + s.replace("_", r"\_") + "}"
    rows = []
    for eq, err in results:
        nb = rf"\href{{\GitHubURL/blob/\GitHubSHA/src/derivations/{eq.derivation}}}{{{tt(eq.derivation)}}}"
        impl = tt(eq.impl.replace("spotgp.", "", 1).replace(":", ".")) if eq.impl else "--"
        errs = f"{sci(err[0])} & {sci(err[1])}" if err else "-- & --"
        rows.append(rf"Eq.~\eqref{{{eq.label}}} & {nb} & {impl} & {errs} \\")
    return "\n".join([
        f"% Written by check_equations.py: {len(results)} of {n_labels} labelled equations "
        "in ms.tex are in the equation registry.",
        r"\begin{tabular}{lllcc}",
        r"\hline",
        r"Equation & Derivation & \texttt{spotgp} implementation & Value error & Gradient error \\",
        r"\hline",
        *rows,
        r"\hline",
        r"\end{tabular}",
        ""])


def icons(results):
    """ms.tex draws \\EquationCheckSymbol in the margin at every \\label that has a <label>_check value."""
    return "\n".join([
        "% Written by check_equations.py, only when every equation in the registry passes its checks.",
        "% ms.tex draws \\EquationCheckSymbol in the margin at each \\label with a <label>_check value,",
        "% linking to that file.",
        r"\providecommand\addvalue[2]{}",
        *(rf"\addvalue{{{eq.label}_check}}{{src/derivations/{eq.derivation}}}" for eq, _ in results),
        ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", nargs="?", type=Path, help="LaTeX table to write if every check passes")
    ap.add_argument("icons", nargs="?", type=Path, help="margin icons to write if every check passes")
    ap.add_argument("--accept", nargs="+", metavar="LABEL",
                    help="record the current ms.tex LaTeX of these equations as reviewed, and exit")
    args = ap.parse_args()

    sources = equation_sources(MS.read_text())
    if args.accept:
        accept(args.accept, sources)
        return

    reviewed = read_reviewed()
    rng = np.random.default_rng(0)
    failed, results = False, []
    if jnp.asarray(1.0).dtype != jnp.float64:
        print("FAIL  jax is running in float32; spotgp should enable jax_enable_x64 on import")
        failed = True
    for eq in equations.EQUATIONS:
        checks = {"manuscript": check_manuscript(eq, sources, reviewed),
                  "derivation": check_derivation(eq)}
        err = None
        if eq.impl:
            checks["spotgp"], err = check_impl(eq, rng)
        for name, failures in checks.items():
            status = "FAIL" if failures else "ok  "
            detail = f"  (value {err[0]:.1e}, gradient {err[1]:.1e})" if name == "spotgp" and err else ""
            print(f"{status}  {eq.label:24s} {name}{detail}")
            for f in failures:
                print(textwrap.indent(f, "        "))
            failed |= bool(failures)
        results.append((eq, err))

    print(f"\n{len(results)} of {len(sources)} labelled equations in ms.tex are in the registry.")
    if failed:
        sys.exit("Equation checks failed.")
    for path, text in [(args.table, table(results, len(sources))), (args.icons, icons(results))]:
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)


if __name__ == "__main__":
    main()
