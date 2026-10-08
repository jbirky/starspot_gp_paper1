"""
Exposes common paths useful for manipulating datasets and generating figures.

"""
import sys
import importlib
from pathlib import Path

import spotgp
for _name in [
    "analytic_kernel", "banded_cholesky", "contrast", "distributions",
    "envelope", "gp_solver", "latitude", "lightcurve", "mcmc", "multiband",
    "numerical_kernel", "observations", "params", "plotting", "psd",
    "sensitivity", "spectral", "spot_model", "transit", "visibility",
]:
    try:
        _mod = importlib.import_module(f"spotgp.{_name}")
    except ModuleNotFoundError:
        continue
    sys.modules[_name] = _mod
sys.modules["starspot"] = sys.modules["lightcurve"]

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[2].absolute()

# Absolute path to the `src` folder
src = root / "src"

# Absolute path to the `src/data` folder (contains datasets)
data = src / "data"

# Absolute path to the `src/static` folder (contains static images)
static = src / "static"

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
scripts = src / "scripts"

# Absolute path to the `src/tex` folder (contains the manuscript)
tex = src / "tex"

# Absolute path to the `src/tex/figures` folder (contains figure output)
figures = tex / "figures"

# Absolute path to the `src/tex/output` folder (contains other user-defined output)
output = tex / "output"
