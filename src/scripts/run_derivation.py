"""
Execute a derivation notebook and, if every cell runs, write a LaTeX link to it.

The notebooks assert their results against the equations typeset in ms.tex, so a
failed check fails the build. ms.tex includes the link with \variable, which makes
showyourwork re-run the notebook whenever it changes.

Usage:
    python run_derivation.py src/derivations/<name>.ipynb src/tex/output/derivations/<name>.tex
"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

nb_path, tex_path = Path(sys.argv[1]), Path(sys.argv[2])

nb = nbformat.read(nb_path, as_version=4)
# raises CellExecutionError on the first failing cell, including a failed assert
NotebookClient(nb, kernel_name="python3", timeout=600,
               resources={"metadata": {"path": str(nb_path.parent)}}).execute()

# \GitHubURL and \GitHubSHA are defined by showyourwork, so the link points at the
# notebook as of the commit the PDF was built from
name = nb_path.name.replace("_", r"\_")
tex_path.parent.mkdir(parents=True, exist_ok=True)
tex_path.write_text(
    rf"\href{{\GitHubURL/blob/\GitHubSHA/{nb_path.as_posix()}}}{{\texttt{{{name}}}}}%" + "\n")
