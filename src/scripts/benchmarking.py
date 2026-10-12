"""
Helpers shared by the hand-run benchmark scripts (compute_benchmark_*.py): running on
one CPU core, timing, peak memory, worker processes, and the provenance saved with
each result (machine, package versions, and the spotgp commit, which must be the one
environment.yml pins).

Import this module, and call use_cpu() if the benchmark runs on the CPU, before
importing jax or paths (which imports spotgp, and so jax).
"""
import gc
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def use_cpu(core=0):
    """Run this process, and the worker processes it starts, on CPU JAX on one core."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.sched_setaffinity(0, {core})


def median_time(func, n_repeats):
    """Median wall-clock time of func() over n_repeats calls [s]."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _rss_mb():
    with open(f"/proc/{os.getpid()}/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


class PeakRSS:
    """Peak increase of this process's resident memory inside a with block [MB],
    polled every 5 ms from a background thread."""

    def __enter__(self):
        gc.collect()
        self._baseline = self._peak = _rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(0.005):
            self._peak = max(self._peak, _rss_mb())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, _rss_mb())

    @property
    def mb(self):
        return max(0.0, self._peak - self._baseline)


def run_worker(script, *args, timeout=3600):
    """Run `python script --worker args...` in a fresh process and return the JSON it
    prints on its last line. A fresh process per measurement keeps the JIT caches and
    the peak-memory measurements of different runs independent."""
    result = subprocess.run([sys.executable, str(script), "--worker", *map(str, args)],
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"worker {' '.join(map(str, args))} failed:\n{result.stderr[-3000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def cpu_name():
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def _git(path, *args):
    result = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _commit(path):
    """HEAD of the git checkout at path, with -dirty if it has uncommitted changes."""
    head = _git(path, "rev-parse", "HEAD")
    return head and head + ("-dirty" if _git(path, "status", "--porcelain") else "")


def spotgp_commit():
    """Commit of the spotgp in use: from the install record of a pip install from git,
    or from the git checkout it was imported from (e.g. one put on PYTHONPATH)."""
    import spotgp
    pkg = Path(spotgp.__file__).resolve().parent
    try:
        dist = metadata.distribution("spotgp")
        if Path(dist.locate_file("spotgp")).resolve() == pkg:
            url = json.loads(dist.read_text("direct_url.json") or "{}")
            if "vcs_info" in url:
                return url["vcs_info"]["commit_id"]
    except metadata.PackageNotFoundError:
        pass
    # only a checkout whose top level holds the package, not a repo it happens to sit in
    top = _git(pkg, "rev-parse", "--show-toplevel")
    return _commit(pkg.parent) if top and Path(top).resolve() == pkg.parent else None


def pinned_spotgp_commit():
    return re.search(r"spotgp\.git@([0-9a-f]{40})", (ROOT / "environment.yml").read_text())[1]


def check_spotgp_pin():
    """Stop unless the spotgp in use is the commit environment.yml pins."""
    import spotgp
    used, pinned = spotgp_commit(), pinned_spotgp_commit()
    if used != pinned:
        sys.exit(f"spotgp in use ({spotgp.__file__}) is at {used or 'an unknown commit'}, "
                 f"but environment.yml pins {pinned}. Run with the pinned spotgp: the "
                 f"showyourwork env, or PYTHONPATH=<checkout of {pinned[:7]}>.")


def provenance(device):
    """What the timings were measured on and with."""
    import jax
    import spotgp
    return dict(
        date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        script=Path(sys.argv[0]).name,
        device=device,
        cpu=cpu_name(),
        python=platform.python_version(),
        spotgp=spotgp.__version__,
        spotgp_commit=spotgp_commit(),
        jax=jax.__version__,
        jaxlib=metadata.version("jaxlib"),
        numpy=np.__version__,
        paper_commit=_commit(ROOT),
    )


def save(path, params, rows, device):
    """Write a benchmark's settings, results and provenance as JSON."""
    path.write_text(json.dumps(dict(params=params, rows=rows, provenance=provenance(device)),
                               indent=2) + "\n")
    print(f"Saved {path}")
