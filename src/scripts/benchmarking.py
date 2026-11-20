"""
Helpers shared by the hand-run benchmark scripts (compute_benchmark_*.py): running on
one CPU core or on the GPU, timing, peak memory, worker processes, and the provenance
saved with each result (machine, package versions, and the spotgp commit, which must be
the one environment.yml pins).

Import this module, and call use_device() (or use_cpu()) before importing jax or paths
(which imports spotgp, and so jax).
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


def use_cpu(core=None):
    """Run this process, and the worker processes it starts, on CPU JAX on one core:
    core BENCHMARK_CPU_CORE (default 0), so that CPU benchmarks can run side by side."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    if core is None:
        core = int(os.environ.get("BENCHMARK_CPU_CORE", 0))
    os.sched_setaffinity(0, {core})


def use_gpu():
    """Run this process, and the worker processes it starts, on the GPU. Memory is
    allocated on demand, so that running out of GPU memory shows as RESOURCE_EXHAUSTED
    rather than as a failed autotuning of whatever kernel comes next. Set through the
    environment, not jax.config: spotgp's gp_solver switches JAX to the CPU on import
    unless JAX_PLATFORMS is set."""
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def use_device(device):
    """use_cpu() for device "cpu", use_gpu() for device "gpu"."""
    {"cpu": use_cpu, "gpu": use_gpu}[device]()


def device_name(device):
    """The hardware a device's timings are measured on, as saved in the provenance."""
    if device == "cpu":
        return f"1 core of {cpu_name()}"
    import jax
    return jax.devices("cuda")[0].device_kind


def median_time(func, n_repeats):
    """Median wall-clock time of func() over n_repeats calls [s]. Garbage is collected
    after each call, outside the timing: a GPSolver holds closures that refer to it, so
    only the cycle collector frees it, and without this the solvers of earlier calls
    would pile up in memory (on the GPU, until it runs out)."""
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
        gc.collect()
    return float(np.median(times))


def _rss_mb():
    with open(f"/proc/{os.getpid()}/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


class PeakRSS:
    """Peak increase of this process's resident memory inside a with block [MB],
    polled every 5 ms from a background thread."""

    interval = 0.005

    def _used(self):
        return _rss_mb()

    def __enter__(self):
        gc.collect()
        self._baseline = self._peak = self._used()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        while not self._stop.wait(self.interval):
            self._peak = max(self._peak, self._used())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._used())

    @property
    def mb(self):
        return max(0.0, self._peak - self._baseline)


class PeakGPUMemory(PeakRSS):
    """Peak increase of the GPU memory in use inside a with block [MB]: exact when the
    allocator's peak rises inside the block, and otherwise (a higher peak came before
    it) polled every 1 ms from a background thread."""

    interval = 0.001

    def __init__(self):
        import jax
        self._device = jax.devices("cuda")[0]

    def _used(self):
        return self._device.memory_stats()["bytes_in_use"] / 1e6

    def __enter__(self):
        self._peak_before = self._device.memory_stats()["peak_bytes_in_use"]
        return super().__enter__()

    def __exit__(self, *exc):
        super().__exit__(*exc)
        peak = self._device.memory_stats()["peak_bytes_in_use"]
        if peak > self._peak_before:
            self._peak = max(self._peak, peak / 1e6)


def peak_memory(device):
    """PeakRSS on the CPU, PeakGPUMemory on the GPU."""
    return PeakRSS() if device == "cpu" else PeakGPUMemory()


def run_worker(script, *args, timeout=3600):
    """Run `python script args...` in a fresh process and return the JSON it prints on
    its last line. A fresh process per measurement keeps the JIT caches and the
    peak-memory measurements of different runs independent."""
    result = subprocess.run([sys.executable, str(script), *map(str, args)],
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
