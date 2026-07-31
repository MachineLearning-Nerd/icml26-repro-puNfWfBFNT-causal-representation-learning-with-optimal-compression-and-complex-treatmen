"""Size BLAS/OpenMP thread pools to the CONTAINER's CPU quota, not the host's core count.

Why this exists
---------------
Runs of this suite on Hugging Face `cpu-upgrade` were 20-40x slower than the same code on the
development machine: a single K=4 fit took 26s locally but had not finished after ~9 minutes
in the job.  Three runs were cancelled before the cause was found.

The cause is thread oversubscription.  torch, OpenMP and MKL size their pools from
`os.cpu_count()`, which inside a container reports the *host's* core count -- often 32-64 --
while the cgroup quota grants only a handful of vCPUs.  Every parallel region then spawns far
more threads than there are runnable cores, and the resulting contention and spin-waiting
dominate the actual arithmetic.  It is worst for exactly this workload: many small ops
(1500x64 matmuls, 1500x1500 kernels) where per-op thread launch overhead is already
comparable to the useful work.

`configure()` reads the real quota from cgroup v2 (`cpu.max`) or v1
(`cpu.cfs_quota_us`/`cpu.cfs_period_us`), falls back to `os.cpu_count()`, and pins every pool
to it.  The environment variables MUST be set before torch/numpy are imported, since both
read them at import time -- hence this module is imported first thing in verify_all.py.
"""
from __future__ import annotations

import os


def detect_cpu_quota() -> int:
    """Effective CPU count for this process: cgroup quota if present, else os.cpu_count()."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        if quota > 0 and period > 0:
            return max(1, quota // period)
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))   # respects CPU pinning
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def configure() -> dict:
    """Pin thread pools to the CPU quota. Call BEFORE importing torch or numpy."""
    n = detect_cpu_quota()
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)
    info = {"cpu_quota": n, "os_cpu_count": os.cpu_count()}
    try:
        import torch

        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
        info["torch_threads"] = torch.get_num_threads()
        info["torch_version"] = torch.__version__
    except Exception as e:                      # torch may not be imported yet; harmless
        info["torch"] = f"not configured: {e}"
    return info
